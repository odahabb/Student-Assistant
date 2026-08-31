"""
backend/service.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Everything the web interface does, independent of how it is served.

The student organises material into subjects. Each subject is a folder under
data/projects holding any number of documents, and a question is answered
against everything in the selected subject at once — one combined index per
subject, not per file. Quiz questions, progress and the chat history are kept
in <subject>/_study/.

A thin layer over the pipeline, not a reimplementation:

    loader.load_file -> preprocessor.preprocess -> embedder.embed
      -> FAISS index (+ BM25) -> retriever.retrieve -> generator.stream

backend/api.py exposes these functions over HTTP. The functions here hold no
web-specific code, so they are tested directly as well as through the API.

Two pieces of shared state matter:
  - indexes are built in a background thread, and callers poll index_status();
  - the models are not safe to run concurrently on one device, so every model
    call goes through MODEL_LOCK.
"""

import json
import logging
import os
import random
import re
import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional

# CPU unless told otherwise, matching the pipeline's demo path. Set before
# importing any pipeline module: device.py reads this when models are built.
os.environ.setdefault("SA_DEVICE", "cpu")
# bge-small-en-v1.5 answered 18/25 evaluation questions end to end against 12
# for all-MiniLM-L6-v2 (data/eval/generation_analysis_bge-small.json).
os.environ.setdefault("SA_EMBEDDER", "bge-small")
# The chat view explains in a paragraph; the quiz still uses short answers
# (see generator.ANSWER_STYLE).
os.environ.setdefault("SA_ANSWER_STYLE", "explain")

import faiss
import numpy as np

from backend.pipeline import quiz, sparse
from backend.pipeline.embedder import embed, model_key
from backend.pipeline.loader import EXTENSION_MAP, load_file
from backend.pipeline.preprocessor import preprocess
from backend.pipeline.recommender import Progress
from backend.pipeline.retriever import retrieve

ROOT = Path(__file__).resolve().parents[1]
# SA_PROJECTS_DIR lets tests run against a temporary folder.
PROJECTS_DIR = Path(os.environ.get("SA_PROJECTS_DIR", ROOT / "data" / "projects"))
SAMPLE_DIR = ROOT / "data" / "raw"

SUPPORTED_EXTENSIONS = sorted({ext.lstrip(".") for ext in EXTENSION_MAP})
# File types that go through a vision or speech model before chunking.
SLOW_EXTENSIONS = {ext for ext in EXTENSION_MAP if ext not in (".pdf", ".txt")}

# Retrieval depth is a development-time setting, not a user-facing control.
TOP_K = 3
# Sentence-aware chunks with hybrid retrieval answered 20/25 evaluation
# questions end to end, the best of the configurations tested (data/eval/).
CHUNKING = "sentence"
# Hybrid retrieval: BM25 keyword scores mixed with the embeddings.
HYBRID = True
QUESTIONS_PER_TOPIC = 2
# Read the pictures on pages whose text layer is thin or empty — diagrams,
# charts, screenshots, and slides exported as images. Off in the library
# (loader.load_pdf), on here, because a student's slides are largely pictures.
FIGURES = "auto"
STUDY_DIR = "_study"
# Above this many passages from one file, the interface says so: a single
# huge upload is slow to index and outweighs everything else in retrieval.
LARGE_DOCUMENT_CHUNKS = 3000
NO_ANSWER = "I couldn't find an answer to that in this subject's materials."

MODEL_LOCK = threading.Lock()

log = logging.getLogger(__name__)

_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class NotFound(LookupError):
    """A subject, document or question that doesn't exist."""


# Subjects and documents on disk

# A folder that could not be removed outright is renamed with this prefix and
# swept later; it is not a subject any more (see delete_subject).
DELETED_PREFIX = ".deleted-"


def subject_names() -> List[str]:
    if not PROJECTS_DIR.is_dir():
        return []
    return sorted((p.name for p in PROJECTS_DIR.iterdir()
                   if p.is_dir() and not p.name.startswith(".")),
                  key=str.lower)


def subject_path(name: str) -> Path:
    """The folder for an existing subject; refuses anything outside PROJECTS_DIR."""
    path = PROJECTS_DIR / name
    if (name in ("", ".", "..") or _ILLEGAL_NAME_CHARS.search(name)
            or not path.is_dir()):
        raise NotFound(f"No subject called {name!r}")
    return path


def create_subject(name: str) -> str:
    safe = _ILLEGAL_NAME_CHARS.sub("", name).strip().strip(".")
    if not safe:
        raise ValueError("That name can't be used as a folder name.")
    (PROJECTS_DIR / safe).mkdir(parents=True, exist_ok=True)
    return safe


def documents(name: str) -> List[Path]:
    """Files in a subject the pipeline knows how to read."""
    return sorted((p for p in subject_path(name).iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSION_MAP),
                  key=lambda p: p.name.lower())


def document_path(name: str, filename: str) -> Path:
    for path in documents(name):
        if path.name == filename:
            return path
    raise NotFound(f"No document called {filename!r}")


def signature(name: str):
    """
    Identity of a subject's document set. Adding, replacing or removing a
    document changes it, and so rebuilds the index; nothing else does.
    """
    return tuple((p.name, p.stat().st_mtime, p.stat().st_size)
                 for p in documents(name))


def add_document(name: str, filename: str, data: bytes) -> bool:
    """Save an upload into a subject. False if the name exists already."""
    filename = Path(filename).name
    if Path(filename).suffix.lower() not in EXTENSION_MAP:
        raise ValueError(f"{filename} isn't a supported file type.")
    destination = subject_path(name) / filename
    if destination.exists():
        return False
    destination.write_bytes(data)
    return True


def remove_document(name: str, filename: str) -> None:
    document_path(name, filename).unlink()


def add_samples(name: str) -> int:
    added = 0
    folder = subject_path(name)
    if SAMPLE_DIR.is_dir():
        for path in sorted(SAMPLE_DIR.iterdir()):
            if path.is_file() and path.suffix.lower() in EXTENSION_MAP:
                if not (folder / path.name).exists():
                    shutil.copy2(path, folder / path.name)
                    added += 1
    return added


def study_path(name: str, filename: str) -> Path:
    return subject_path(name) / STUDY_DIR / filename


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


# Indexing

@dataclass
class SubjectIndex:
    signature: tuple
    chunks: list
    vectors: Optional[faiss.Index]
    keywords: Optional[sparse.BM25]
    per_document: List[dict] = field(default_factory=list)
    failures: List[dict] = field(default_factory=list)
    seconds: float = 0.0

    def find(self, query: str) -> list:
        """The TOP_K chunks for a query, hybrid when a keyword index exists."""
        return retrieve(query, self.vectors, self.chunks, k=TOP_K,
                        sparse=self.keywords)


_indexes: Dict[str, SubjectIndex] = {}
_building: Dict[str, dict] = {}
_state_lock = threading.Lock()


def build_index(name: str, sig, report=lambda **_: None,
                figures: str = "off") -> SubjectIndex:
    """
    Ingest every document in a subject into one combined index.

    The index is held in memory rather than written through vector_store's
    single fixed path, since several subjects coexist and would otherwise
    overwrite each other's store. Chunking, embedding and retrieval are
    unchanged; only where the index lives differs.

    figures="auto" also reads the pictures on pages with little or no text,
    which is slow; _build_in_background does that as a second pass so the
    subject can be asked questions in the meantime.
    """
    started = time.time()
    paths = documents(name)
    chunks, per_document, failures = [], [], []
    for done, path in enumerate(paths):
        report(done=done, total=len(paths), current=path.name,
               stage="reading", pictures=None)

        def picture_progress(read, total_pictures, page):
            """Reading pictures is the slow part; show it page by page."""
            report(done=done, total=len(paths), current=path.name,
                   stage="pictures",
                   pictures={"done": read, "total": total_pictures, "page": page})

        try:
            if figures == "auto" and path.suffix.lower() == ".pdf":
                # The picture pass takes minutes, so it takes the model lock
                # one page at a time; a question asked meanwhile waits for a
                # page, not for the whole document.
                loaded = load_file(str(path), figures=figures,
                                   report=picture_progress, lock=MODEL_LOCK)
            else:
                with MODEL_LOCK:
                    loaded = load_file(str(path), figures=figures,
                                       report=picture_progress)
            with MODEL_LOCK:
                file_chunks = preprocess(loaded, source_file=path.name,
                                         chunking=CHUNKING)
        except Exception as exc:  # a bad upload shouldn't sink the subject
            # One readable line: some libraries raise pages of diagnostics.
            reason = " ".join(str(exc).split())[:200] or exc.__class__.__name__
            failures.append({"name": path.name, "error": reason})
            continue
        chunks.extend(file_chunks)
        pages = [c.page for c in file_chunks if c.page is not None]
        entry = {"name": path.name, "chunks": len(file_chunks),
                 "pages": max(pages) if pages else None}
        if len(file_chunks) > LARGE_DOCUMENT_CHUNKS:
            # Indexed in full, but the student should know why this upload
            # took minutes and why it dominates the subject's answers.
            entry["note"] = (f"very large — {len(file_chunks)} passages, which "
                             f"may crowd out your other documents")
        per_document.append(entry)

    vectors = keywords = None
    if chunks:
        report(done=len(paths), total=len(paths), current=None,
               stage="embedding", pictures=None)
        with MODEL_LOCK:
            embeddings = embed(chunks)
        vectors = faiss.IndexFlatL2(embeddings.shape[1])
        vectors.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        keywords = sparse.build_index(chunks) if HYBRID else None
    return SubjectIndex(sig, chunks, vectors, keywords, per_document, failures,
                        time.time() - started)


def _build_in_background(name: str, sig) -> None:
    """
    Build a subject's index in two passes.

    The first pass reads text only and takes seconds, and the subject can be
    asked questions as soon as it lands. The second pass reads the pictures on
    pages whose text layer is thin — minutes on a deck of diagrams — and
    replaces the index when it finishes. Waiting for the pictures before
    answering anything would mean a student uploading a term's slides could
    not ask a question for the best part of an hour.
    """
    job = _building[name]

    def report(**fields):
        job.update(fields)

    def publish(index) -> bool:
        """
        Store the finished index, unless the documents changed while it was
        being built — a stale index would answer from files the student has
        already replaced.
        """
        with _state_lock:
            if _building.get(name) is not job:
                return False           # a newer build has taken over
            if signature(name) != sig:
                _building.pop(name, None)
                return False
            _indexes[name] = index
            return True

    try:
        index = build_index(name, sig, report, figures="off")
        if not publish(index):
            return
        if FIGURES != "auto" or not index.chunks:
            with _state_lock:
                if _building.get(name) is job:
                    _building.pop(name, None)
            return
        job.update(state="enriching", stage="pictures", done=0,
                   current=None, pictures=None)

        enriched = build_index(name, sig, report, figures=FIGURES)
        if publish(enriched):
            with _state_lock:
                if _building.get(name) is job:
                    _building.pop(name, None)
    except Exception as exc:
        with _state_lock:
            if _building.get(name) is job:
                job.update(state="error", error=str(exc))


def index_status(name: str, start: bool = True) -> dict:
    """
    Where a subject's index stands: "empty" (no documents), "indexing",
    "ready" or "error". Starts a build when the documents changed since the
    last one, unless start is False.
    """
    sig = signature(name)
    slow = [n for n, *_ in sig if Path(n).suffix.lower() in SLOW_EXTENSIONS]
    with _state_lock:
        index = _indexes.get(name)
        job = _building.get(name)
        if not sig:
            return {"state": "empty"}
        if index is not None and index.signature == sig:
            # Nothing readable came out of any document: the subject cannot
            # answer anything, and saying "ready" would invite a question that
            # crashes on an empty index.
            state = "ready" if index.chunks else "unreadable"
            status = {"state": state, "chunks": len(index.chunks),
                      "documents": index.per_document, "failures": index.failures,
                      "seconds": round(index.seconds, 1)}
            if job is not None and job.get("state") == "enriching":
                # Answers work already; the pictures are still being read.
                status["enriching"] = {"current": job.get("current"),
                                       "pictures": job.get("pictures"),
                                       "done": job.get("done"),
                                       "total": job.get("total")}
            return status
        if job is not None and job["signature"] == sig:
            status = {k: v for k, v in job.items() if k != "signature"}
            status["slow"] = slow
            return status
        # Any job left here is for an older set of documents. Its own publish
        # step will see that and drop its result, so a fresh build starts now.
        if not start:
            return {"state": "stale"}
        job = {"state": "indexing", "signature": sig, "done": 0,
               "total": len(sig), "current": None, "stage": "queued",
               "pictures": None}
        _building[name] = job
    threading.Thread(target=_build_in_background, args=(name, sig),
                     daemon=True).start()
    return {**{k: v for k, v in job.items() if k != "signature"}, "slow": slow}


def ready_index(name: str) -> SubjectIndex:
    """The subject's index if it is up to date, else raise IndexNotReady."""
    status = index_status(name)
    if status["state"] != "ready":
        raise IndexNotReady(status)
    return _indexes[name]


class IndexNotReady(RuntimeError):
    def __init__(self, status: dict):
        super().__init__(f"Index is {status['state']}")
        self.status = status


def forget(name: str) -> None:
    with _state_lock:
        _indexes.pop(name, None)


# Asking

def describe(chunk) -> dict:
    return {"file": getattr(chunk, "source_file", None),
            "page": getattr(chunk, "page", None),
            "page_end": getattr(chunk, "page_end", None),
            "pages": getattr(chunk, "pages", None),
            "kind": getattr(chunk, "kind", "page"),
            "timecode": getattr(chunk, "timecode", None),
            "from_image": bool(getattr(chunk, "from_image", False)),
            "section": getattr(chunk, "section", None),
            "text": str(chunk)}


# Conversations
#
# A subject holds any number of conversations, one JSON file each under
# <subject>/_study/chats/. A student revising a term's material asks about
# several things, and keeping those threads apart is the difference between a
# record they can come back to and one long scroll.

CHATS_DIR = "chats"
TITLE_MAX_WORDS = 8
TITLE_MAX_CHARS = 60


def _chats_dir(name: str) -> Path:
    return subject_path(name) / STUDY_DIR / CHATS_DIR


def _chat_path(name: str, chat_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{8,32}", chat_id or ""):
        raise NotFound(f"No conversation called {chat_id!r}")
    path = _chats_dir(name) / f"{chat_id}.json"
    if not path.exists():
        raise NotFound(f"No conversation called {chat_id!r}")
    return path


def chat_title(question: str) -> str:
    """
    A conversation's title, taken from the question that started it: enough of
    it to recognise the thread in a list, without a paragraph in the sidebar.
    """
    words = question.strip().split()
    title = " ".join(words[:TITLE_MAX_WORDS])
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS].rsplit(" ", 1)[0]
    if len(words) > TITLE_MAX_WORDS or len(title) < len(question.strip()):
        title = title.rstrip(" ,.;:") + "…"
    return title or "New conversation"


def _migrate_single_chat(name: str) -> None:
    """
    Move the one-chat-per-subject file written by earlier versions into the
    conversations folder, so an existing subject keeps its history.
    """
    legacy = study_path(name, "chat.json")
    if not legacy.exists():
        return
    try:
        messages = json.loads(legacy.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        messages = []
    if messages:
        first = next((m["content"] for m in messages if m.get("role") == "user"), "")
        record = {"id": uuid.uuid4().hex, "title": chat_title(first),
                  "created": messages[0].get("time", time.time()),
                  "updated": messages[-1].get("time", time.time()),
                  "messages": messages}
        _write_json(_chats_dir(name) / f"{record['id']}.json", record)
    legacy.unlink(missing_ok=True)


def chat_list(name: str) -> List[dict]:
    """Every conversation in a subject, most recently used first."""
    _migrate_single_chat(name)
    folder = _chats_dir(name)
    if not folder.is_dir():
        return []
    chats = []
    for path in folder.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):     # a half-written file must not hide the rest
            continue
        chats.append({"id": record.get("id", path.stem),
                      "title": record.get("title") or "New conversation",
                      "created": record.get("created"),
                      "updated": record.get("updated"),
                      "exchanges": sum(1 for m in record.get("messages", [])
                                       if m.get("role") == "user")})
    return sorted(chats, key=lambda c: c.get("updated") or 0, reverse=True)


def subject_overview(name: str) -> dict:
    """
    What the subjects grid shows for one subject: its documents, how many
    conversations it holds, when it was last used, and whether its index is
    already built. Deliberately cheap — listing subjects must not start
    indexing every one of them.
    """
    docs = documents(name)
    chats = chat_list(name)
    times = [c["updated"] for c in chats if c.get("updated")]
    times += [p.stat().st_mtime for p in docs]
    status = index_status(name, start=False)
    return {
        "name": name,
        "documents": [{"name": p.name, "size": p.stat().st_size,
                       "type": p.suffix.lower().lstrip(".")} for p in docs],
        "chats": len(chats),
        "updated": max(times) if times else None,
        "state": status.get("state"),
        "chunks": status.get("chunks"),
    }


def recent_chats(limit: int = 12) -> List[dict]:
    """The most recently used conversations across every subject."""
    chats = []
    for name in subject_names():
        for record in chat_list(name):
            chats.append({**record, "subject": name})
    chats.sort(key=lambda c: c.get("updated") or 0, reverse=True)
    return chats[:limit]


def create_chat(name: str) -> dict:
    """Start an empty conversation; it takes its title from the first question."""
    subject_path(name)
    record = {"id": uuid.uuid4().hex, "title": "New conversation",
              "created": time.time(), "updated": time.time(), "messages": []}
    _write_json(_chats_dir(name) / f"{record['id']}.json", record)
    return {k: v for k, v in record.items() if k != "messages"} | {"exchanges": 0}


def chat(name: str, chat_id: str) -> dict:
    """One conversation, with its messages."""
    _migrate_single_chat(name)
    record = json.loads(_chat_path(name, chat_id).read_text(encoding="utf-8"))
    record.setdefault("messages", [])
    return record


def chat_history(name: str, chat_id: Optional[str] = None) -> List[dict]:
    """The messages of one conversation, or of the most recent one."""
    if chat_id is None:
        chats = chat_list(name)
        if not chats:
            return []
        chat_id = chats[0]["id"]
    return chat(name, chat_id)["messages"]


def delete_chat(name: str, chat_id: str) -> None:
    _chat_path(name, chat_id).unlink()


def clear_chat(name: str) -> None:
    """Delete every conversation in a subject."""
    _migrate_single_chat(name)
    folder = _chats_dir(name)
    if folder.is_dir():
        for path in folder.glob("*.json"):
            path.unlink(missing_ok=True)


def _sweep_deleted() -> None:
    """Remove folders left behind by an earlier delete that could not finish."""
    if not PROJECTS_DIR.is_dir():
        return
    for path in PROJECTS_DIR.iterdir():
        if path.is_dir() and path.name.startswith(DELETED_PREFIX):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except OSError:      # still held; it will be swept next time
                pass


def delete_subject(name: str) -> None:
    """
    Delete a subject: its documents, its conversations, its quiz questions and
    its progress. Nothing here is recoverable, so the interface asks first.

    On Windows a folder inside a synchronised OneDrive tree can refuse to be
    removed for a moment even once it is empty, because the sync client still
    holds a handle on it. Deleting the subject must not fail for that: after a
    few attempts the folder is renamed out of the way — it stops being a
    subject immediately — and swept up on the next delete.
    """
    folder = subject_path(name)
    forget(name)
    with _quiz_lock:
        for qid in [q for q, issued in _issued.items() if issued["subject"] == name]:
            del _issued[qid]
    _sweep_deleted()

    for attempt in range(3):
        try:
            shutil.rmtree(folder)
            return
        except OSError as exc:
            if attempt == 2:
                log.warning(f"Could not remove {folder} ({exc}) — renaming it instead")
            else:
                time.sleep(0.2)

    aside = PROJECTS_DIR / f"{DELETED_PREFIX}{folder.name}-{int(time.time())}"
    try:
        folder.rename(aside)
    except OSError as exc:
        raise RuntimeError(
            f"{name} could not be deleted — another program is using its "
            f"folder ({exc}). Close anything reading those files and try again.")
    try:                          # gone from the student's view either way
        shutil.rmtree(aside, ignore_errors=True)
    except OSError:
        pass


def _stream_answer(question: str, context: list) -> Iterator[str]:
    from backend.pipeline.generator import stream
    return stream(question, [str(c) for c in context])


def _tidy(text: str) -> str:
    from backend.pipeline.generator import _fix_number_spacing
    return _fix_number_spacing(text).strip()


def ask(name: str, question: str, chat_id: Optional[str] = None) -> Iterator[dict]:
    """
    Answer a question, as a series of events:
      {"type": "sources", "sources": [...]}   once retrieval is done
      {"type": "token", "text": ...}          as the answer is written
      {"type": "done", "answer": ..., "seconds": ..., "chat": {...}}
    The finished exchange is appended to the conversation given by chat_id, or
    to a new one, which takes its title from this question.
    """
    question = question.strip()
    if not question:
        raise ValueError("Ask a question first.")
    if chat_id is not None:
        _chat_path(name, chat_id)      # fail before answering, not after
    index = ready_index(name)
    started = time.time()
    with MODEL_LOCK:
        retrieved = index.find(question)
        sources = [describe(c) for c in retrieved]
        yield {"type": "sources", "sources": sources}
        pieces = []
        for piece in _stream_answer(question, retrieved):
            if piece:
                pieces.append(piece)
                yield {"type": "token", "text": piece}
    answer = _tidy("".join(pieces)) or NO_ANSWER
    seconds = round(time.time() - started, 1)

    record = (chat(name, chat_id) if chat_id is not None
              else {"id": uuid.uuid4().hex, "title": "", "created": started,
                    "messages": []})
    record["messages"].append({"role": "user", "content": question, "time": started})
    record["messages"].append({"role": "assistant", "content": answer,
                               "sources": sources, "seconds": seconds,
                               "time": time.time()})
    # A conversation is named after the question that started it.
    if not record.get("title") or record["title"] == "New conversation":
        record["title"] = chat_title(question)
    record["updated"] = time.time()
    _write_json(_chats_dir(name) / f"{record['id']}.json", record)
    yield {"type": "done", "answer": answer, "seconds": seconds,
           "chat": {"id": record["id"], "title": record["title"],
                    "updated": record["updated"]}}


# Quiz

def _pool_key(sig) -> str:
    # The answering model is part of the key: the stored reference answers
    # were written by it, so a different one must not inherit them. Imported
    # here rather than at the top, so that nothing loads transformers until a
    # question is actually asked.
    from backend.pipeline import generator
    return repr((CHUNKING, model_key(), HYBRID, generator.MODEL_NAME, sig))


def load_pool(name: str, sig) -> dict:
    """
    Saved quiz items by topic id, plus the chunk indices already tried for
    each topic. Discarded when the documents, chunking mode, embedding model,
    retrieval mode or answering model change, since chunk indices, the
    reference answers or the round-trip check would differ.
    """
    path = study_path(name, "quiz_pool.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("signature") == _pool_key(sig):
            return {
                "items": {tid: [quiz.QuizItem.from_record(r) for r in records]
                          for tid, records in data["items"].items()},
                "tried": {tid: set(v) for tid, v in data.get("tried", {}).items()},
            }
    return {"items": {}, "tried": {}}


def save_pool(name: str, sig, pool: dict) -> None:
    _write_json(study_path(name, "quiz_pool.json"), {
        "signature": _pool_key(sig),
        "items": {tid: [i.to_record() for i in items]
                  for tid, items in pool["items"].items()},
        "tried": {tid: sorted(v) for tid, v in pool["tried"].items()},
    })


def load_progress(name: str) -> Progress:
    return Progress.load(study_path(name, "progress.json"))


def reset_progress(name: str) -> None:
    study_path(name, "progress.json").unlink(missing_ok=True)


def times_asked(progress: Progress) -> dict:
    counts = {}
    for attempt in progress.attempts:
        counts[attempt.get("question")] = counts.get(attempt.get("question"), 0) + 1
    return counts


def topic_label(topic_id: str) -> str:
    return topic_id.replace(" › ", " — ")


def subject_topics(chunks) -> List[quiz.Topic]:
    """
    The topics a student can be quizzed on: every section, plus a whole-file
    topic for each document with more than one section.
    """
    return quiz.build_topics(chunks, whole_documents=True)


def topics(name: str) -> List[dict]:
    return [{"id": t.id, "document": t.source_file, "section": t.section,
             "scope": t.scope}
            for t in subject_topics(ready_index(name).chunks)]


def extend_topic(name, index: SubjectIndex, pool, topic, add: int) -> int:
    """
    Try up to `add` new questions for a topic from chunks not tried before.
    Returns how many were added.
    """
    items = pool["items"].setdefault(topic.id, [])
    tried = pool["tried"].setdefault(topic.id, set())
    untried = [i for i in topic.chunk_indices if i not in tried]
    random.Random(len(tried)).shuffle(untried)
    # A whole-file topic draws on the same chunks as that file's sections, so
    # compare against every question already written from this document.
    known = {quiz.normalize(i.question)
             for group in pool["items"].values() for i in group
             if i.source_file == topic.source_file} | {
        quiz.normalize(i.question) for i in items}

    added = 0
    for i in untried[:add * 3]:
        if added >= add:
            break
        tried.add(i)
        with MODEL_LOCK:
            item, _ = quiz.generate_item(index.chunks[i], retrieve_fn=index.find)
        if item is None or quiz.normalize(item.question) in known:
            continue
        if topic.scope != "document":
            # A question from a whole-file topic keeps the section it came
            # from, so mastery and levels stay per section.
            item.topic_id = topic.id
        items.append(item)
        known.add(quiz.normalize(item.question))
        added += 1
    save_pool(name, index.signature, pool)
    return added


def pick_item(items, progress: Progress):
    """
    The item asked least often so far, avoiding a repeat of the previous
    question when there is any alternative. Ties are broken at random.
    """
    asked = times_asked(progress)
    last = progress.attempts[-1].get("question") if progress.attempts else None
    choices = [i for i in items if i.question != last] or list(items)
    fewest = min(asked.get(i.question, 0) for i in choices)
    return random.choice([i for i in choices if asked.get(i.question, 0) == fewest])


# Questions handed out and not yet answered, by id. One student, one machine:
# kept in memory, so a restart simply means asking for a new question.
_issued: Dict[str, dict] = {}
_quiz_lock = threading.Lock()


def new_question(name: str, topic: Optional[str] = None) -> Optional[dict]:
    """
    Choose a topic (the recommender's pick when topic is None), make sure it
    has questions, and hand out the next one. None if no usable question
    could be written.
    """
    index = ready_index(name)
    by_id = {t.id: t for t in subject_topics(index.chunks)}
    if topic is not None and topic not in by_id:
        raise NotFound(f"No topic called {topic!r}")

    with _quiz_lock:   # two tabs asking at once would write the pool twice
        progress = load_progress(name)
        if topic is None:
            # Recommend sections, not whole files: a file is only ever chosen
            # deliberately, and its mastery is the average of its sections.
            sections = [t.id for t in by_id.values() if t.scope == "section"]
            candidates = [r.topic_id
                          for r in progress.recommend(sections, n=len(sections))]
        else:
            candidates = [topic]

        pool = load_pool(name, index.signature)
        asked = times_asked(progress)
        items, fallback = [], []
        for topic_id in candidates[:5]:
            t = by_id[topic_id]
            items = pool["items"].get(topic_id, [])
            untried = set(t.chunk_indices) - pool["tried"].get(topic_id, set())
            # New topic, or every question in it already asked: write more.
            if untried and all(asked.get(i.question, 0) for i in items):
                extend_topic(name, index, pool, t,
                             add=QUESTIONS_PER_TOPIC if not items else 1)
                items = pool["items"].get(topic_id, [])
            if any(not asked.get(i.question, 0) for i in items):
                break
            # Only repeats left here — use them only if no later topic has fresh ones.
            fallback = fallback or items
        else:
            items = fallback
        if not items:
            return None

        item = pick_item(items, progress)
        level = progress.next_level(item.topic_id)
        options = None
        if level == 1:
            def everything():
                return [i for group in pool["items"].values() for i in group]

            # Distractors come from other questions; write some for other
            # topics if there aren't enough yet.
            others = [t for t in candidates + list(by_id) if not pool["items"].get(t)]
            while len(everything()) < quiz.MC_OPTIONS and others:
                extend_topic(name, index, pool, by_id[others.pop(0)],
                             add=QUESTIONS_PER_TOPIC)
            options = quiz.multiple_choice(item, everything())
            if len(options) < 3:   # still too few distractors — ask as short answer
                level, options = 2, None

        qid = uuid.uuid4().hex
        _issued[qid] = {"subject": name, "item": item, "level": level,
                        "options": options}

    hint = None
    if level == 2:
        hint = {"section": item.section, "file": item.source_file, "page": item.page}
    return {"id": qid, "question": item.question, "level": level,
            "level_name": quiz.LEVELS[level], "topic": item.topic_id,
            "topic_label": topic_label(item.topic_id),
            "options": options, "hint": hint,
            "mastery": progress.mastery(item.topic_id),
            "practised": progress.attempts_on(item.topic_id)}


def answer_question(name: str, qid: str, answer: str) -> dict:
    """Grade an answer, record it, and return the outcome."""
    answer = answer.strip()
    if not answer:
        raise ValueError("Give an answer first.")
    with _quiz_lock:
        issued = _issued.get(qid)
        if issued is None or issued["subject"] != name:
            raise NotFound("That question has expired — ask for a new one.")
        del _issued[qid]
        item, level = issued["item"], issued["level"]
        if level == 1:
            correct = quiz.normalize(answer) == quiz.normalize(item.answer)
            score = float(correct)
        else:
            g = quiz.grade(answer, item.answer)
            correct, score = g.correct, g.score
        progress = load_progress(name)
        before = progress.mastery(item.topic_id)
        progress.record(item.topic_id, level, correct, question=item.question,
                        answer=answer, reference=item.answer, score=score)
        progress.save(study_path(name, "progress.json"))

    return {"correct": bool(correct), "score": round(float(score), 3),
            "reference": item.answer,
            "mastery_before": before, "mastery": progress.mastery(item.topic_id),
            "next_level": progress.next_level(item.topic_id),
            "source": {"file": item.source_file, "page": item.page,
                       "pages": str(item.page) if item.page else None,
                       "timecode": None, "from_image": False,
                       "section": item.section, "text": item.passage}}


# Progress

def progress_summary(name: str) -> dict:
    topic_list = quiz.build_topics(ready_index(name).chunks)
    progress = load_progress(name)
    answered = len(progress.attempts)
    correct = sum(a["correct"] for a in progress.attempts)
    rows = []
    for t in topic_list:
        n = progress.attempts_on(t.id)
        rows.append({
            "id": t.id, "document": t.source_file, "section": t.section,
            "answered": n,
            "correct": sum(a["correct"] for a in progress.attempts
                           if a["topic_id"] == t.id),
            "mastery": progress.mastery(t.id) if n else None,
            "next_level": progress.next_level(t.id),
            "next_level_name": quiz.LEVELS[progress.next_level(t.id)],
        })
    recent = [{"topic": a["topic_id"], "level": a["level"], "correct": a["correct"],
               "question": a.get("question"), "time": a.get("time")}
              for a in progress.attempts[-8:][::-1]]
    return {
        "answered": answered,
        "correct": correct,
        "accuracy": correct / answered if answered else None,
        "practised": sum(1 for r in rows if r["answered"]),
        "topics": rows,
        "recommendations": [
            {"topic": r.topic_id, "label": topic_label(r.topic_id),
             "mastery": r.mastery, "attempts": r.attempts, "reason": r.reason}
            for r in progress.recommend([t.id for t in topic_list], n=3)],
        "recent": recent,
    }


def settings() -> dict:
    from backend.pipeline import generator
    from backend.pipeline.device import get_torch_device, should_use_npu
    explaining = generator.ANSWER_STYLE == "explain"
    return {"embedder": model_key(), "chunking": CHUNKING, "hybrid": HYBRID,
            "top_k": TOP_K,
            "answer_model": (generator.CHAT_MODEL_NAME.split("/")[-1] if explaining
                             else generator.MODEL_NAME.split("/")[-1]),
            "quiz_model": generator.MODEL_NAME.split("/")[-1],
            "answer_style": generator.ANSWER_STYLE,
            "device": "npu" if should_use_npu() else get_torch_device(),
            "extensions": SUPPORTED_EXTENSIONS,
            "levels": quiz.LEVELS,
            "samples": SAMPLE_DIR.is_dir() and any(
                p.suffix.lower() in EXTENSION_MAP for p in SAMPLE_DIR.iterdir())}
