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

_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class NotFound(LookupError):
    """A subject, document or question that doesn't exist."""


# Subjects and documents on disk

def subject_names() -> List[str]:
    if not PROJECTS_DIR.is_dir():
        return []
    return sorted((p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()),
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


def chat_history(name: str) -> List[dict]:
    path = study_path(name, "chat.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def clear_chat(name: str) -> None:
    study_path(name, "chat.json").unlink(missing_ok=True)


def _stream_answer(question: str, context: list) -> Iterator[str]:
    from backend.pipeline.generator import stream
    return stream(question, [str(c) for c in context])


def _tidy(text: str) -> str:
    from backend.pipeline.generator import _fix_number_spacing
    return _fix_number_spacing(text).strip()


def ask(name: str, question: str) -> Iterator[dict]:
    """
    Answer a question, as a series of events:
      {"type": "sources", "sources": [...]}   once retrieval is done
      {"type": "token", "text": ...}          as the answer is written
      {"type": "done", "answer": ..., "seconds": ...}
    The finished exchange is appended to the subject's chat history.
    """
    question = question.strip()
    if not question:
        raise ValueError("Ask a question first.")
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

    history = chat_history(name)
    history.append({"role": "user", "content": question, "time": started})
    history.append({"role": "assistant", "content": answer, "sources": sources,
                    "seconds": seconds, "time": time.time()})
    _write_json(study_path(name, "chat.json"), history)
    yield {"type": "done", "answer": answer, "seconds": seconds}


# Quiz

def _pool_key(sig) -> str:
    return repr((CHUNKING, model_key(), HYBRID, sig))


def load_pool(name: str, sig) -> dict:
    """
    Saved quiz items by topic id, plus the chunk indices already tried for
    each topic. Discarded when the documents, chunking mode, embedding model or
    retrieval mode change, since chunk indices or the round-trip check would
    differ.
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


def topics(name: str) -> List[dict]:
    return [{"id": t.id, "document": t.source_file, "section": t.section}
            for t in quiz.build_topics(ready_index(name).chunks)]


def extend_topic(name, index: SubjectIndex, pool, topic, add: int) -> int:
    """
    Try up to `add` new questions for a topic from chunks not tried before.
    Returns how many were added.
    """
    items = pool["items"].setdefault(topic.id, [])
    tried = pool["tried"].setdefault(topic.id, set())
    untried = [i for i in topic.chunk_indices if i not in tried]
    random.Random(len(tried)).shuffle(untried)
    known = {quiz.normalize(i.question) for i in items}

    added = 0
    for i in untried[:add * 3]:
        if added >= add:
            break
        tried.add(i)
        with MODEL_LOCK:
            item, _ = quiz.generate_item(index.chunks[i], retrieve_fn=index.find)
        if item is None or quiz.normalize(item.question) in known:
            continue
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
    by_id = {t.id: t for t in quiz.build_topics(index.chunks)}
    if topic is not None and topic not in by_id:
        raise NotFound(f"No topic called {topic!r}")

    with _quiz_lock:   # two tabs asking at once would write the pool twice
        progress = load_progress(name)
        if topic is None:
            candidates = [r.topic_id for r in progress.recommend(by_id, n=len(by_id))]
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
    from backend.pipeline.device import get_torch_device, should_use_npu
    return {"embedder": model_key(), "chunking": CHUNKING, "hybrid": HYBRID,
            "top_k": TOP_K,
            "device": "npu" if should_use_npu() else get_torch_device(),
            "extensions": SUPPORTED_EXTENSIONS,
            "levels": quiz.LEVELS,
            "samples": SAMPLE_DIR.is_dir() and any(
                p.suffix.lower() in EXTENSION_MAP for p in SAMPLE_DIR.iterdir())}
