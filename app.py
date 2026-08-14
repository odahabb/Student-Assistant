"""
app.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Streamlit interface: ask questions, take quizzes, track progress.

The student organises material into projects (subjects). Each project holds any
number of uploaded documents, and a question is answered against everything in
the selected project at once — one combined index per project, not per file.

A thin wrapper over the existing pipeline, not a reimplementation:

    loader.load_file -> preprocessor.preprocess -> embedder.embed
      -> FAISS index -> retriever.retrieve -> generator.generate

Three views share that index:
  Ask      — the chat interface.
  Quiz     — questions generated per topic (document section) by quiz.py, at
             a difficulty chosen by recommender.py from past answers.
  Progress — estimated mastery per topic and what to revise next.
Quiz questions and progress are saved under <project>/_study/.

Run from the repo root:
    streamlit run app.py
"""

import json
import os
import random
import re
import shutil
import time
from pathlib import Path

# CPU-only, matching the pipeline's demo path. Set before importing any pipeline
# module: device.py reads this when the models are lazily constructed.
os.environ.setdefault("SA_DEVICE", "cpu")

import faiss
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROJECTS_DIR = ROOT / "data" / "projects"
SAMPLE_DIR = ROOT / "data" / "raw"

from backend.pipeline.loader import EXTENSION_MAP, load_file
from backend.pipeline.preprocessor import preprocess
from backend.pipeline.embedder import embed
from backend.pipeline.retriever import retrieve
from backend.pipeline.generator import generate
from backend.pipeline import quiz
from backend.pipeline.recommender import Progress

SUPPORTED_EXTENSIONS = sorted({ext.lstrip(".") for ext in EXTENSION_MAP})

# Retrieval depth is a development-time setting, not a user-facing control.
TOP_K = 3
QUESTIONS_PER_TOPIC = 2
STUDY_DIR = "_study"
VIEWS = {
    "Ask": ":material/chat: Ask",
    "Quiz": ":material/quiz: Quiz",
    "Progress": ":material/monitoring: Progress",
}

_ILLEGAL_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


# Projects on disk

def project_dirs():
    """Existing projects, oldest first."""
    if not PROJECTS_DIR.is_dir():
        return []
    return sorted((p for p in PROJECTS_DIR.iterdir() if p.is_dir()),
                  key=lambda p: p.name.lower())


def create_project(name: str) -> Path:
    safe = _ILLEGAL_NAME_CHARS.sub("", name).strip().strip(".")
    if not safe:
        raise ValueError("That name can't be used as a folder name.")
    path = PROJECTS_DIR / safe
    path.mkdir(parents=True, exist_ok=True)
    return path


def project_documents(project: Path):
    """Files in a project the pipeline knows how to read."""
    return sorted((p for p in project.iterdir()
                   if p.is_file() and p.suffix.lower() in EXTENSION_MAP),
                  key=lambda p: p.name.lower())


def project_signature(project: Path):
    """
    Identity of a project's document set. Used as a cache key so adding or
    replacing a document rebuilds the index, and nothing else does.
    """
    return tuple((p.name, p.stat().st_mtime, p.stat().st_size)
                 for p in project_documents(project))


# Pipeline

@st.cache_resource(show_spinner=False, max_entries=8)
def build_project_index(project_name: str, signature):
    """
    Ingest every document in a project into one combined index.

    The index is held in memory rather than written through vector_store's
    single fixed path, since several projects coexist and would otherwise
    overwrite each other's store. Chunking, embedding and retrieval are
    unchanged; only where the index lives differs.
    """
    project = PROJECTS_DIR / project_name
    chunks, per_document, failures = [], [], []

    for path in project_documents(project):
        try:
            loaded = load_file(str(path))
            file_chunks = preprocess(loaded, source_file=path.name)
        except Exception as exc:  # a bad upload shouldn't sink the project
            failures.append((path.name, str(exc)))
            continue
        chunks.extend(file_chunks)
        pages = [c.page for c in file_chunks if c.page is not None]
        per_document.append({
            "name": path.name,
            "chunks": len(file_chunks),
            "pages": max(pages) if pages else None,
        })

    if not chunks:
        return None, [], per_document, failures

    embeddings = embed(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index, chunks, per_document, failures


def describe_source(chunk) -> str:
    source_file = getattr(chunk, "source_file", None) or "unknown file"
    page = getattr(chunk, "page", None)
    section = getattr(chunk, "section", None)
    label = f"{source_file} — page {page}" if page is not None else source_file
    return f"{label} · {section}" if section else label


# Quiz state on disk

def study_path(project: Path, name: str) -> Path:
    return project / STUDY_DIR / name


def load_pool(project: Path, signature) -> dict:
    """
    Saved quiz items by topic id, plus the chunk indices already tried for each
    topic. Discarded when the documents change, since chunk indices then shift.
    """
    path = study_path(project, "quiz_pool.json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("signature") == repr(signature):
            return {
                "items": {tid: [quiz.QuizItem.from_record(r) for r in records]
                          for tid, records in data["items"].items()},
                "tried": {tid: set(v) for tid, v in data.get("tried", {}).items()},
            }
    return {"items": {}, "tried": {}}


def save_pool(project: Path, signature, pool: dict) -> None:
    path = study_path(project, "quiz_pool.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"signature": repr(signature),
               "items": {tid: [i.to_record() for i in items]
                         for tid, items in pool["items"].items()},
               "tried": {tid: sorted(v) for tid, v in pool["tried"].items()}}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_progress(project: Path) -> Progress:
    return Progress.load(study_path(project, "progress.json"))


def times_asked(progress: Progress) -> dict:
    counts = {}
    for attempt in progress.attempts:
        counts[attempt.get("question")] = counts.get(attempt.get("question"), 0) + 1
    return counts


def extend_topic(project, signature, pool, topic, index, chunks, add: int) -> int:
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
        item, _ = quiz.generate_item(
            chunks[i], retrieve_fn=lambda q: retrieve(q, index, chunks, k=TOP_K))
        if item is None or quiz.normalize(item.question) in known:
            continue
        item.topic_id = topic.id
        items.append(item)
        known.add(quiz.normalize(item.question))
        added += 1
    save_pool(project, signature, pool)
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


# UI

def render_sidebar():
    """Project picker, creation, and per-project uploads. Returns the selection."""
    with st.sidebar:
        st.title("📚 Subjects")

        projects = project_dirs()
        names = [p.name for p in projects]

        selected_name = None
        if names:
            if st.session_state.get("selected_project") not in names:
                st.session_state["selected_project"] = names[0]
            selected_name = st.radio(
                "Your subjects", names, key="selected_project",
                label_visibility="collapsed",
            )
        else:
            st.caption("No subjects yet — create one below to get started.")

        with st.form("new_project", clear_on_submit=True):
            new_name = st.text_input(
                "New subject", placeholder="e.g. Machine Learning",
                label_visibility="collapsed",
            )
            if st.form_submit_button("➕ New subject", width="stretch"):
                if new_name.strip():
                    try:
                        created = create_project(new_name)
                        st.session_state["selected_project"] = created.name
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                else:
                    st.warning("Give the subject a name first.")

        if selected_name is None:
            return None

        project = PROJECTS_DIR / selected_name
        st.divider()
        st.subheader("Materials")

        documents = project_documents(project)
        if documents:
            for path in documents:
                st.caption(f"📄 {path.name}")
        else:
            st.caption("No documents yet.")

        uploaded = st.file_uploader(
            "Add documents", type=SUPPORTED_EXTENSIONS,
            accept_multiple_files=True, key=f"upload_{selected_name}",
            help="PDFs, images, audio or text — everything loader.py supports. "
                 "Questions are answered across all of them together.",
        )
        if uploaded:
            added = 0
            for item in uploaded:
                destination = project / item.name
                if not destination.exists():
                    destination.write_bytes(item.getbuffer())
                    added += 1
            if added:
                st.success(f"Added {added} document(s).")
                st.rerun()

        if not documents and SAMPLE_DIR.is_dir():
            if st.button("Use the sample papers", width="stretch"):
                for path in SAMPLE_DIR.iterdir():
                    if path.is_file() and path.suffix.lower() in EXTENSION_MAP:
                        shutil.copy2(path, project / path.name)
                st.rerun()

        return project


def render_history(project_name: str):
    for message in st.session_state["chats"].get(project_name, []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for i, source in enumerate(message["sources"], start=1):
                        st.markdown(f"**{i}. {source['label']}**")
                        st.caption(source["text"])


def render_ask(project: Path, index, chunks):
    render_history(project.name)

    question = st.chat_input(f"Ask something about {project.name}…")
    if not question:
        return

    st.session_state["chats"][project.name].append(
        {"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your materials and writing an answer… "
                        "(first answer loads the model and is slower)"):
            started = time.time()
            retrieved = retrieve(question, index, chunks, k=TOP_K)
            answer = generate(question, retrieved)
            elapsed = time.time() - started

        if not answer.strip():
            answer = "_I couldn't find an answer to that in this subject's materials._"
        st.markdown(answer)

        sources = [{"label": describe_source(c), "text": str(c)} for c in retrieved]
        with st.expander(f"Sources ({len(sources)})"):
            for i, source in enumerate(sources, start=1):
                st.markdown(f"**{i}. {source['label']}**")
                st.caption(source["text"])
        st.caption(f"Answered in {elapsed:.1f}s")

    st.session_state["chats"][project.name].append(
        {"role": "assistant", "content": answer, "sources": sources})


def topic_label(topic_id: str) -> str:
    return topic_id.replace(" › ", " — ")


def practise(project_name: str, topic_id: str):
    """Button callback: jump to the quiz view on a given topic."""
    st.session_state["view"] = "Quiz"
    st.session_state[f"quiz_topic_{project_name}"] = topic_id


def new_question(project, signature, index, chunks, topics, progress, choice):
    """Choose a topic, make sure it has questions, and set up the next one."""
    by_id = {t.id: t for t in topics}
    if choice == "Recommended":
        candidates = [r.topic_id for r in progress.recommend(by_id, n=len(by_id))]
    else:
        candidates = [choice]

    pool = load_pool(project, signature)
    asked = times_asked(progress)
    items, fallback = [], []
    for topic_id in candidates[:5]:
        topic = by_id[topic_id]
        items = pool["items"].get(topic_id, [])
        untried = set(topic.chunk_indices) - pool["tried"].get(topic_id, set())
        # New topic, or every question in it already asked: write more.
        if untried and all(asked.get(i.question, 0) for i in items):
            with st.spinner(f"Writing questions on {topic_label(topic_id)}… "
                            "(this takes a little while)"):
                extend_topic(project, signature, pool, topic, index, chunks,
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

        # Distractors come from other questions; write some for other topics
        # if there aren't enough yet.
        others = [t for t in candidates + list(by_id) if not pool["items"].get(t)]
        while len(everything()) < quiz.MC_OPTIONS and others:
            topic_id = others.pop(0)
            with st.spinner("Writing a few more questions for the answer options…"):
                extend_topic(project, signature, pool, by_id[topic_id], index, chunks,
                             add=QUESTIONS_PER_TOPIC)
        options = quiz.multiple_choice(item, everything())
        if len(options) < 3:   # still too few distractors — ask as short answer
            level, options = 2, None
    return {"item": item.to_record(), "level": level, "options": options}


def render_quiz(project: Path, signature, index, chunks):
    topics = quiz.build_topics(chunks)
    if not topics:
        st.info("There isn't enough text in this subject to write quiz questions from.")
        return

    progress = load_progress(project)
    state = st.session_state["quiz"].setdefault(project.name,
                                                {"current": None, "result": None})

    with st.container(horizontal=True, vertical_alignment="bottom"):
        choice = st.selectbox(
            "Topic", ["Recommended"] + [t.id for t in topics],
            key=f"quiz_topic_{project.name}",
            format_func=lambda t: ("Recommended for me" if t == "Recommended"
                                   else topic_label(t)),
            width="stretch")
        if st.button("New question", type="primary", icon=":material/refresh:"):
            state["current"] = new_question(project, signature, index, chunks,
                                            topics, progress, choice)
            state["result"] = None
            if state["current"] is None:
                st.warning("Couldn't write a usable question for that topic — "
                           "try another one.")

    current = state["current"]
    if current is None:
        st.caption("Pick a topic, or let the recommendations choose, then press "
                   "**New question**. Difficulty adapts to your answers.")
        return

    item = quiz.QuizItem.from_record(current["item"])
    level = current["level"]
    with st.container(border=True):
        with st.container(horizontal=True):
            st.badge(quiz.LEVELS[level], icon=":material/signal_cellular_alt:",
                     color=["green", "orange", "red"][level - 1])
            st.caption(topic_label(item.topic_id))
        st.markdown(f"#### {item.question}")
        if level == 2:
            page = f", page {item.page}" if item.page else ""
            st.caption(f":material/lightbulb: Hint: look in *{item.section or item.source_file}*"
                       f" ({item.source_file}{page}).")

        with st.form(f"answer_{project.name}", border=False):
            if level == 1:
                answer = st.radio("Your answer", current["options"], index=None)
            else:
                answer = st.text_input("Your answer")
            submitted = st.form_submit_button("Check answer",
                                              disabled=state["result"] is not None)

    if submitted and state["result"] is None:
        if not answer:
            st.warning("Give an answer first.")
            return
        if level == 1:
            correct = quiz.normalize(answer) == quiz.normalize(item.answer)
            result = {"correct": correct, "score": float(correct)}
        else:
            g = quiz.grade(answer, item.answer)
            result = {"correct": g.correct, "score": g.score}
        progress.record(item.topic_id, level, result["correct"],
                        question=item.question, answer=answer,
                        reference=item.answer, score=result["score"])
        progress.save(study_path(project, "progress.json"))
        result["mastery"] = progress.mastery(item.topic_id)
        state["result"] = result

    result = state["result"]
    if result:
        if result["correct"]:
            st.success(f"Correct. Reference answer: **{item.answer}**", icon=":material/check_circle:")
        else:
            st.error(f"Not quite. Reference answer: **{item.answer}**", icon=":material/cancel:")
        st.caption(f"Estimated mastery of this topic is now {result['mastery']:.0%}.")
        page = f", page {item.page}" if item.page else ""
        with st.expander(f"Source — {item.source_file}{page}"):
            st.caption(item.passage)


def render_progress(project: Path, chunks):
    import pandas as pd

    topics = quiz.build_topics(chunks)
    progress = load_progress(project)
    if not progress.attempts:
        st.info("No quiz answers yet. Your progress per topic will appear here "
                "once you've answered a few questions.")
        return

    answered = len(progress.attempts)
    correct = sum(a["correct"] for a in progress.attempts)
    practised = sum(1 for t in topics if progress.attempts_on(t.id))
    with st.container(horizontal=True):
        st.metric("Questions answered", answered, border=True)
        st.metric("Correct", f"{correct / answered:.0%}", border=True)
        st.metric("Topics practised", f"{practised} / {len(topics)}", border=True)

    st.subheader("Revise next")
    for rec in progress.recommend([t.id for t in topics], n=3):
        with st.container(border=True, horizontal=True, vertical_alignment="center"):
            with st.container():
                st.markdown(f"**{topic_label(rec.topic_id)}**")
                st.caption(rec.reason)
            st.button("Practise", key=f"practise_{rec.topic_id}", icon=":material/quiz:",
                      on_click=practise, args=(project.name, rec.topic_id))

    st.subheader("All topics")
    rows = []
    for t in topics:
        n = progress.attempts_on(t.id)
        rows.append({
            "Document": t.source_file,
            "Section": t.section,
            "Answered": n,
            "Correct": sum(a["correct"] for a in progress.attempts
                           if a["topic_id"] == t.id),
            "Mastery": progress.mastery(t.id) if n else None,
            "Next question": quiz.LEVELS[progress.next_level(t.id)],
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True,
        column_config={"Mastery": st.column_config.ProgressColumn(
            "Mastery", min_value=0.0, max_value=1.0, format="percent")})

    with st.expander("Reset progress"):
        st.caption("Deletes every recorded answer for this subject.")
        if st.button("Delete my progress", type="secondary"):
            study_path(project, "progress.json").unlink(missing_ok=True)
            st.rerun()


def main():
    st.set_page_config(page_title="Study Assistant", page_icon="📚",
                       layout="centered")
    st.session_state.setdefault("chats", {})
    st.session_state.setdefault("quiz", {})
    st.session_state.setdefault("view", "Ask")

    project = render_sidebar()

    if project is None:
        st.title("📚 Study Assistant")
        st.info("Create a subject in the sidebar to get started.")
        return

    st.title(project.name)
    st.session_state["chats"].setdefault(project.name, [])

    documents = project_documents(project)
    if not documents:
        st.info("Add some documents to this subject in the sidebar, "
                "then ask a question about them.")
        return

    # Ingestion — cached, so it only reruns when the document set changes.
    slow_types = [p.name for p in documents
                  if p.suffix.lower() not in (".pdf", ".txt")]
    spinner_text = f"Reading {len(documents)} document(s)…"
    if slow_types:
        spinner_text += (" Images and audio run a vision or speech model first, "
                         "which can take a minute or more on CPU.")
    signature = project_signature(project)
    with st.spinner(spinner_text):
        index, chunks, per_document, failures = build_project_index(
            project.name, signature)

    for name, error in failures:
        st.warning(f"Couldn't read **{name}** — {error}")

    if index is None:
        st.error("None of the documents in this subject could be read.")
        return

    st.caption(
        f"{len(documents)} document(s) · {len(chunks)} chunks indexed · "
        f"answers are drawn from all of them"
    )

    view = st.segmented_control(
        "View", list(VIEWS), key="view", format_func=VIEWS.get,
        label_visibility="collapsed")

    if view == "Quiz":
        render_quiz(project, signature, index, chunks)
    elif view == "Progress":
        render_progress(project, chunks)
    else:
        render_ask(project, index, chunks)


def _running_under_streamlit() -> bool:
    """
    True when the script is being executed by `streamlit run`.

    Run as plain `python app.py`, Streamlit works in "bare mode": its element
    calls become no-ops that return None, which surfaces later as a confusing
    AttributeError deep in the page rather than a message about the command.
    """
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # pragma: no cover - older layouts
        try:
            from streamlit.runtime.scriptrunner.script_run_context import (
                get_script_run_ctx,
            )
        except ImportError:
            return False
    try:
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if not _running_under_streamlit():
        raise SystemExit(
            "\nThis is a Streamlit app, so it has to be started by Streamlit "
            "rather than run directly.\n\n"
            "    streamlit run app.py\n\n"
            "(Running `python app.py` puts Streamlit in bare mode, where its "
            "UI calls do nothing.)\n"
        )
    main()
