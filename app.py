"""
app.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Streamlit chat interface.

The student organises material into projects (subjects). Each project holds any
number of uploaded documents, and a question is answered against everything in
the selected project at once — one combined index per project, not per file.

A thin wrapper over the existing pipeline, not a reimplementation:

    loader.load_file -> preprocessor.preprocess -> embedder.embed
      -> FAISS index -> retriever.retrieve -> generator.generate

Run from the repo root:
    streamlit run app.py
"""

import os
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

SUPPORTED_EXTENSIONS = sorted({ext.lstrip(".") for ext in EXTENSION_MAP})

# Retrieval depth is a development-time setting, not a user-facing control.
TOP_K = 3

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
    return f"{source_file} — page {page}" if page is not None else source_file


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
            if st.form_submit_button("➕ New subject", use_container_width=True):
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
            if st.button("Use the sample papers", use_container_width=True):
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


def main():
    st.set_page_config(page_title="Study Assistant", page_icon="📚",
                       layout="centered")
    st.session_state.setdefault("chats", {})

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
    with st.spinner(spinner_text):
        index, chunks, per_document, failures = build_project_index(
            project.name, project_signature(project))

    for name, error in failures:
        st.warning(f"Couldn't read **{name}** — {error}")

    if index is None:
        st.error("None of the documents in this subject could be read.")
        return

    st.caption(
        f"{len(documents)} document(s) · {len(chunks)} chunks indexed · "
        f"answers are drawn from all of them"
    )

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
