"""
backend/scripts/eval_latency.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Times every stage a student waits for, on the machine the project runs on.

The system's central constraint is that it runs on one laptop with no network
call, so the cost of that choice has to be stated rather than implied. This
script measures it end to end: loading a document, chunking, embedding,
building the index, retrieving, answering in both styles, and writing one
quiz question.

Cold and warm are reported separately, because they are different experiences.
The first answer of a session includes loading a 1.5B-parameter model from
disk; every answer after it does not, and the interface tells the student so.

Nothing here changes behaviour: each stage is called exactly as the
application calls it. Times are medians of REPEATS runs, except the cold ones,
which happen once by definition.

Run from anywhere, with nothing else running:
    SA_DEVICE=cpu SA_EMBEDDER=bge-small python backend/scripts/eval_latency.py
    SA_DEVICE=gpu SA_EMBEDDER=bge-small python backend/scripts/eval_latency.py
"""

import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend.pipeline import quiz, sparse as sparse_module  # noqa: E402
from backend.pipeline.device import get_torch_device, should_use_npu  # noqa: E402
from backend.pipeline.embedder import embed, model_key  # noqa: E402
from backend.pipeline.generator import (CHAT_MODEL_NAME, MODEL_NAME,  # noqa: E402
                                        answer_short, complete, explain)
from backend.pipeline.loader import load_pdf  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import retrieve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
DECK_DIR = ROOT / "data" / "projects" / "AI"
EVAL_DIR = ROOT / "data" / "eval"
# The application pins SA_DEVICE=cpu (service.py), so that is the run that
# describes what a student waits for; SA_DEVICE=gpu measures the Arc GPU and
# is written to its own file.
OUT_PATH = EVAL_DIR / (f"latency_{os.environ.get('SA_DEVICE', 'gpu')}.json")

REPEATS = 3
TOP_K = 3
PROSE_PDF = RAW_DIR / "Whisper.pdf"
SLIDE_PDF = DECK_DIR / "ai-2-breeding.pdf"
QUESTIONS = [
    "How many hours of audio was Whisper trained on?",
    "What sample rate does Whisper use?",
    "What architecture does Whisper use?",
]


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0


def repeat(fn, *args, **kwargs):
    """(result of the last run, seconds for each run)."""
    seconds, result = [], None
    for _ in range(REPEATS):
        result, elapsed = timed(fn, *args, **kwargs)
        seconds.append(elapsed)
    return result, seconds


def stat(seconds):
    return {"median_seconds": round(statistics.median(seconds), 2),
            "min_seconds": round(min(seconds), 2),
            "max_seconds": round(max(seconds), 2),
            "runs": len(seconds)}


def one(seconds):
    return {"seconds": round(seconds, 2), "runs": 1}


def main():
    # Slide text carries Unicode the Windows console cannot encode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    device = "npu" if should_use_npu() else get_torch_device()
    print(f"device: {device}   embedder: {model_key()}   model: {MODEL_NAME}\n")
    results = {}

    # Reading documents
    print("reading documents")
    prose_pages, seconds = repeat(load_pdf, str(PROSE_PDF))
    results["load_prose_pdf"] = {**stat(seconds), "file": PROSE_PDF.name,
                                 "pages": len(prose_pages)}
    slide_pages, seconds = repeat(load_pdf, str(SLIDE_PDF), figures="off")
    results["load_slides_text_only"] = {**stat(seconds), "file": SLIDE_PDF.name,
                                        "slides": len(slide_pages)}
    slide_pictures, seconds = repeat(load_pdf, str(SLIDE_PDF), figures="auto")
    results["load_slides_pictures_cached"] = {**stat(seconds),
                                              "file": SLIDE_PDF.name,
                                              "slides": len(slide_pictures)}
    for key in ("load_prose_pdf", "load_slides_text_only",
                "load_slides_pictures_cached"):
        print(f"  {key:<32} {results[key]['median_seconds']:>7.2f}s")

    # Chunking and embedding, per document and for a whole subject
    print("\nchunking, embedding, indexing")
    prose_chunks, seconds = repeat(preprocess, prose_pages, chunking="sentence")
    results["chunk_prose_pdf"] = {**stat(seconds), "chunks": len(prose_chunks)}
    slide_chunks, seconds = repeat(preprocess, slide_pictures)
    results["chunk_slide_deck"] = {**stat(seconds), "chunks": len(slide_chunks)}

    # Cold embedder load is the first embed() call of the process.
    _, cold = timed(embed, [str(c) for c in prose_chunks[:1]])
    results["embed_cold_first_call"] = one(cold)
    _, seconds = repeat(embed, prose_chunks)
    results["embed_prose_chunks"] = {**stat(seconds), "chunks": len(prose_chunks)}

    subject_paths = sorted(DECK_DIR.glob("*.pdf"))
    t0 = time.perf_counter()
    subject_chunks = []
    for path in subject_paths:
        subject_chunks.extend(preprocess(load_pdf(str(path), figures="auto")))
    vectors = embed(subject_chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    keywords = sparse_module.build_index(subject_chunks)
    results["index_whole_subject"] = {
        **one(time.perf_counter() - t0),
        "documents": len(subject_paths), "chunks": len(subject_chunks),
        "note": "fifteen slide decks, pictures already in the figure cache",
    }
    for key in ("chunk_prose_pdf", "chunk_slide_deck", "embed_cold_first_call",
                "embed_prose_chunks", "index_whole_subject"):
        value = results[key]
        print(f"  {key:<32} "
              f"{value.get('median_seconds', value.get('seconds')):>7.2f}s")

    # Retrieval
    print("\nretrieval")
    _, seconds = repeat(retrieve, QUESTIONS[0], index, subject_chunks, k=TOP_K,
                        sparse=keywords)
    results["retrieve_hybrid"] = {**stat(seconds), "k": TOP_K,
                                  "chunks_searched": len(subject_chunks)}
    print(f"  {'retrieve_hybrid':<32} {results['retrieve_hybrid']['median_seconds']:>7.2f}s")

    # Answering. The first call loads the model, so it is timed on its own.
    print("\nanswering")
    context = [str(c) for c in retrieve(QUESTIONS[0], index, subject_chunks,
                                        k=TOP_K, sparse=keywords)]
    _, cold = timed(answer_short, QUESTIONS[0], context)
    results["answer_short_cold"] = {**one(cold), "note": f"includes loading {MODEL_NAME}"}
    print(f"  {'answer_short_cold':<32} {cold:>7.2f}s")

    seconds = []
    for question in QUESTIONS:
        ctx = [str(c) for c in retrieve(question, index, subject_chunks,
                                        k=TOP_K, sparse=keywords)]
        _, elapsed = timed(answer_short, question, ctx)
        seconds.append(elapsed)
    results["answer_short_warm"] = {**stat(seconds), "questions": len(QUESTIONS)}

    _, cold = timed(explain, QUESTIONS[0], context)
    results["explain_first_call"] = {
        **one(cold),
        "note": ("same weights as the short answer when SA_CHAT_MODEL is unset"
                 if CHAT_MODEL_NAME == MODEL_NAME else f"loads {CHAT_MODEL_NAME}")}
    seconds, words = [], []
    for question in QUESTIONS:
        ctx = [str(c) for c in retrieve(question, index, subject_chunks,
                                        k=TOP_K, sparse=keywords)]
        answer, elapsed = timed(explain, question, ctx)
        seconds.append(elapsed)
        words.append(len(answer.split()))
    results["explain_warm"] = {**stat(seconds), "questions": len(QUESTIONS),
                               "median_words": statistics.median(words)}
    for key in ("answer_short_warm", "explain_first_call", "explain_warm"):
        value = results[key]
        print(f"  {key:<32} "
              f"{value.get('median_seconds', value.get('seconds')):>7.2f}s")

    # One quiz question, written and round-trip checked as the app does it
    print("\nquiz")
    def find(question):
        return retrieve(question, index, subject_chunks, k=TOP_K, sparse=keywords)

    seconds, kept = [], 0
    usable = [c for c in subject_chunks if quiz.usable_for_quiz(c)][:REPEATS * 2]
    for chunk in usable[:REPEATS]:
        item, elapsed = timed(quiz.generate_item, chunk, retrieve_fn=find)
        seconds.append(elapsed)
        kept += item[0] is not None
    results["quiz_item_with_roundtrip"] = {**stat(seconds), "kept": kept,
                                           "attempts": len(seconds)}
    print(f"  {'quiz_item_with_roundtrip':<32} "
          f"{results['quiz_item_with_roundtrip']['median_seconds']:>7.2f}s "
          f"({kept}/{len(seconds)} kept)")

    OUT_PATH.write_text(json.dumps({
        "note": "Wall-clock time per pipeline stage, measured on the project "
                "machine with nothing else running. Cold entries include "
                "loading a model from disk and happen once per process.",
        "device": device,
        "embedder": model_key(),
        "model": MODEL_NAME,
        "chat_model": CHAT_MODEL_NAME,
        "repeats": REPEATS,
        "top_k": TOP_K,
        "stages": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
