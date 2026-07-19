"""
backend/scripts/eval_recall.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Retrieval evaluation: Recall@1 / Recall@3 / Recall@5 against a hand-built
ground-truth set (data/eval/retrieval_ground_truth.json).

A question counts as a hit at k when at least one of the top-k retrieved chunks
has BOTH source_file and page matching the ground-truth entry exactly. Page
numbers are 1-indexed throughout, matching loader.load_pdf().

This script is read-only with respect to pipeline behaviour — it drives
loader -> preprocessor -> embedder -> vector_store -> retriever exactly as the
pipeline does and only measures what comes back. Nothing is tuned or corrected
here on the basis of the results.

Run from anywhere:
    python backend/scripts/eval_recall.py
"""

import json
import os
import sys
import time
from pathlib import Path

# Resolve the repo root and work from it: vector_store.py uses paths relative to
# the working directory ("data/processed/..."), so the results are only
# reproducible if the working directory is pinned.
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.embedder import embed  # noqa: E402
from backend.pipeline.vector_store import build_and_save, load  # noqa: E402
from backend.pipeline.retriever import retrieve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
# Optional first argument redirects the results file, so a variant run (e.g. with
# page-1 boilerplate stripping) can be saved without overwriting the baseline.
RESULTS_PATH = ((ROOT / sys.argv[1]).resolve() if len(sys.argv) > 1
                else EVAL_DIR / "recall_results.json")

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]

K_VALUES = [1, 3, 5]
PREVIEW_CHARS = 240


def build_combined_index():
    """
    Load all four PDFs through the pipeline and build ONE index spanning them.

    Returns (index, stored_chunks, per_document_stats).
    """
    all_chunks = []
    stats = []

    for name in DOCUMENTS:
        path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(f"Source PDF not found: {path}")

        t0 = time.time()
        pages = load_file(str(path))
        chunks = preprocess(pages)
        elapsed = time.time() - t0

        all_chunks.extend(chunks)
        stats.append({
            "source_file": name,
            "pages_with_text": len(pages),
            "max_page": max((p["page"] for p in pages), default=0),
            "chunks": len(chunks),
            "load_preprocess_sec": round(elapsed, 2),
        })
        print(f"  {name:<52} {len(pages):>3} pages -> {len(chunks):>4} chunks "
              f"({elapsed:.1f}s)")

    print(f"\n  combined: {len(all_chunks)} chunks across {len(DOCUMENTS)} documents")

    embeddings = embed(all_chunks)
    build_and_save(embeddings, all_chunks)
    index, stored_chunks = load()
    print(f"  FAISS index: {index.ntotal} vectors x {embeddings.shape[1]} dims\n")

    return index, stored_chunks, stats


def describe(chunk, rank, expected):
    """One retrieved chunk, as a JSON-serialisable record."""
    source_file = getattr(chunk, "source_file", None)
    page = getattr(chunk, "page", None)
    return {
        "rank": rank,
        "source_file": source_file,
        "page": page,
        "match": bool(source_file == expected["source_file"] and page == expected["page"]),
        "text_preview": " ".join(str(chunk).split())[:PREVIEW_CHARS],
    }


def main():
    print("=" * 78)
    print("RECALL@k RETRIEVAL EVALUATION")
    print("=" * 78)

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    print(f"Ground truth : {len(ground_truth)} questions "
          f"({GROUND_TRUTH_PATH.relative_to(ROOT)})\n")

    print("Building combined index across all four documents:")
    index, stored_chunks, doc_stats = build_combined_index()

    # How many chunks exist for each (file, page) the ground truth points at.
    # A zero here means the target page produced no chunks at all, so no
    # retriever could ever hit it — reported, not corrected.
    available = {}
    for c in stored_chunks:
        available[(c.source_file, c.page)] = available.get((c.source_file, c.page), 0) + 1

    results = []
    hits = {k: 0 for k in K_VALUES}

    for entry in ground_truth:
        expected = {"source_file": entry["source_file"], "page": entry["page"]}

        retrieved_by_k = {}
        hit_by_k = {}
        for k in K_VALUES:
            chunks = retrieve(entry["question"], index, stored_chunks, k=k)
            records = [describe(c, i + 1, expected) for i, c in enumerate(chunks)]
            retrieved_by_k[f"k={k}"] = records
            hit_by_k[k] = any(r["match"] for r in records)
            if hit_by_k[k]:
                hits[k] += 1

        # Rank of the first correct chunk within the deepest k searched.
        deepest = retrieved_by_k[f"k={max(K_VALUES)}"]
        first_correct = next((r["rank"] for r in deepest if r["match"]), None)

        results.append({
            "question": entry["question"],
            "answer": entry["answer"],
            "expected": expected,
            "chunks_available_for_expected_page": available.get(
                (expected["source_file"], expected["page"]), 0),
            "hit_at_1": hit_by_k[1],
            "hit_at_3": hit_by_k[3],
            "hit_at_5": hit_by_k[5],
            "first_correct_rank": first_correct,
            "retrieved": retrieved_by_k,
        })

    total = len(ground_truth)
    recall = {f"recall_at_{k}": (hits[k] / total if total else 0.0) for k in K_VALUES}

    # ---- per-question detail -------------------------------------------------
    print("=" * 78)
    print("PER-QUESTION RESULTS")
    print("=" * 78)
    for i, r in enumerate(results, start=1):
        flags = " ".join(
            f"@{k}:{'HIT ' if r[f'hit_at_{k}'] else 'MISS'}" for k in K_VALUES
        )
        print(f"\nQ{i}. {r['question']}")
        print(f"    expected : {r['expected']['source_file']} p.{r['expected']['page']}"
              f"   ({r['chunks_available_for_expected_page']} chunk(s) exist for that page)")
        print(f"    result   : {flags}"
              + (f"   first correct at rank {r['first_correct_rank']}"
                 if r["first_correct_rank"] else ""))
        print(f"    retrieved (top {max(K_VALUES)}):")
        for rec in r["retrieved"][f"k={max(K_VALUES)}"]:
            mark = "**" if rec["match"] else "  "
            print(f"      {mark} #{rec['rank']}  {rec['source_file']} p.{rec['page']}")
            print(f"           {rec['text_preview'][:110]}...")

    # ---- misses --------------------------------------------------------------
    misses = [r for r in results if not r["hit_at_5"]]
    print()
    print("=" * 78)
    print(f"MISSES AT k=5  ({len(misses)} of {total})")
    print("=" * 78)
    if not misses:
        print("  none — every question hit within the top 5")
    for r in misses:
        print(f"\n  Q: {r['question']}")
        print(f"     expected : {r['expected']['source_file']} p.{r['expected']['page']}")
        print(f"     got      : " + ", ".join(
            f"#{rec['rank']} {rec['source_file']} p.{rec['page']}"
            for rec in r["retrieved"]["k=5"]))

    # ---- summary -------------------------------------------------------------
    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k in K_VALUES:
        print(f"  Recall@{k} = {hits[k]}/{total} = {recall[f'recall_at_{k}']:.3f}")

    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "documents": doc_stats,
        "total_chunks": len(stored_chunks),
        "total_questions": total,
        "k_values": K_VALUES,
        "hits": {f"k={k}": hits[k] for k in K_VALUES},
        "recall": recall,
        "results": results,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n  full per-question results -> {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
