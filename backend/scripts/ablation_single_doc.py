"""
backend/scripts/ablation_single_doc.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Single-document ablation for the Recall@k investigation.

Each ground-truth question is re-run against an index built from ONLY its own
source PDF's chunks, instead of the 584-chunk combined index. Comparing the
rank of the correct chunk in isolation against its rank in the combined index
separates two causes of low recall:

  - cross-document competition : rank improves sharply in isolation, i.e. the
    correct chunk was being outranked by chunks from the other three PDFs
  - intra-document dilution    : rank stays poor even alone, i.e. the correct
    chunk loses to other chunks within its own document

Read-only. Indexes are built in memory (as notebooks/easyocr_blip_vs_qwen2vl_eval
.ipynb does for its per-sample indexes) so the shared store at data/processed/
is left untouched. Nothing is tuned or corrected here.

Run from anywhere:
    python backend/scripts/ablation_single_doc.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.embedder import embed  # noqa: E402
from backend.pipeline.retriever import retrieve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
FULL_RESULTS_PATH = EVAL_DIR / "recall_results.json"      # read-only
OUT_PATH = EVAL_DIR / "recall_ablation_single_doc.json"   # new file

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
K_VALUES = [1, 3, 5]
PREVIEW_CHARS = 160


def build_per_document_indexes():
    """Chunk each PDF and build one in-memory IndexFlatL2 per document."""
    per_doc = {}
    for name in DOCUMENTS:
        chunks = preprocess(load_file(str(RAW_DIR / name)))
        embeddings = embed(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        per_doc[name] = {"index": index, "chunks": chunks}
        print(f"  {name:<52} {len(chunks):>4} chunks -> own index")
    return per_doc


def main():
    print("=" * 78)
    print("SINGLE-DOCUMENT ABLATION")
    print("=" * 78)

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    full = json.loads(FULL_RESULTS_PATH.read_text(encoding="utf-8"))
    full_by_question = {r["question"]: r for r in full["results"]}
    print(f"Ground truth : {len(ground_truth)} questions")
    print(f"Full-index baseline read from {FULL_RESULTS_PATH.relative_to(ROOT)} "
          f"({full['total_chunks']} chunks, recall {full['recall']})\n")

    print("Building one index per document:")
    per_doc = build_per_document_indexes()
    print()

    rows = []
    hits = {k: 0 for k in K_VALUES}

    for entry in ground_truth:
        source_file, page = entry["source_file"], entry["page"]
        bundle = per_doc[source_file]
        index, chunks = bundle["index"], bundle["chunks"]

        retrieved_by_k, hit_by_k = {}, {}
        for k in K_VALUES:
            got = retrieve(entry["question"], index, chunks, k=k)
            records = [{
                "rank": i + 1,
                "source_file": c.source_file,
                "page": c.page,
                "match": bool(c.source_file == source_file and c.page == page),
                "text_preview": " ".join(str(c).split())[:PREVIEW_CHARS],
            } for i, c in enumerate(got)]
            retrieved_by_k[f"k={k}"] = records
            hit_by_k[k] = any(r["match"] for r in records)
            if hit_by_k[k]:
                hits[k] += 1

        deepest = retrieved_by_k[f"k={max(K_VALUES)}"]
        single_rank = next((r["rank"] for r in deepest if r["match"]), None)

        base = full_by_question[entry["question"]]
        full_rank = base["first_correct_rank"]

        def rank_val(r):
            return r if r is not None else 99

        rows.append({
            "question": entry["question"],
            "answer": entry["answer"],
            "expected": {"source_file": source_file, "page": page},
            "document_chunks": len(chunks),
            "single_doc": {
                "hit_at_1": hit_by_k[1], "hit_at_3": hit_by_k[3],
                "hit_at_5": hit_by_k[5], "first_correct_rank": single_rank,
                "retrieved": retrieved_by_k,
            },
            "full_index": {
                "hit_at_1": base["hit_at_1"], "hit_at_3": base["hit_at_3"],
                "hit_at_5": base["hit_at_5"],
                "first_correct_rank": full_rank,
            },
            "rank_improvement": rank_val(full_rank) - rank_val(single_rank),
            "cross_document_interference": bool(
                single_rank is not None
                and (full_rank is None or full_rank > single_rank)
            ),
        })

    total = len(ground_truth)
    recall = {f"recall_at_{k}": hits[k] / total for k in K_VALUES}

    # ---- comparison table ----------------------------------------------------
    print("=" * 78)
    print("SINGLE-DOC RANK vs FULL-INDEX RANK (rank of first correct chunk)")
    print("=" * 78)
    print(f"{'#':<3}{'expected':<38}{'own-doc':>9}{'full-idx':>10}{'delta':>8}"
          f"{'doc chunks':>12}")
    for i, r in enumerate(rows, 1):
        e = r["expected"]
        exp = f"{e['source_file'][:24]} p.{e['page']}"
        s = r["single_doc"]["first_correct_rank"]
        f_ = r["full_index"]["first_correct_rank"]
        delta = ("+" + str(r["rank_improvement"])) if r["rank_improvement"] > 0 \
            else str(r["rank_improvement"])
        print(f"{i:<3}{exp:<38}{str(s) or '-':>9}{str(f_):>10}{delta:>8}"
              f"{r['document_chunks']:>12}")
    print("\n  '-'/None = correct chunk not in top 5;  delta = full rank - own-doc rank")
    print("  (delta computed with None treated as 99)")

    print()
    print("=" * 78)
    print("RECALL: SINGLE-DOC vs FULL INDEX")
    print("=" * 78)
    for k in K_VALUES:
        fk = full["recall"][f"recall_at_{k}"]
        sk = recall[f"recall_at_{k}"]
        print(f"  Recall@{k}:  full index {fk:.3f}  ->  own document only {sk:.3f}"
              f"   ({hits[k]}/{total})")

    improved = [r for r in rows if r["cross_document_interference"]]
    still_poor = [r for r in rows
                  if r["single_doc"]["first_correct_rank"] is None]
    print()
    print(f"  questions whose rank improved in isolation : {len(improved)}/{total}")
    print(f"  questions still missing at k=5 even alone  : {len(still_poor)}/{total}")

    payload = {
        "note": "Single-document ablation. Read-only diagnostic; official "
                "ground truth and recall_results.json are untouched.",
        "k_values": K_VALUES,
        "full_index_recall": full["recall"],
        "single_doc_recall": recall,
        "single_doc_hits": {f"k={k}": hits[k] for k in K_VALUES},
        "total_questions": total,
        "questions_rank_improved_in_isolation": len(improved),
        "questions_still_missing_at_k5_alone": len(still_poor),
        "results": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
