"""
backend/scripts/retrieval_variants.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Compares retrieval variants against the baseline on the 25-question ground
truth, using the same hit rule as eval_recall.py (a retrieved chunk from the
expected file and page).

Variants (2 x 2):
  chunking         window   — fixed 220-token windows, 40-token overlap
                              (the baseline)
                   sentence — whole sentences packed up to 220 tokens
  section context  off      — chunks embedded as they are (the baseline)
                   on       — "<section title>. <chunk>" embedded instead;
                              stored text and generator input unchanged

Why these two: competitor_analysis.json shows the chunks that beat the
correct one are mostly ordinary body text from the same paper, not
bibliography text, and no answer in the set is split across a chunk
boundary. Sentence chunking was the fix the draft report committed to, so it
is measured anyway; section context targets the observed failure directly by
telling otherwise similar chunks from one paper apart.

Indexes are built in memory; data/processed is not touched.

Run from anywhere:
    python backend/scripts/retrieval_variants.py
"""

import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend.pipeline.embedder import _get_model, embed  # noqa: E402
from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import retrieve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
OUT_PATH = EVAL_DIR / "retrieval_variants.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
K_VALUES = [1, 3, 5]
VARIANTS = [
    ("baseline", "window", False),
    ("section_context", "window", True),
    ("sentence_chunking", "sentence", False),
    ("sentence_chunking+section_context", "sentence", True),
]


def evaluate(pages, ground_truth, chunking, section_context):
    chunks = []
    for doc_pages in pages:
        chunks.extend(preprocess(doc_pages, chunking=chunking))
    embeddings = embed(chunks, section_context=section_context)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    tokenizer = _get_model().tokenizer
    ranks = []
    for entry in ground_truth:
        expected = (entry["source_file"], entry["page"])
        got = retrieve(entry["question"], index, chunks, k=max(K_VALUES))
        rank = next((i for i, c in enumerate(got, 1)
                     if (c.source_file, c.page) == expected), None)
        ranks.append(rank)

    total = len(ground_truth)
    return {
        "chunks": len(chunks),
        "mean_chunk_tokens": round(mean(
            len(tokenizer.encode(str(c), add_special_tokens=False)) for c in chunks), 1),
        "recall": {f"recall_at_{k}": round(sum(1 for r in ranks if r and r <= k) / total, 3)
                   for k in K_VALUES},
        "mrr_top5": round(mean(1 / r if r else 0 for r in ranks), 4),
        "ranks": ranks,
    }


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    pages = [load_file(str(RAW_DIR / name)) for name in DOCUMENTS]

    results = {}
    for name, chunking, section_context in VARIANTS:
        results[name] = {"chunking": chunking, "section_context": section_context,
                         **evaluate(pages, ground_truth, chunking, section_context)}

    base = results["baseline"]["ranks"]
    print(f"{'variant':<36} {'chunks':>6} {'tok':>6} {'R@1':>6} {'R@3':>6} "
          f"{'R@5':>6} {'MRR':>7}  gained / lost at k=5")
    for name, r in results.items():
        hit = [x is not None for x in r["ranks"]]
        base_hit = [x is not None for x in base]
        r["gained_at_5"] = [i + 1 for i, (a, b) in enumerate(zip(hit, base_hit)) if a and not b]
        r["lost_at_5"] = [i + 1 for i, (a, b) in enumerate(zip(hit, base_hit)) if b and not a]
        rc = r["recall"]
        print(f"{name:<36} {r['chunks']:>6} {r['mean_chunk_tokens']:>6} "
              f"{rc['recall_at_1']:>6} {rc['recall_at_3']:>6} {rc['recall_at_5']:>6} "
              f"{r['mrr_top5']:>7}  {r['gained_at_5']} / {r['lost_at_5']}")

    OUT_PATH.write_text(json.dumps({
        "note": "Retrieval variants on the 25-question ground truth. Question "
                "numbers in gained/lost are 1-based positions in "
                "retrieval_ground_truth.json. ranks: first matching rank in the "
                "top 5, null if none.",
        "questions": len(ground_truth),
        "variants": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
