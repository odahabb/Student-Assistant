"""
backend/scripts/embedder_comparison.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Compares embedding models for retrieval on the 25-question ground truth.

Every earlier diagnostic points at the embedding model: the correct chunk
loses to other body text from the same paper (competitor_analysis.json), and
chunking changes move only one or two questions (retrieval_variants.json).
all-MiniLM-L6-v2 was never compared against alternatives, so this script
does that, holding everything else fixed:

  all-MiniLM-L6-v2           the pipeline's model (22M parameters)
  multi-qa-MiniLM-L6-cos-v1  same architecture, trained for question answering
  bge-small-en-v1.5          a newer retrieval model of similar size (33M)
  all-mpnet-base-v2          a larger general-purpose model (110M)

Chunks come from preprocess() exactly as the pipeline makes them (chunk
boundaries are set with the MiniLM tokenizer, and semantic breaks with MiniLM
sentence vectors, so each model sees identical text), for all four chunking
modes: fixed windows, sentence-aware, heading-aware and semantic. bge expects
an instruction in front of queries; the other models take text as-is. Vectors are normalised
and searched exactly, so the ranking is by cosine similarity, which matches
IndexFlatL2 on unit vectors. Hit rule as in eval_recall.py: a retrieved
chunk from the expected file and page.

Exact McNemar tests compare each model with MiniLM on the same chunks.

Run from anywhere:
    python backend/scripts/embedder_comparison.py
"""

import json
import os
import sys
import time
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from backend.pipeline.device import get_torch_device  # noqa: E402
from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
OUT_PATH = EVAL_DIR / "embedder_comparison.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
MODELS = {
    "all-MiniLM-L6-v2": {"id": "sentence-transformers/all-MiniLM-L6-v2",
                         "query_prefix": ""},
    "multi-qa-MiniLM-L6-cos-v1": {"id": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                                  "query_prefix": ""},
    "bge-small-en-v1.5": {"id": "BAAI/bge-small-en-v1.5",
                          "query_prefix": "Represent this sentence for searching "
                                          "relevant passages: "},
    "all-mpnet-base-v2": {"id": "sentence-transformers/all-mpnet-base-v2",
                          "query_prefix": ""},
}
BASELINE = "all-MiniLM-L6-v2"
CHUNKINGS = ["window", "sentence", "heading", "semantic"]
K_VALUES = [1, 3, 5]


def mcnemar_p(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(gained, lost) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def evaluate(model, prefix, chunks, ground_truth):
    t0 = time.time()
    doc_vecs = model.encode([str(c) for c in chunks], normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False,
                            batch_size=32)
    encode_sec = time.time() - t0
    q_vecs = model.encode([prefix + e["question"] for e in ground_truth],
                          normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False)

    ranks = []
    for q, entry in zip(q_vecs, ground_truth):
        expected = (entry["source_file"], entry["page"])
        top = np.argsort(-(doc_vecs @ q))[:max(K_VALUES)]
        ranks.append(next((r for r, i in enumerate(top, 1)
                           if (chunks[i].source_file, chunks[i].page) == expected), None))

    total = len(ground_truth)
    return {
        "recall": {f"recall_at_{k}": round(sum(1 for r in ranks if r and r <= k) / total, 3)
                   for k in K_VALUES},
        "mrr_top5": round(float(np.mean([1 / r if r else 0 for r in ranks])), 4),
        "encode_sec_per_100_chunks": round(100 * encode_sec / len(chunks), 3),
        "ranks": ranks,
    }


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    pages = [load_file(str(RAW_DIR / name)) for name in DOCUMENTS]
    chunk_sets = {mode: [c for p in pages for c in preprocess(p, chunking=mode)]
                  for mode in CHUNKINGS}
    device = get_torch_device()

    results = {}
    for name, spec in MODELS.items():
        model = SentenceTransformer(spec["id"], device=device)
        info = {
            "hf_id": spec["id"],
            "parameters_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "dimensions": model.get_sentence_embedding_dimension(),
            "max_seq_length": model.max_seq_length,
            "query_prefix": spec["query_prefix"],
        }
        results[name] = {**info, "by_chunking": {
            mode: evaluate(model, spec["query_prefix"], chunk_sets[mode], ground_truth)
            for mode in CHUNKINGS}}
        del model

    print(f"device: {device}\n")
    print(f"{'model':<27} {'params':>6} {'chunking':<9} {'R@1':>5} {'R@3':>5} "
          f"{'R@5':>5} {'MRR':>6} {'s/100':>6}  vs MiniLM at k=5 (gained/lost, p)")
    for name, r in results.items():
        for mode in CHUNKINGS:
            m = r["by_chunking"][mode]
            base = results[BASELINE]["by_chunking"][mode]["ranks"]
            for k in K_VALUES:
                hit = [x is not None and x <= k for x in m["ranks"]]
                base_hit = [x is not None and x <= k for x in base]
                gained = sum(a and not b for a, b in zip(hit, base_hit))
                lost = sum(b and not a for a, b in zip(hit, base_hit))
                m.setdefault("vs_baseline", {})[f"k={k}"] = {
                    "gained": gained, "lost": lost,
                    "mcnemar_p": round(mcnemar_p(gained, lost), 3)}
            v = m["vs_baseline"]["k=5"]
            rc = m["recall"]
            print(f"{name:<27} {r['parameters_millions']:>6} {mode:<9} "
                  f"{rc['recall_at_1']:>5} {rc['recall_at_3']:>5} {rc['recall_at_5']:>5} "
                  f"{m['mrr_top5']:>6} {m['encode_sec_per_100_chunks']:>6}  "
                  f"{v['gained']}/{v['lost']}, p={v['mcnemar_p']}")

    OUT_PATH.write_text(json.dumps({
        "note": "Embedding-model comparison on the 25-question retrieval ground "
                "truth. ranks: first matching rank in the top 5, null if none.",
        "device": device,
        "chunks": {mode: len(c) for mode, c in chunk_sets.items()},
        "models": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
