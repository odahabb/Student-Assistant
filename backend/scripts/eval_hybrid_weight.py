"""
backend/scripts/eval_hybrid_weight.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Chooses retriever.DENSE_WEIGHT, the share of the hybrid score that comes from
the embeddings rather than from BM25.

The value in use, 0.4, was carried over from a hybrid retriever built for a
different project and had never been tested on this one. This script sweeps
the whole range on the 25-question retrieval ground truth, using the same hit
rule as eval_recall.py: a retrieved chunk whose source_file and page both
match the labelled answer page.

Weight 0.0 is BM25 alone and weight 1.0 is dense retrieval alone, so the sweep
also reproduces both single-method baselines. Nothing is tuned beyond this one
number, and the ground truth is the same 25 questions used everywhere else —
with only 25 questions a difference of one question is 0.04 recall, so the
curve matters more than its highest point.

Run from anywhere:
    python backend/scripts/eval_hybrid_weight.py
    SA_EMBEDDER=bge-small python backend/scripts/eval_hybrid_weight.py
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

from backend.pipeline import sparse as sparse_module  # noqa: E402
from backend.pipeline.embedder import embed, model_key  # noqa: E402
from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import DENSE_WEIGHT, retrieve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
# The app's chunking. --chunking window measures the configuration the
# earlier recorded results used.
CHUNKING = (sys.argv[sys.argv.index("--chunking") + 1]
            if "--chunking" in sys.argv else "sentence")
# Which ground truth to sweep against. "papers" is the 25 questions over the
# four research papers; "slides" is the 38 text-layer questions over the
# fifteen lecture decks (build_slide_ground_truth.py). Running both matters: a
# weight chosen on one set of 25 questions is a fit to that set, and two
# independent sets agreeing is a different claim.
QUESTION_SET = (sys.argv[sys.argv.index("--set") + 1]
                if "--set" in sys.argv else "papers")
K_VALUES = [1, 3, 5]
WEIGHTS = [round(w / 20, 2) for w in range(21)]   # 0.00 .. 1.00 in 0.05 steps
OUT_PATH = EVAL_DIR / ("hybrid_weight_sweep"
                       + ("" if QUESTION_SET == "papers" else f"_{QUESTION_SET}")
                       + ("" if CHUNKING == "sentence" else f"_{CHUNKING}")
                       + ("" if model_key() == "bge-small" else f"_{model_key()}")
                       + ".json")


def covers(chunk, source_file, page) -> bool:
    """
    Whether a chunk holds the labelled page. Slide chunks pack several slides
    together, so the match is against the chunk's page range.
    """
    if getattr(chunk, "source_file", None) != source_file:
        return False
    first = getattr(chunk, "page", None)
    if first is None:
        return False
    return first <= page <= (getattr(chunk, "page_end", None) or first)


def corpus():
    """(chunks, questions, documents) for the chosen question set."""
    if QUESTION_SET == "slides":
        truth = json.loads((EVAL_DIR / "slide_ground_truth.json")
                           .read_text(encoding="utf-8"))
        questions = [q for q in truth["questions"] if q["source"] == "text_layer"]
        decks = sorted((ROOT / "data" / "projects" / "AI").glob("*.pdf"))
        chunks = []
        for path in decks:
            # The app's own chunking for slides: packed, not split.
            chunks.extend(preprocess(load_file(str(path), figures="auto")))
        return chunks, questions, len(decks)

    questions = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    chunks = []
    for name in DOCUMENTS:
        chunks.extend(preprocess(load_file(str(RAW_DIR / name)), chunking=CHUNKING))
    return chunks, questions, len(DOCUMENTS)


def score(ranks, total):
    return {
        "recall": {f"recall_at_{k}": round(sum(1 for r in ranks if r and r <= k) / total, 3)
                   for k in K_VALUES},
        "mrr_top5": round(mean(1 / r if r else 0 for r in ranks), 4),
        "ranks": ranks,
    }


def main():
    chunks, ground_truth, documents = corpus()
    print(f"Ground truth: {len(ground_truth)} questions ({QUESTION_SET})")
    print(f"Chunking: {'packed slides' if QUESTION_SET == 'slides' else CHUNKING}"
          f"   embedder: {model_key()}")
    print(f"{len(chunks)} chunks across {documents} documents")

    embeddings = embed(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    keywords = sparse_module.build_index(chunks)

    results = {}
    print(f"\n{'weight':>7} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>7}   note")
    for weight in WEIGHTS:
        ranks = []
        for entry in ground_truth:
            got = retrieve(entry["question"], index, chunks, k=max(K_VALUES),
                           sparse=keywords, dense_weight=weight)
            ranks.append(next((i for i, c in enumerate(got, 1)
                               if covers(c, entry["source_file"],
                                         entry["page"])), None))
        results[f"{weight:.2f}"] = score(ranks, len(ground_truth))
        r = results[f"{weight:.2f}"]["recall"]
        note = {0.0: "BM25 only", 1.0: "dense only",
                DENSE_WEIGHT: "in use"}.get(weight, "")
        print(f"{weight:>7.2f} {r['recall_at_1']:>6} {r['recall_at_3']:>6} "
              f"{r['recall_at_5']:>6} {results[f'{weight:.2f}']['mrr_top5']:>7}   {note}")

    best_at_3 = max(results, key=lambda w: (results[w]["recall"]["recall_at_3"],
                                            results[w]["mrr_top5"]))
    in_use = results[f"{DENSE_WEIGHT:.2f}"]
    # Every weight whose Recall@3 equals the best: with 25 questions the
    # choice inside this band is not supported by the data.
    top = results[best_at_3]["recall"]["recall_at_3"]
    band = [w for w, r in results.items() if r["recall"]["recall_at_3"] == top]
    print(f"\n  best Recall@3 {top} at weight {best_at_3} "
          f"(MRR {results[best_at_3]['mrr_top5']})")
    print(f"  weight {DENSE_WEIGHT:.2f} in use: Recall@3 {in_use['recall']['recall_at_3']}, "
          f"MRR {in_use['mrr_top5']}")
    print(f"  tied at the best Recall@3: {', '.join(band)}")

    OUT_PATH.write_text(json.dumps({
        "note": "Sweep of retriever.DENSE_WEIGHT (share of the hybrid score "
                "taken from the embeddings) on the 25-question retrieval "
                "ground truth. 0.0 is BM25 alone, 1.0 is dense alone. ranks: "
                "first matching rank in the top 5, null if none.",
        "questions": len(ground_truth),
        "question_set": QUESTION_SET,
        "chunking": "packed slides" if QUESTION_SET == "slides" else CHUNKING,
        "embedder": model_key(),
        "chunks": len(chunks),
        "weight_in_use": DENSE_WEIGHT,
        "best_recall_at_3_weight": best_at_3,
        "tied_at_best_recall_at_3": band,
        "weights": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
