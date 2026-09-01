"""
backend/scripts/eval_slide_chunking.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Tests the decision that slides are *packed* rather than split.

Prose pages hold more text than a chunk, so the chunker cuts them up. A slide
holds far less, and the first version of this system embedded each slide on
its own: chunks of a median 19 tokens, most of them a title and three bullet
points, which is too little context for an embedding to mean much.
preprocessor._pack_slides now packs consecutive slides up to the same
220-token limit and starts a new chunk when the slide title changes.

Four ways of chunking the same fifteen decks are compared on the slide ground
truth (build_slide_ground_truth.py), whose questions were written from single
slides before any chunker ran, so no strategy is favoured by construction:

  packed     — the system's rule: pack until full or until the title changes
  per_slide  — one chunk per slide, the original behaviour
  window     — treat a deck like prose: fixed 220-token windows, 40 overlap
  sentence   — treat a deck like prose: sentence-aware windows

A question is a hit at k when one of the top k chunks comes from the labelled
file and its page range covers the labelled slide. Both retrieval modes the
app can run are reported, because a packing rule that only looks good with
BM25 alongside it would be a different claim.

Only text_layer questions are used; the picture questions are what
eval_figure_reading.py measures. Pictures are still read when building the
index, as the app reads them, so every mode sees the same text.

Run from anywhere:
    SA_EMBEDDER=bge-small python backend/scripts/eval_slide_chunking.py
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

from backend.pipeline import sparse as sparse_module  # noqa: E402
from backend.pipeline.chunk import Chunk  # noqa: E402
from backend.pipeline.embedder import _get_model, embed, model_key  # noqa: E402
from backend.pipeline.loader import load_pdf  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import DENSE_WEIGHT, retrieve  # noqa: E402

DECK_DIR = ROOT / "data" / "projects" / "AI"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "slide_ground_truth.json"
OUT_PATH = EVAL_DIR / "slide_chunking.json"

K_VALUES = [1, 3, 5]
MODES = ["packed", "per_slide", "window", "sentence"]


def chunks_for(mode: str, pages_by_deck):
    """Every deck chunked one way, as one list."""
    chunks = []
    for pages in pages_by_deck:
        if mode == "packed":
            chunks.extend(preprocess(pages))          # kind == "slide" -> packed
        elif mode == "per_slide":
            chunks.extend(Chunk(" ".join(p["text"].split()), p["source_file"],
                                p["page"], p.get("section"), kind="slide")
                          for p in pages if p["text"].strip())
        else:
            # Hide the fact that these are slides, so the prose path runs.
            as_prose = [{k: v for k, v in p.items() if k != "kind"} for p in pages]
            chunks.extend(preprocess(as_prose, chunking=mode))
    return chunks


def covers(chunk, source_file, page) -> bool:
    if getattr(chunk, "source_file", None) != source_file:
        return False
    first = getattr(chunk, "page", None)
    if first is None:
        return False
    last = getattr(chunk, "page_end", None) or first
    return first <= page <= last


def slides_in(chunks) -> int:
    """How many distinct slides a set of chunks spans."""
    seen = set()
    for c in chunks:
        first = getattr(c, "page", None)
        if first is None:
            continue
        last = getattr(c, "page_end", None) or first
        seen.update((getattr(c, "source_file", None), p)
                    for p in range(first, last + 1))
    return len(seen)


def evaluate(chunks, questions, keywords, index, hybrid: bool):
    ranks, covered = [], []
    for entry in questions:
        got = retrieve(entry["question"], index, chunks, k=max(K_VALUES),
                       sparse=keywords if hybrid else None,
                       dense_weight=DENSE_WEIGHT)
        ranks.append(next((i for i, c in enumerate(got, 1)
                           if covers(c, entry["source_file"], entry["page"])), None))
        covered.append(slides_in(got[:3]))
    total = len(questions)
    return {
        "recall": {f"recall_at_{k}": round(sum(1 for r in ranks if r and r <= k) / total, 3)
                   for k in K_VALUES},
        "mrr_top5": round(statistics.mean(1 / r if r else 0 for r in ranks), 4),
        # How much of the deck the top 3 chunks put in front of the reader. A
        # packed chunk spans several slides, so it has more chances to cover
        # the labelled one: recall alone would flatter it, and this is the
        # measure that says by how much.
        "slides_covered_at_3": round(statistics.mean(covered), 1),
        "ranks": ranks,
    }


def main():
    # Slide text carries Unicode the Windows console cannot encode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    questions = [q for q in truth["questions"] if q["source"] == "text_layer"]
    print(f"{len(questions)} text-layer questions over {truth['decks']} decks")
    print(f"embedder: {model_key()}   dense weight: {DENSE_WEIGHT}\n")

    paths = sorted(p for p in DECK_DIR.glob("*.pdf"))
    pages_by_deck = [load_pdf(str(p), figures="auto") for p in paths]
    tokenizer = _get_model().tokenizer

    results = {}
    print(f"{'mode':<12} {'chunks':>7} {'median tok':>11} "
          f"{'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>7} {'slides':>7}   retrieval")
    for mode in MODES:
        t0 = time.time()
        chunks = chunks_for(mode, pages_by_deck)
        lengths = [len(tokenizer.encode(str(c), add_special_tokens=False))
                   for c in chunks]
        embeddings = embed(chunks)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        keywords = sparse_module.build_index(chunks)

        results[mode] = {
            "chunks": len(chunks),
            "median_tokens": round(statistics.median(lengths), 1),
            "mean_tokens": round(statistics.mean(lengths), 1),
            "build_seconds": round(time.time() - t0, 1),
            "hybrid": evaluate(chunks, questions, keywords, index, True),
            "dense_only": evaluate(chunks, questions, keywords, index, False),
        }
        for label in ("hybrid", "dense_only"):
            r = results[mode][label]
            print(f"{mode if label == 'hybrid' else '':<12} "
                  f"{len(chunks) if label == 'hybrid' else '':>7} "
                  f"{results[mode]['median_tokens'] if label == 'hybrid' else '':>11} "
                  f"{r['recall']['recall_at_1']:>6} {r['recall']['recall_at_3']:>6} "
                  f"{r['recall']['recall_at_5']:>6} {r['mrr_top5']:>7} "
                  f"{r['slides_covered_at_3']:>7}   {label}")

    base = results["per_slide"]["hybrid"]["ranks"]
    packed = results["packed"]["hybrid"]["ranks"]
    gained = [i + 1 for i, (a, b) in enumerate(zip(packed, base))
              if (a and a <= 3) and not (b and b <= 3)]
    lost = [i + 1 for i, (a, b) in enumerate(zip(packed, base))
            if (b and b <= 3) and not (a and a <= 3)]
    print(f"\n  packed vs per_slide at k=3 (hybrid): gained {gained}, lost {lost}")

    OUT_PATH.write_text(json.dumps({
        "note": "Slide chunking strategies on the slide ground truth. A hit at "
                "k means a top-k chunk is from the labelled file and its page "
                "range covers the labelled slide. ranks: first such rank in "
                "the top 5, null if none. Question numbers in gained/lost are "
                "1-based positions in the text_layer question list. "
                "slides_covered_at_3 is how many distinct slides the top 3 "
                "chunks span on average: a packed chunk covers several, so it "
                "has more chances to satisfy the hit rule.",
        "questions": len(questions),
        "decks": len(paths),
        "embedder": model_key(),
        "dense_weight": DENSE_WEIGHT,
        "packed_vs_per_slide_at_3": {"gained": gained, "lost": lost},
        "modes": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
