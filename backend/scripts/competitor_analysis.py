"""
backend/scripts/competitor_analysis.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

What is outranking the correct chunk?

rank_diagnostics.py shows that misses are chunks losing on similarity, and the
single-document ablation shows the winners come from the same document. This
script looks at WHAT those winning chunks are, so the retrieval fix is chosen
from evidence rather than assumed:

  - reference_list : bibliography text (citation years, arXiv/DOI/URLs, et al.)
  - near_page      : same document, within one page of the expected page
  - same_document  : same document, further away
  - other_document : a different PDF

It also records, per question, whether the ground-truth answer text sits whole
inside one chunk on the expected page or is split across a chunk boundary, and
a counterfactual: Recall@k if reference-list chunks were removed from the
candidates. Nothing in the pipeline is changed.

Run from anywhere:
    python backend/scripts/competitor_analysis.py
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.embedder import embed, _get_model  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
OUT_PATH = EVAL_DIR / "competitor_analysis.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
K_VALUES = [1, 3, 5]
TOP_K = 5
PREVIEW_CHARS = 160

# Chunk text is tokenizer-decoded: lowercase, with spaces around punctuation,
# e.g. "( 2024 )", "https : / / doi. org".
_REF_PATTERNS = [
    r"\(\s*(?:19|20)\d\d\s*[a-z]?\s*\)",   # (2024)
    r"\barxiv\b",
    r"\bdoi\b",
    r"\bhttps?\b",
    r"\bet al\b",
    r"\bproceedings\b",
    r"\bin advances in\b",
    r"\bpreprint\b",
    r"\bconference\b",
    r"\bjournal\b",
]
REF_THRESHOLD = 4


def reference_signals(text: str) -> int:
    t = " ".join(str(text).lower().split())
    return sum(len(re.findall(p, t)) for p in _REF_PATTERNS)


def is_reference_chunk(text: str) -> bool:
    return reference_signals(text) >= REF_THRESHOLD


def _squash(text: str) -> str:
    """Letters and digits only, so '3,197' and '3, 197' compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def category(chunk, expected) -> str:
    if is_reference_chunk(chunk):
        return "reference_list"
    if chunk.source_file != expected[0]:
        return "other_document"
    if chunk.page is not None and abs(chunk.page - expected[1]) <= 1:
        return "near_page"
    return "same_document"


def main():
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))

    chunks = []
    for name in DOCUMENTS:
        chunks.extend(preprocess(load_file(str(RAW_DIR / name))))
    embeddings = embed(chunks)
    model = _get_model()
    q_vecs = np.asarray(model.encode([e["question"] for e in ground_truth],
                                     convert_to_numpy=True, show_progress_bar=False),
                        dtype=np.float32)

    ref_mask = np.array([is_reference_chunk(c) for c in chunks])
    print(f"{len(chunks)} chunks, {int(ref_mask.sum())} classified as reference-list")

    rows = []
    hits = {"baseline": Counter(), "reference_filtered": Counter()}
    miss_categories = Counter()

    for qi, entry in enumerate(ground_truth):
        expected = (entry["source_file"], entry["page"])
        on_page = [i for i, c in enumerate(chunks)
                   if (c.source_file, c.page) == expected]
        sims = embeddings @ q_vecs[qi]            # unit-norm => cosine
        order = [int(i) for i in np.argsort(-sims)]

        def first_rank(candidates):
            for rank, i in enumerate(candidates, 1):
                if i in on_page:
                    return rank
            return None

        base_rank = first_rank(order)
        filtered_rank = first_rank([i for i in order if not ref_mask[i]])
        for k in K_VALUES:
            hits["baseline"][k] += bool(base_rank and base_rank <= k)
            hits["reference_filtered"][k] += bool(filtered_rank and filtered_rank <= k)

        best_correct = max(on_page, key=lambda i: float(sims[i]))
        above = [i for i in order[:order.index(best_correct)]]
        top_above = [{
            "rank": r + 1,
            "source_file": chunks[i].source_file,
            "page": chunks[i].page,
            "cosine": round(float(sims[i]), 4),
            "category": category(chunks[i], expected),
            "reference_signals": reference_signals(chunks[i]),
            "text_preview": " ".join(str(chunks[i]).split())[:PREVIEW_CHARS],
        } for r, i in enumerate(above[:TOP_K])]

        answer = _squash(entry["answer"])
        in_chunk = [i for i in on_page if answer in _squash(chunks[i])]
        page_text = _squash(" ".join(str(chunks[i]) for i in on_page))
        if in_chunk:
            placement = "whole_in_one_chunk"
        elif answer in page_text:
            placement = "split_across_chunks"
        else:
            placement = "paraphrased_or_not_verbatim"

        is_miss = not (base_rank and base_rank <= TOP_K)
        if is_miss:
            miss_categories.update(a["category"] for a in top_above)

        rows.append({
            "question": entry["question"],
            "answer": entry["answer"],
            "expected": {"source_file": expected[0], "page": expected[1]},
            "chunks_on_expected_page": len(on_page),
            "answer_placement": placement,
            "correct_chunk_is_reference_list": bool(ref_mask[best_correct]),
            "rank_baseline": base_rank,
            "rank_reference_filtered": filtered_rank,
            "miss_at_5": is_miss,
            "chunks_above_correct": len(above),
            "category_counts_above_correct": dict(Counter(
                category(chunks[i], expected) for i in above)),
            "top_chunks_above_correct": top_above,
        })

    total = len(ground_truth)
    recall = {name: {f"recall_at_{k}": round(c[k] / total, 3) for k in K_VALUES}
              for name, c in hits.items()}
    placements = Counter(r["answer_placement"] for r in rows)

    print("\nRecall@k:")
    for name, r in recall.items():
        print(f"  {name:<20} {r}")
    print(f"\nTop-{TOP_K} chunks outranking the correct one, over the "
          f"{sum(r['miss_at_5'] for r in rows)} misses: {dict(miss_categories)}")
    print(f"Answer placement over all {total}: {dict(placements)}")
    print("\nMisses:")
    for r in rows:
        if r["miss_at_5"]:
            cats = Counter(a["category"] for a in r["top_chunks_above_correct"])
            print(f"  {r['expected']['source_file'][:24]:<24} p.{r['expected']['page']:<3}"
                  f" above={r['chunks_above_correct']:<4} filtered_rank="
                  f"{str(r['rank_reference_filtered']):<5} {r['answer_placement']:<28}"
                  f" {dict(cats)}")

    payload = {
        "note": "Read-only diagnostic of what outranks the correct chunk. "
                "Pipeline and ground truth unchanged.",
        "reference_rule": {"patterns": _REF_PATTERNS, "threshold": REF_THRESHOLD},
        "total_chunks": len(chunks),
        "reference_list_chunks": int(ref_mask.sum()),
        "total_questions": total,
        "recall": recall,
        "miss_top5_competitor_categories": dict(miss_categories),
        "answer_placement_counts": dict(placements),
        "results": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
