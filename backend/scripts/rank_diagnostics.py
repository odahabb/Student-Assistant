"""
backend/scripts/rank_diagnostics.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Rank diagnostics for the Recall@k investigation.

Computes, over the existing 8-question full-index results:
  - Mean Reciprocal Rank (1/rank of first correct hit, 0 if outside top 5)
  - per question: token length of the correct chunk, and the cosine similarity
    gap between (question, correct chunk) and (question, top-ranked incorrect
    chunk)

"Correct chunk" is the best-scoring chunk on the expected page — a page usually
yields several chunks, and the fairest measure of what retrieval was up against
is the one closest to the question. The gap is therefore an upper bound on how
close the target page got.

Embeddings are unit-norm (the model ends in a Normalize layer), so cosine
similarity is a plain dot product.

Read-only: recall_results.json and retrieval_ground_truth.json are only read.

Run from anywhere:
    python backend/scripts/rank_diagnostics.py
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
from backend.pipeline.embedder import embed, _get_model  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
FULL_RESULTS_PATH = EVAL_DIR / "recall_results.json"   # read-only
OUT_PATH = EVAL_DIR / "rank_diagnostics.json"          # new file

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
TOP_K = 5


def main():
    print("=" * 78)
    print("RANK DIAGNOSTICS (MRR, chunk length, similarity gap)")
    print("=" * 78)

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    full = json.loads(FULL_RESULTS_PATH.read_text(encoding="utf-8"))
    full_by_question = {r["question"]: r for r in full["results"]}

    # ---- MRR from the existing results (no retrieval re-run needed) ----------
    reciprocals = []
    for entry in ground_truth:
        rank = full_by_question[entry["question"]]["first_correct_rank"]
        reciprocals.append(1.0 / rank if rank else 0.0)
    mrr = sum(reciprocals) / len(reciprocals)
    print(f"\nMRR over {len(reciprocals)} questions (top-{TOP_K} cutoff): {mrr:.4f}")

    # ---- rebuild vectors so similarities can be measured ---------------------
    all_chunks = []
    for name in DOCUMENTS:
        all_chunks.extend(preprocess(load_file(str(RAW_DIR / name))))
    embeddings = embed(all_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    print(f"Rebuilt {len(all_chunks)} chunk vectors {embeddings.shape} "
          f"for similarity measurement")

    model = _get_model()
    tokenizer = model.tokenizer
    questions = [e["question"] for e in ground_truth]
    q_vecs = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    q_vecs = np.asarray(q_vecs, dtype=np.float32)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Chunk vector norms: min {norms.min():.4f} max {norms.max():.4f} "
          f"(unit-norm => cosine == dot product)")

    rows = []
    for qi, entry in enumerate(ground_truth):
        q = q_vecs[qi]
        expected = (entry["source_file"], entry["page"])

        sims = embeddings @ q                      # cosine, vectors are unit-norm
        on_page = [i for i, c in enumerate(all_chunks)
                   if (c.source_file, c.page) == expected]
        best_correct = max(on_page, key=lambda i: float(sims[i]))

        # highest-ranked chunk that is NOT on the expected page
        order = np.argsort(-sims)
        top_incorrect = next(int(i) for i in order
                             if (all_chunks[i].source_file, all_chunks[i].page) != expected)

        correct_sim = float(sims[best_correct])
        incorrect_sim = float(sims[top_incorrect])
        gap = correct_sim - incorrect_sim

        chunk = all_chunks[best_correct]
        token_len = len(tokenizer.encode(str(chunk), add_special_tokens=False))

        base = full_by_question[entry["question"]]
        rank = base["first_correct_rank"]

        # how many chunks beat the best correct chunk
        beaten_by = int(np.sum(sims > correct_sim))

        rows.append({
            "question": entry["question"],
            "expected": {"source_file": expected[0], "page": expected[1]},
            "first_correct_rank_full_index": rank,
            "reciprocal_rank": (1.0 / rank if rank else 0.0),
            "hit_at_5": base["hit_at_5"],
            "correct_chunk": {
                "index_position": best_correct,
                "token_length_minilm": token_len,
                "char_length": len(str(chunk)),
                "cosine_to_question": round(correct_sim, 4),
                "text_preview": " ".join(str(chunk).split())[:160],
            },
            "top_incorrect_chunk": {
                "index_position": top_incorrect,
                "source_file": all_chunks[top_incorrect].source_file,
                "page": all_chunks[top_incorrect].page,
                "cosine_to_question": round(incorrect_sim, 4),
                "same_document": bool(
                    all_chunks[top_incorrect].source_file == expected[0]),
                "text_preview": " ".join(str(all_chunks[top_incorrect]).split())[:160],
            },
            "cosine_gap_correct_minus_incorrect": round(gap, 4),
            "chunks_ranked_above_correct": beaten_by,
        })

    # ---- report --------------------------------------------------------------
    print()
    print("=" * 78)
    print("PER-QUESTION DIAGNOSTICS")
    print("=" * 78)
    print(f"{'#':<3}{'expected':<30}{'rank':>6}{'RR':>6}{'tok':>6}"
          f"{'cos(correct)':>14}{'cos(top-wrong)':>16}{'gap':>9}{'above':>7}")
    for i, r in enumerate(rows, 1):
        e = r["expected"]
        exp = f"{e['source_file'][:18]} p.{e['page']}"
        print(f"{i:<3}{exp:<30}{str(r['first_correct_rank_full_index']):>6}"
              f"{r['reciprocal_rank']:>6.2f}"
              f"{r['correct_chunk']['token_length_minilm']:>6}"
              f"{r['correct_chunk']['cosine_to_question']:>14.4f}"
              f"{r['top_incorrect_chunk']['cosine_to_question']:>16.4f}"
              f"{r['cosine_gap_correct_minus_incorrect']:>9.4f}"
              f"{r['chunks_ranked_above_correct']:>7}")
    print("\n  gap < 0 means the best correct chunk scored BELOW the top wrong chunk")
    print("  'above' = how many of the 584 chunks outscored the best correct chunk")

    # ---- simple correlations -------------------------------------------------
    hits = np.array([1.0 if r["hit_at_5"] else 0.0 for r in rows])
    lengths = np.array([r["correct_chunk"]["token_length_minilm"] for r in rows], dtype=float)
    gaps = np.array([r["cosine_gap_correct_minus_incorrect"] for r in rows])

    def safe_corr(a, b):
        if a.std() == 0 or b.std() == 0:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    corr_len_hit = safe_corr(lengths, hits)
    corr_gap_hit = safe_corr(gaps, hits)
    corr_len_gap = safe_corr(lengths, gaps)

    print()
    print("=" * 78)
    print(f"CORRELATIONS  (n={len(rows)} - indicative only, small sample)")
    print("=" * 78)
    print(f"  correct-chunk token length  vs hit@5 : "
          f"{corr_len_hit:+.3f}" if corr_len_hit is not None else "  n/a")
    print(f"  cosine gap                  vs hit@5 : "
          f"{corr_gap_hit:+.3f}" if corr_gap_hit is not None else "  n/a")
    print(f"  correct-chunk token length  vs gap   : "
          f"{corr_len_gap:+.3f}" if corr_len_gap is not None else "  n/a")

    hit_rows = [r for r in rows if r["hit_at_5"]]
    miss_rows = [r for r in rows if not r["hit_at_5"]]

    def mean(rs, f):
        return sum(f(r) for r in rs) / len(rs) if rs else float("nan")

    print()
    print(f"  HITS  (n={len(hit_rows)}): mean token length "
          f"{mean(hit_rows, lambda r: r['correct_chunk']['token_length_minilm']):.0f}, "
          f"mean gap {mean(hit_rows, lambda r: r['cosine_gap_correct_minus_incorrect']):+.4f}")
    print(f"  MISSES(n={len(miss_rows)}): mean token length "
          f"{mean(miss_rows, lambda r: r['correct_chunk']['token_length_minilm']):.0f}, "
          f"mean gap {mean(miss_rows, lambda r: r['cosine_gap_correct_minus_incorrect']):+.4f}")

    payload = {
        "note": "Rank diagnostics. Read-only; official ground truth and "
                "recall_results.json untouched. 'Correct chunk' = best-scoring "
                "chunk on the expected page.",
        "mrr_top5": round(mrr, 4),
        "total_questions": len(rows),
        "correlations_indicative_only": {
            "n": len(rows),
            "correct_chunk_token_length_vs_hit_at_5": corr_len_hit,
            "cosine_gap_vs_hit_at_5": corr_gap_hit,
            "correct_chunk_token_length_vs_cosine_gap": corr_len_gap,
        },
        "group_means": {
            "hits": {
                "n": len(hit_rows),
                "mean_token_length": mean(hit_rows, lambda r: r["correct_chunk"]["token_length_minilm"]),
                "mean_cosine_gap": mean(hit_rows, lambda r: r["cosine_gap_correct_minus_incorrect"]),
            },
            "misses": {
                "n": len(miss_rows),
                "mean_token_length": mean(miss_rows, lambda r: r["correct_chunk"]["token_length_minilm"]),
                "mean_cosine_gap": mean(miss_rows, lambda r: r["cosine_gap_correct_minus_incorrect"]),
            },
        },
        "results": rows,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
