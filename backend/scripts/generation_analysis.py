"""
backend/scripts/generation_analysis.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Splits wrong answers into retrieval failures and generation failures.

For each ground-truth question the full pipeline runs end to end at the same k
the Streamlit app uses (TOP_K = 3), and two independent facts are recorded:

  retrieved_correct : was a chunk matching the expected source_file + page
                      among the chunks actually handed to the generator
  answer_correct    : does the generated text carry the ground-truth answer

Crossing those gives four buckets:

  1. retrieved + correct answer   working as intended
  2. retrieved + wrong answer     GENERATION failure - the evidence was in the
                                  prompt and the model still got it wrong
  3. not retrieved + wrong answer RETRIEVAL failure - the model never saw it
  4. not retrieved + correct answer answered without the supporting chunk
                                  (parametric knowledge, or the fact appears
                                  elsewhere in the corpus)

Read-only: retrieval, embedding, chunking and generation are all called exactly
as the app calls them. Nothing is tuned, and no fix is attempted.

Run from anywhere:
    python backend/scripts/generation_analysis.py
"""

import json
import os
import re
import sys
import time
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
from backend.pipeline.generator import generate  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
OUT_PATH = EVAL_DIR / "generation_analysis.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]

# Matches app.py's TOP_K — this measures the configuration that actually ships.
TOP_K = 3

BUCKETS = {
    (True, True): "1. retrieved + correct answer",
    (True, False): "2. retrieved + wrong answer (generation failure)",
    (False, False): "3. not retrieved + wrong answer (retrieval failure)",
    (False, True): "4. not retrieved + correct answer",
}


def normalise(text: str) -> str:
    """Lowercase, strip punctuation, and unify number formatting (3,197 -> 3197)."""
    text = str(text).lower()
    text = re.sub(r'(?<=\d),(?=\d)', '', text)          # thousands separators
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)        # "1 550" -> "1550"
    text = re.sub(r'[^a-z0-9.]+', ' ', text)
    text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' ', text)     # keep decimal points only
    return re.sub(r'\s+', ' ', text).strip()


def numbers_in(text: str):
    return set(re.findall(r'\d+(?:\.\d+)?', normalise(text)))


def token_f1(a: str, b: str) -> float:
    ta, tb = normalise(a).split(), normalise(b).split()
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for token in ta:
        if token in pool:
            pool.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(ta), common / len(tb)
    return 2 * precision * recall / (precision + recall)


def judge(expected: str, produced: str) -> dict:
    """
    Decide whether a generated answer carries the ground-truth answer.

    flan-t5 answers tersely and extractively ("3197" for "3,197 sentence
    pairs"), so exact string equality is far too strict. The rule is:
    containment either way, or a high token overlap, or - when the expected
    answer is numeric - every one of its numbers appearing in the output.
    Every signal is stored so a borderline call can be re-judged by hand.
    """
    exp, got = normalise(expected), normalise(produced)
    expected_numbers = numbers_in(expected)

    signals = {
        "exact_match": exp == got,
        "expected_in_answer": bool(exp) and exp in got,
        "answer_in_expected": bool(got) and len(got) >= 2 and got in exp,
        "token_f1": round(token_f1(expected, produced), 3),
        "expected_numbers": sorted(expected_numbers),
        "all_expected_numbers_present": bool(expected_numbers)
        and expected_numbers <= numbers_in(produced),
    }

    correct = (
        signals["exact_match"]
        or signals["expected_in_answer"]
        or signals["answer_in_expected"]
        or signals["token_f1"] >= 0.6
        or signals["all_expected_numbers_present"]
    )
    # Flag the grey zone rather than pretending the rule is unambiguous.
    signals["needs_human_review"] = bool(
        not correct and 0.3 <= signals["token_f1"] < 0.6)
    return correct, signals


def build_index():
    chunks = []
    for name in DOCUMENTS:
        chunks.extend(preprocess(load_file(str(RAW_DIR / name))))
    embeddings = embed(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index, chunks


def main():
    print("=" * 78)
    print(f"GENERATION vs RETRIEVAL FAILURE ANALYSIS  (k={TOP_K}, as shipped)")
    print("=" * 78)

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    print(f"Ground truth: {len(ground_truth)} questions")

    index, chunks = build_index()
    print(f"Combined index: {index.ntotal} chunks across {len(DOCUMENTS)} documents\n")

    results = []
    counts = {label: 0 for label in BUCKETS.values()}

    for i, entry in enumerate(ground_truth, start=1):
        expected = (entry["source_file"], entry["page"])

        started = time.time()
        retrieved = retrieve(entry["question"], index, chunks, k=TOP_K)
        answer = generate(entry["question"], retrieved)
        elapsed = time.time() - started

        retrieved_records = [{
            "rank": r,
            "source_file": c.source_file,
            "page": c.page,
            "is_expected": (c.source_file, c.page) == expected,
            "text": str(c),
        } for r, c in enumerate(retrieved, start=1)]

        retrieved_correct = any(rec["is_expected"] for rec in retrieved_records)
        answer_correct, signals = judge(entry["answer"], answer)
        bucket = BUCKETS[(retrieved_correct, answer_correct)]
        counts[bucket] += 1

        results.append({
            "index": i,
            "question": entry["question"],
            "expected_answer": entry["answer"],
            "expected_source": {"source_file": expected[0], "page": expected[1]},
            "retrieved_correct": retrieved_correct,
            "generated_answer": answer,
            "answer_correct": answer_correct,
            "bucket": bucket,
            "match_signals": signals,
            "seconds": round(elapsed, 2),
            "retrieved_chunks": retrieved_records,
        })

        flag = "R+" if retrieved_correct else "R-"
        flag += "A+" if answer_correct else "A-"
        print(f"  {i:>2}/{len(ground_truth)}  [{flag}] {elapsed:>5.1f}s  "
              f"{expected[0][:26]:<26} p.{expected[1]:<3} "
              f"expected={entry['answer'][:34]!r} got={answer[:34]!r}", flush=True)

    # ---- buckets -------------------------------------------------------------
    print()
    print("=" * 78)
    print("BUCKETS")
    print("=" * 78)
    total = len(results)
    for label in BUCKETS.values():
        n = counts[label]
        print(f"  {n:>2}/{total}  ({n / total:>5.1%})  {label}")

    review = [r for r in results if r["match_signals"]["needs_human_review"]]
    if review:
        print(f"\n  {len(review)} answer(s) sit near the grading threshold and are "
              f"worth eyeballing (flagged in the JSON):")
        for r in review:
            print(f"    Q{r['index']}: expected {r['expected_answer']!r} "
                  f"got {r['generated_answer']!r} (F1 {r['match_signals']['token_f1']})")

    # ---- bucket 2 detail -----------------------------------------------------
    bucket2 = [r for r in results
               if r["bucket"].startswith("2.")]
    print()
    print("=" * 78)
    print(f"BUCKET 2 DETAIL - correct chunk WAS retrieved, answer still wrong "
          f"({len(bucket2)} case(s))")
    print("=" * 78)
    if not bucket2:
        print("  none")
    for r in bucket2:
        print()
        print("-" * 78)
        print(f"Q{r['index']}. {r['question']}")
        print(f"  expected answer : {r['expected_answer']!r}")
        print(f"  generated answer: {r['generated_answer']!r}")
        print(f"  token F1        : {r['match_signals']['token_f1']}")
        print(f"  chunks passed to the generator:")
        for rec in r["retrieved_chunks"]:
            mark = ">> EXPECTED" if rec["is_expected"] else "           "
            print(f"    {mark} #{rec['rank']} {rec['source_file']} p.{rec['page']}")
            body = " ".join(rec["text"].split())
            print(f"        {body}")

    payload = {
        "note": "Read-only analysis. Retrieval, embedding, chunking and "
                "generation are unchanged; no fix attempted.",
        "k": TOP_K,
        "total_questions": total,
        "bucket_counts": counts,
        "grading_rule": "An answer counts as correct if the normalised expected "
                        "answer is contained in the output (or vice versa), or "
                        "token F1 >= 0.6, or every number in the expected answer "
                        "appears in the output. All signals are stored per "
                        "question so calls can be re-judged.",
        "results": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
