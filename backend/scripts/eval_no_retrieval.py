"""
backend/scripts/eval_no_retrieval.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Asks whether retrieval is doing the work.

Every other script here measures the system with retrieval switched on, which
cannot separate two explanations of a correct answer: the passages carried it,
or the model already knew it. Qwen2.5 was trained on the public internet, and
two of the four evaluation papers (Whisper and FLAN) are well known, so this
is not a hypothetical worry.

The same 25 questions are answered three ways, in one process, with the same
model and the same grading rule (generation_analysis.judge, imported rather
than copied so the two sets of numbers stay comparable):

  no_context    — closed book: the question alone, no passages at all.
  retrieved     — the shipped configuration: sentence chunks, bge-small,
                  hybrid retrieval, the top 3 passages.
  wrong_context — the top 3 passages for a DIFFERENT question. A control for
                  the middle case, where any passage at all steadies the
                  model: if this scores like no_context, the gain comes from
                  the right passages rather than from having something to
                  read; if it scores like retrieved, the passages are not
                  what the answer rests on.

The short answer style is used throughout, because that is what every recorded
measurement describes.

Differences on 25 questions are small, so retrieved is compared with the other
two by McNemar's exact test, which looks only at the questions whose outcome
changed.

Run from anywhere:
    SA_EMBEDDER=bge-small python backend/scripts/eval_no_retrieval.py
"""

import json
import os
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from backend.pipeline import sparse as sparse_module  # noqa: E402
from backend.pipeline.embedder import embed, model_key  # noqa: E402
from backend.pipeline.generator import MODEL_NAME, answer_short  # noqa: E402
from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import DENSE_WEIGHT, retrieve  # noqa: E402
from backend.scripts.generation_analysis import judge  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "retrieval_ground_truth.json"
OUT_PATH = EVAL_DIR / "no_retrieval.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]
CHUNKING = "sentence"
TOP_K = 3
# Which other question's passages the wrong_context arm uses. Coprime with 25,
# so every question is paired with a different one.
OFFSET = 7


def mcnemar(a_correct, b_correct) -> dict:
    """
    McNemar's exact test on paired correct/incorrect outcomes. Only the
    questions where the two arms disagree carry information; under the null
    each is equally likely to fall either way.
    """
    only_a = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    only_b = sum(1 for a, b in zip(a_correct, b_correct) if b and not a)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    p = min(1.0, 2 * tail / (2 ** n))
    # Three significant figures, not three decimals: a decisive result here is
    # of the order of 1e-5, and rounding it to 0.0 would state something
    # stronger than the test supports.
    return {"only_a": only_a, "only_b": only_b, "p_value": float(f"{p:.3g}")}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    print(f"{len(ground_truth)} questions   model: {MODEL_NAME}   "
          f"embedder: {model_key()}\n")

    chunks = []
    for name in DOCUMENTS:
        chunks.extend(preprocess(load_file(str(RAW_DIR / name)), chunking=CHUNKING))
    vectors = embed(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    keywords = sparse_module.build_index(chunks)
    print(f"{len(chunks)} chunks across {len(DOCUMENTS)} documents\n")

    passages = [retrieve(entry["question"], index, chunks, k=TOP_K,
                         sparse=keywords, dense_weight=DENSE_WEIGHT)
                for entry in ground_truth]

    results = []
    print(f"{'#':>3}  {'closed':>6} {'top 3':>6} {'wrong':>6}   question")
    for i, entry in enumerate(ground_truth):
        other = passages[(i + OFFSET) % len(ground_truth)]
        arms = {
            "no_context": [],
            "retrieved": [str(c) for c in passages[i]],
            "wrong_context": [str(c) for c in other],
        }
        record = {"index": i + 1, "question": entry["question"],
                  "expected_answer": entry["answer"],
                  "expected_source": {"file": entry["source_file"],
                                      "page": entry["page"]},
                  # Did the control accidentally hand over the right page?
                  "wrong_context_holds_the_source": any(
                      getattr(c, "source_file", None) == entry["source_file"]
                      and getattr(c, "page", None) == entry["page"] for c in other),
                  "arms": {}}
        for arm, context in arms.items():
            answer = answer_short(entry["question"], context)
            correct, signals = judge(entry["answer"], answer)
            record["arms"][arm] = {"answer": answer, "correct": bool(correct),
                                   "signals": signals}
        results.append(record)
        marks = {True: "  yes ", False: "   no "}
        print(f"{i + 1:>3}  {marks[record['arms']['no_context']['correct']]:>6}"
              f"{marks[record['arms']['retrieved']['correct']]:>6}"
              f"{marks[record['arms']['wrong_context']['correct']]:>6}   "
              f"{entry['question'][:58]}")

    arms = ["no_context", "retrieved", "wrong_context"]
    outcomes = {a: [r["arms"][a]["correct"] for r in results] for a in arms}
    totals = {a: sum(outcomes[a]) for a in arms}
    leaked = sum(1 for r in results if r["wrong_context_holds_the_source"])

    print()
    for arm in arms:
        print(f"  {arm:<14} {totals[arm]:>2}/{len(results)}  "
              f"({totals[arm] / len(results):.0%})")
    print(f"\n  the control handed over the right page anyway: {leaked} question(s)")

    tests = {
        "retrieved_vs_no_context": mcnemar(outcomes["retrieved"],
                                           outcomes["no_context"]),
        "retrieved_vs_wrong_context": mcnemar(outcomes["retrieved"],
                                              outcomes["wrong_context"]),
        "wrong_context_vs_no_context": mcnemar(outcomes["wrong_context"],
                                               outcomes["no_context"]),
    }
    for name, t in tests.items():
        print(f"  {name:<30} gained {t['only_a']}, lost {t['only_b']}, "
              f"p = {t['p_value']:.3g}")

    OUT_PATH.write_text(json.dumps({
        "note": "Does retrieval do the work? The same questions answered with "
                "no passages, with the retrieved passages, and with another "
                "question's passages. Same model, same grading rule as "
                "generation_analysis.py. only_a / only_b in the tests are the "
                "questions the first arm gets right and the second does not, "
                "and the other way round; p is McNemar's exact two-sided test.",
        "model": MODEL_NAME,
        "embedder": model_key(),
        "chunking": CHUNKING,
        "k": TOP_K,
        "dense_weight": DENSE_WEIGHT,
        "answer_style": "short",
        "wrong_context_offset": OFFSET,
        "correct": totals,
        "control_leaked_source_page": leaked,
        "tests": tests,
        "results": results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
