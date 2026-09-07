"""
backend/scripts/eval_own_material.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Answers the question a document benchmark cannot: does the system work for the
job it was built for?

Such a benchmark asks a retrieval pipeline to read charts, count marks on a
page and join facts across a fifty-page financial report. It scores 6.2% on the
questions that have an answer, and that is a real result about the generator.
It is not a result about revising from your own lecture slides, which is what
this system is for, and nothing measured so far covers that end to end: the
slide ground truth was used only to score retrieval.

So the same 51 questions over the student's own fifteen lecture decks are run
all the way through, in both styles the application uses:

  short   — the extractive answer, which the quiz compares against
  explain — the paragraph the chat view streams

Scored by generation_analysis.judge, the same rule as every other end-to-end
number here.

Why the automatic score here is not trustworthy, and what to use instead
------------------------------------------------------------------------
Measured: 12/51 short, 15/51 explain. Those numbers should not be quoted.
Thirty-six of the fifty-one reference answers are two words or fewer, because
they were produced by the short style from a single slide: "CGAT", "SLIDING",
"100100", "writing tests". A fragment like that is a fine label for *which
slide* the answer is on, which is what the retrieval evaluation uses it for.
It is a poor reference for whether a paragraph is right, and the judge marks
plainly correct answers wrong on it:

    "How many songs were in the dataset?"  reference "200k",
        answered "approximately 200,000 songs"            marked wrong
    "What are the labels of the nodes?"    reference "A, B, C",
        answered "The nodes are labeled as A, B, and C."  marked wrong
    "What are the Four Ps of creativity?"  reference "Product, Process,
        Press/Environment", answered with all three and Person/Producer as
        well                                             marked wrong

There are real failures in there too — SketchRNN answered as "Dall-e", 1987 as
2006 — but the two cannot be separated by this rule, so the script also writes
a blind rating sheet, own_material_rating_sheet.csv, with the answers in a
random order and no indication of how they were scored. Ten minutes of ticking
settles what the automatic rule cannot.

What this can and cannot show
-----------------------------
The questions were written by this system's own question model from single
slides, so the reference answer is one this system produced with that slide in
hand. Retrieval then has to find that slide again among all fifteen decks, and
the model has to answer from whatever comes back. That makes this a round-trip
consistency measure in the sense of Alberti et al. (2019) — can the pipeline
recover, through retrieval, the answer it would have given with the page in
front of it — and not a measure of absolute correctness. It is biased upward
for exactly that reason and the report must say so.

It is still the closest thing to the real task, on the real material, and it
is the number that says whether the application does its job.

Run from anywhere:
    SA_EMBEDDER=bge-small python backend/scripts/eval_own_material.py
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

from backend.pipeline import generator, sparse as sparse_module  # noqa: E402
from backend.pipeline.embedder import embed, model_key  # noqa: E402
from backend.pipeline.loader import load_pdf  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import DENSE_WEIGHT, retrieve  # noqa: E402
from backend.scripts.generation_analysis import judge  # noqa: E402

DECK_DIR = ROOT / "data" / "projects" / "AI"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "slide_ground_truth.json"
OUT_PATH = EVAL_DIR / "own_material.json"
SHEET_PATH = EVAL_DIR / "own_material_rating_sheet.csv"

TOP_K = 3
STYLES = ["short", "explain"]


def covers(chunk, source_file, page) -> bool:
    if getattr(chunk, "source_file", None) != source_file:
        return False
    first = getattr(chunk, "page", None)
    if first is None:
        return False
    return first <= page <= (getattr(chunk, "page_end", None) or first)


def write_rating_sheet(results):
    """
    The answers in a random order, with no hint of how the rule scored them,
    for a person to mark. Columns: ok = 1 if the answer is right, 0 if not,
    blank if the question itself is unusable.
    """
    import csv
    import random

    rows = []
    for style in STYLES:
        for i, r in enumerate(results[style]["rows"]):
            rows.append({"id": f"{style[0]}{i:02d}", "style": style,
                         "question": r["question"],
                         "answer": " ".join(r["answer"].split()),
                         "reference_from_the_slide": r["expected"],
                         "source_file": r["source_file"], "slide": r["page"],
                         "ok": ""})
    random.Random(0).shuffle(rows)
    with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> {SHEET_PATH.relative_to(ROOT)}  ({len(rows)} answers to rate)")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    questions = truth["questions"]
    print(f"{len(questions)} questions over the student's own decks "
          f"({sum(1 for q in questions if q['source'] == 'picture')} of them "
          f"answerable only from text read out of a picture)")
    print(f"model {generator.MODEL_NAME}   embedder {model_key()}   "
          f"k={TOP_K}   abstain={generator.ABSTAIN}\n")

    paths = sorted(DECK_DIR.glob("*.pdf"))
    chunks = []
    for path in paths:
        chunks.extend(preprocess(load_pdf(str(path), figures="auto")))
    vectors = embed(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    keywords = sparse_module.build_index(chunks)
    print(f"{len(chunks)} chunks across {len(paths)} decks\n")

    passages, found = [], []
    for entry in questions:
        got = retrieve(entry["question"], index, chunks, k=TOP_K,
                       sparse=keywords, dense_weight=DENSE_WEIGHT)
        passages.append([str(c) for c in got])
        found.append(any(covers(c, entry["source_file"], entry["page"])
                         for c in got))

    results = {"retrieval_recall_at_3": round(sum(found) / len(found), 4)}
    print(f"retrieval found the source slide for {sum(found)}/{len(found)} "
          f"({sum(found) / len(found):.0%})\n")

    for style in STYLES:
        was = generator.ANSWER_STYLE
        generator.ANSWER_STYLE = style
        rows, seconds = [], []
        try:
            for entry, context, hit in zip(questions, passages, found):
                t0 = time.time()
                answer = generator.generate(entry["question"], context)
                seconds.append(time.time() - t0)
                correct, signals = judge(entry["answer"], answer)
                rows.append({
                    "question": entry["question"],
                    "expected": entry["answer"],
                    "source": entry["source"],
                    "source_file": entry["source_file"],
                    "page": entry["page"],
                    "retrieved_source_slide": hit,
                    "answer": answer,
                    "correct": bool(correct),
                    "signals": signals,
                })
        finally:
            generator.ANSWER_STYLE = was

        correct = sum(r["correct"] for r in rows)
        with_page = [r for r in rows if r["retrieved_source_slide"]]
        without = [r for r in rows if not r["retrieved_source_slide"]]
        picture = [r for r in rows if r["source"] == "picture"]
        text = [r for r in rows if r["source"] == "text_layer"]
        results[style] = {
            "correct": correct,
            "questions": len(rows),
            "accuracy": round(correct / len(rows), 4),
            "accuracy_when_slide_retrieved": round(
                sum(r["correct"] for r in with_page) / len(with_page), 4) if with_page else None,
            "accuracy_when_slide_missed": round(
                sum(r["correct"] for r in without) / len(without), 4) if without else None,
            "accuracy_text_layer": round(
                sum(r["correct"] for r in text) / len(text), 4) if text else None,
            "accuracy_from_pictures": round(
                sum(r["correct"] for r in picture) / len(picture), 4) if picture else None,
            "median_seconds": round(statistics.median(seconds), 2),
            "median_words": statistics.median(len(r["answer"].split()) for r in rows),
            "rows": rows,
        }
        r = results[style]
        print(f"  {style:<8} {correct:>2}/{len(rows)}  ({r['accuracy']:.0%})   "
              f"slide retrieved {r['accuracy_when_slide_retrieved']:.0%} / "
              f"missed {r['accuracy_when_slide_missed']:.0%}   "
              f"text {r['accuracy_text_layer']:.0%} / "
              f"picture {r['accuracy_from_pictures']:.0%}   "
              f"{r['median_seconds']:.1f}s, {r['median_words']:.0f} words")

    write_rating_sheet(results)

    OUT_PATH.write_text(json.dumps({
        "note": "The system end to end on the student's own lecture decks, "
                "the task it was built for. Questions come from "
                "slide_ground_truth.json, written by this system's own "
                "question model from single slides, so the reference answer "
                "is one this system gave with the slide in hand: this is a "
                "round-trip consistency measure, biased upward, not a measure "
                "of absolute correctness. Scored by the same judge rule as "
                "every other end-to-end number here.",
        "model": generator.MODEL_NAME,
        "embedder": model_key(),
        "abstain": generator.ABSTAIN,
        "k": TOP_K,
        "decks": len(paths),
        "chunks": len(chunks),
        "questions": len(questions),
        **results,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
