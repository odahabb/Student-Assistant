"""
backend/scripts/eval_grader.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Calibrates and checks quiz.grade(), the function that marks a student's
free-text answer against a reference answer.

Labelled pairs come from data/eval/generation_analysis.json, which stores 25
generated answers with a correct/incorrect label against the ground-truth
answer:
  - the 25 stored pairs (12 labelled correct, 13 incorrect), with the manual
    review in generation_manual_review.json applied — it changes no label;
  - 25 mismatched pairs: each generated answer scored against the reference of
    a different question, all incorrect by construction.

Caveats:
  - the stored labels were produced by a rule (containment, token F1 >= 0.6,
    or all numbers present), so they share token F1 with the grader;
  - every correct pair here is caught by containment or token F1, so the set
    has no true paraphrases and cannot show how high the threshold may go;
  - the number-mismatch rule in quiz.grade() was added after this set showed
    three numeric false positives, so results on it are not an independent
    test of that rule.
data/eval/grader_decisions.csv lists every decision for human checking.

Run from anywhere:
    python backend/scripts/eval_grader.py
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from backend.pipeline import quiz  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
SOURCE = EVAL_DIR / "generation_analysis.json"
OUT_PATH = EVAL_DIR / "grader_calibration.json"
CSV_PATH = EVAL_DIR / "grader_decisions.csv"
THRESHOLDS = [round(0.40 + 0.05 * i, 2) for i in range(12)]   # 0.40 .. 0.95


def build_pairs():
    rows = json.loads(SOURCE.read_text(encoding="utf-8"))["results"]
    pairs = [{"kind": "stored", "question": r["question"],
              "answer": r["generated_answer"], "reference": r["expected_answer"],
              "label": bool(r["answer_correct"])} for r in rows]
    n = len(rows)
    for i, r in enumerate(rows):
        other = rows[(i + 1) % n]
        pairs.append({"kind": "mismatched", "question": other["question"],
                      "answer": r["generated_answer"],
                      "reference": other["expected_answer"], "label": False})
    return pairs


def confusion(pairs, threshold):
    tp = fp = tn = fn = 0
    for p in pairs:
        predicted = p["method"] == "containment" or p["score"] >= threshold
        if predicted and p["label"]:
            tp += 1
        elif predicted:
            fp += 1
        elif p["label"]:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round((tp + tn) / len(pairs), 3),
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3)}


def main():
    pairs = build_pairs()
    for p in pairs:
        # threshold 2.0 = never pass on score, so .score is the raw evidence
        g = quiz.grade(p["answer"], p["reference"], threshold=2.0)
        p["score"], p["method"] = g.score, g.method

    sweep = [confusion(pairs, t) for t in THRESHOLDS]
    chosen = confusion(pairs, quiz.GRADE_THRESHOLD)
    stored_only = confusion([p for p in pairs if p["kind"] == "stored"],
                            quiz.GRADE_THRESHOLD)

    print(f"{len(pairs)} pairs, {sum(p['label'] for p in pairs)} labelled correct\n")
    print(f"{'thr':>5} {'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6}   tp fp tn fn")
    for s in sweep:
        mark = "  <- GRADE_THRESHOLD" if s["threshold"] == quiz.GRADE_THRESHOLD else ""
        print(f"{s['threshold']:>5} {s['accuracy']:>6} {s['precision']:>6} "
              f"{s['recall']:>6} {s['f1']:>6}   {s['tp']:>2} {s['fp']:>2} "
              f"{s['tn']:>2} {s['fn']:>2}{mark}")
    print("\nstored pairs only:", stored_only)

    disagreements = [p for p in pairs
                     if (p["method"] == "containment"
                         or p["score"] >= quiz.GRADE_THRESHOLD) != p["label"]]
    print(f"\n{len(disagreements)} disagreements at the chosen threshold:")
    for p in disagreements:
        print(f"  [{p['kind']}] label={p['label']} score={p['score']} ({p['method']})"
              f"\n     answer   : {p['answer']}\n     reference: {p['reference']}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "kind", "question", "answer", "reference", "label", "score", "method",
            "grader_correct", "human_correct"])
        writer.writeheader()
        for p in pairs:
            writer.writerow({**p, "grader_correct":
                             p["method"] == "containment" or p["score"] >= quiz.GRADE_THRESHOLD,
                             "human_correct": ""})

    OUT_PATH.write_text(json.dumps({
        "note": "Grader calibration on rule-labelled pairs; see script docstring "
                "for the caveat. Scores use MiniLM cosine and token F1.",
        "pairs": len(pairs),
        "labelled_correct": sum(p["label"] for p in pairs),
        "grade_threshold": quiz.GRADE_THRESHOLD,
        "at_threshold": chosen,
        "at_threshold_stored_pairs_only": stored_only,
        "sweep": sweep,
        "disagreements": disagreements,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}\n  -> {CSV_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
