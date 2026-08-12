"""
backend/scripts/dataset_and_metrics.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Describes both evaluation datasets and re-scores the saved image-extraction
results with the full metric set, including ANLS — the official DocVQA metric
(Mathew et al., 2021) — so the results can be set against published numbers.

Nothing is re-run: scores come from the answers already stored in
data/eval/easyocr_blip_vs_qwen2vl_results.csv.

Metrics (each takes the best score over a question's accepted answers):
  containment : normalised answer appears inside the generated answer
                (what the eval notebook calls "exact_match")
  strict_em   : normalised generated answer equals a normalised accepted answer
  token_f1    : SQuAD-style token overlap F1
  anls        : 1 - normalised Levenshtein distance, zeroed when the distance
                is >= 0.5 (tau = 0.5, as in the DocVQA benchmark)

DocVQA metadata (question types, document ids, split size) is read from the
locally cached lmms-lab/DocVQA parquet files when they are present; the script
still runs without them and just omits those fields.

Run from anywhere:
    python backend/scripts/dataset_and_metrics.py
"""

import ast
import csv
import glob
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "data" / "eval"
DOCVQA_DIR = ROOT / "data" / "raw" / "docvqa_eval25"
RESULTS_CSV = EVAL_DIR / "easyocr_blip_vs_qwen2vl_results.csv"
COMPETITORS = EVAL_DIR / "competitor_analysis.json"
RECALL = EVAL_DIR / "recall_results.json"
GROUND_TRUTH = EVAL_DIR / "retrieval_ground_truth.json"
OUT_PATH = EVAL_DIR / "dataset_and_metrics.json"

HF_PARQUET_GLOB = str(Path.home() / ".cache" / "huggingface" / "hub"
                      / "datasets--lmms-lab--DocVQA" / "snapshots" / "*"
                      / "DocVQA" / "validation-*.parquet")

METHODS = {"easyocr_blip": "ocr_blip", "qwen2vl": "qwen2vl"}
ANLS_TAU = 0.5


# ---- metrics ------------------------------------------------------------------

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def containment(pred: str, gt: str) -> float:
    return float(normalize(gt) in normalize(pred))


def strict_em(pred: str, gt: str) -> float:
    return float(normalize(gt) == normalize(pred))


def token_f1(pred: str, gt: str) -> float:
    p, g = normalize(pred).split(), normalize(gt).split()
    if not p or not g:
        return 0.0
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls(pred: str, gt: str) -> float:
    # DocVQA's official scorer compares lowercased, whitespace-trimmed strings.
    p, g = " ".join(str(pred).lower().split()), " ".join(str(gt).lower().split())
    if not p and not g:
        return 1.0
    nl = levenshtein(p, g) / max(len(p), len(g))
    return 1.0 - nl if nl < ANLS_TAU else 0.0


METRICS = {"containment": containment, "strict_em": strict_em,
           "token_f1": token_f1, "anls": anls}


def best(pred, answers, fn):
    return max(fn(pred, a) for a in answers)


# ---- DocVQA -------------------------------------------------------------------

def load_docvqa_metadata():
    files = sorted(glob.glob(HF_PARQUET_GLOB))
    if not files:
        return None
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["questionId", "question_types", "docId", "answers"]
    table = pa.concat_tables([pq.read_table(f, columns=cols) for f in files])
    return table.to_pylist()


def docvqa_section():
    eval_rows = list(csv.DictReader(open(DOCVQA_DIR / "eval_set.csv", encoding="utf-8")))
    results = list(csv.DictReader(open(RESULTS_CSV, encoding="utf-8")))
    assert len(eval_rows) == len(results)

    answers = [json.loads(r["answers"]) for r in eval_rows]
    section = {
        "source": "lmms-lab/DocVQA, config 'DocVQA', validation split",
        "eval_subset_size": len(eval_rows),
        "accepted_answers_per_question": {
            "mean": round(mean(len(a) for a in answers), 2),
            "max": max(len(a) for a in answers)},
        "answer_length_words": {
            "mean": round(mean(len(a[0].split()) for a in answers), 2),
            "median": median(len(a[0].split()) for a in answers),
            "max": max(len(a[0].split()) for a in answers)},
        "numeric_answers": sum(bool(re.search(r"\d", a[0])) for a in answers),
    }

    try:
        from PIL import Image
        sizes = [Image.open(DOCVQA_DIR / "images" / r["image_filename"]).size
                 for r in eval_rows]
        section["image_size_px"] = {
            "width_range": [min(w for w, _ in sizes), max(w for w, _ in sizes)],
            "height_range": [min(h for _, h in sizes), max(h for _, h in sizes)]}
    except Exception as e:                                    # noqa: BLE001
        section["image_size_px"] = f"unavailable ({e})"

    meta = load_docvqa_metadata()
    types_by_qid = {}
    if meta:
        ids = [str(m["questionId"]) for m in meta]
        eval_ids = [r["question_id"] for r in eval_rows]
        by_id = {str(m["questionId"]): m for m in meta}
        types_by_qid = {q: by_id[q]["question_types"] for q in eval_ids}
        section.update({
            "validation_split_questions": len(meta),
            "validation_split_documents": len({m["docId"] for m in meta}),
            "eval_subset_is_first_n_rows": ids[:len(eval_ids)] == eval_ids,
            "eval_subset_documents": len({by_id[q]["docId"] for q in eval_ids}),
            "question_types_eval_subset": dict(Counter(
                t for q in eval_ids for t in by_id[q]["question_types"])),
            "question_types_validation_split": dict(Counter(
                t for m in meta for t in m["question_types"])),
        })

    # ---- re-score ----
    per_question, scores = [], {m: {k: [] for k in METRICS} for m in METHODS}
    for row, acc in zip(results, answers):
        item = {"question_id": row["question_id"], "question": row["question"],
                "question_types": types_by_qid.get(row["question_id"])}
        for method, prefix in METHODS.items():
            pred = row[f"{prefix}_final_answer"] or ""
            item[method] = {"answer": pred}
            for name, fn in METRICS.items():
                v = best(pred, acc, fn)
                scores[method][name].append(v)
                item[method][name] = round(v, 4)
        per_question.append(item)

    summary = {m: {k: round(mean(v), 3) for k, v in s.items()} for m, s in scores.items()}
    for method, prefix in METHODS.items():
        ext = [float(r[f"{prefix}_extraction_time_sec"]) for r in results]
        pipe = [float(r[f"{prefix}_pipeline_time_sec"]) for r in results]
        summary[method].update({
            "extraction_sec_mean": round(mean(ext), 2),
            "extraction_sec_median": round(median(ext), 2),
            "pipeline_sec_mean": round(mean(pipe), 2),
            "empty_answers": sum(not (r[f"{prefix}_final_answer"] or "").strip()
                                 for r in results),
        })

    a, b = scores["qwen2vl"]["anls"], scores["easyocr_blip"]["anls"]
    head_to_head = {"qwen_better": sum(x > y for x, y in zip(a, b)),
                    "blip_better": sum(x < y for x, y in zip(a, b)),
                    "tied": sum(x == y for x, y in zip(a, b))}

    by_type = {}
    if types_by_qid:
        for t in sorted({t for ts in types_by_qid.values() for t in ts}):
            idx = [i for i, q in enumerate(per_question) if t in (q["question_types"] or [])]
            by_type[t] = {"n": len(idx), **{
                m: round(mean(scores[m]["anls"][i] for i in idx), 3) for m in METHODS}}

    return section, summary, head_to_head, by_type, per_question


# ---- retrieval corpus ---------------------------------------------------------

def retrieval_section():
    recall = json.loads(RECALL.read_text(encoding="utf-8"))
    gt = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    section = {
        "documents": [{k: d[k] for k in ("source_file", "pages_with_text", "chunks")}
                      for d in recall["documents"]],
        "total_pages": sum(d["pages_with_text"] for d in recall["documents"]),
        "total_chunks": recall["total_chunks"],
        "questions": len(gt),
        "questions_per_document": dict(Counter(e["source_file"] for e in gt)),
        "distinct_expected_pages": len({(e["source_file"], e["page"]) for e in gt}),
        "numeric_answers": sum(bool(re.search(r"\d", e["answer"])) for e in gt),
        "chunking": {"window_tokens": 220, "overlap_tokens": 40,
                     "page_bounded": True, "page1_boilerplate_stripped": True},
    }
    if COMPETITORS.exists():
        comp = json.loads(COMPETITORS.read_text(encoding="utf-8"))
        section["answer_placement"] = comp["answer_placement_counts"]
    return section


def main():
    docvqa, summary, h2h, by_type, per_q = docvqa_section()
    retrieval = retrieval_section()

    print("DocVQA subset:", json.dumps(docvqa, indent=1))
    print("\nScores (mean over 25):")
    for m, s in summary.items():
        print(f"  {m:<13} {s}")
    print("\nANLS head-to-head:", h2h)
    print("ANLS by question type:", json.dumps(by_type, indent=1))
    print("\nRetrieval corpus:", json.dumps(retrieval, indent=1))

    payload = {
        "note": "Dataset description and re-scoring of stored answers. Nothing re-run.",
        "metric_definitions": {
            "containment": "normalised accepted answer is a substring of the output "
                           "(reported as 'exact_match' by the eval notebook)",
            "strict_em": "normalised output equals a normalised accepted answer",
            "token_f1": "token-overlap F1 between output and accepted answer",
            "anls": f"1 - normalised Levenshtein distance, 0 if distance >= {ANLS_TAU}",
            "aggregation": "max over accepted answers, then mean over questions",
        },
        "docvqa": docvqa,
        "image_extraction_scores": summary,
        "anls_head_to_head": h2h,
        "anls_by_question_type": by_type,
        "retrieval_corpus": retrieval,
        "per_question": per_q,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
