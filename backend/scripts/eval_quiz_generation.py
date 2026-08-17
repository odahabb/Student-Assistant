"""
backend/scripts/eval_quiz_generation.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Evaluates quiz question generation (quiz.py) on the project's documents: the
four papers used for the retrieval evaluation plus the six-page sample
lecture notes, indexed together as one subject.

Two modes:

  generate (default)
      For every topic, up to ATTEMPTS_PER_TOPIC chunks are turned into
      question/answer pairs. Every attempt is logged with why it was rejected,
      and well-formed pairs are round-trip checked separately so that pairs
      the filter rejects are kept for comparison. Also records whether
      retrieval, given the generated question, returns the source page.
      Writes data/eval/quiz_generation.json and a blind rating sheet,
      data/eval/quiz_rating_sheet.csv, which does not show the filter result.

  score
      After the rating sheet has been filled in by a person (1 = yes,
      0 = no in each rating column), compares the ratings of pairs the
      round-trip filter kept with those it rejected.
          python backend/scripts/eval_quiz_generation.py score

Run from anywhere:
    python backend/scripts/eval_quiz_generation.py
"""

import csv
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

EVAL_DIR = ROOT / "data" / "eval"
OUT_PATH = EVAL_DIR / "quiz_generation.json"
SHEET_PATH = EVAL_DIR / "quiz_rating_sheet.csv"
SCORES_PATH = EVAL_DIR / "quiz_rating_results.json"

DOCUMENTS = [
    ROOT / "data" / "raw" / "embedding.pdf",
    ROOT / "data" / "raw" / "Whisper.pdf",
    ROOT / "data" / "raw" / "Flant5pdf.pdf",
    ROOT / "data" / "raw" / "Hallucinations_in_Large_Language_Models_LLMs.pdf",
    ROOT / "data" / "Prototype" / "sample_lecture_notes.pdf",
]
ATTEMPTS_PER_TOPIC = 3
CHUNKING = "sentence"   # as app.py
TOP_K = 3
SEED = 7
RATINGS = {
    "clear": "The question is grammatical and unambiguous",
    "answerable": "The passage contains the answer to the question",
    "answer_correct": "The reference answer is correct for the question",
    "useful": "The question tests something worth revising (not trivia)",
}


def generate():
    import faiss
    import numpy as np

    from backend.pipeline import quiz
    from backend.pipeline.embedder import embed
    from backend.pipeline.generator import complete, generate as answer
    from backend.pipeline.loader import load_file
    from backend.pipeline.preprocessor import preprocess
    from backend.pipeline.retriever import retrieve

    chunks = []
    for path in DOCUMENTS:
        chunks.extend(preprocess(load_file(str(path)), chunking=CHUNKING))
    embeddings = embed(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    topics = quiz.build_topics(chunks)
    usable = sum(len(t.chunk_indices) for t in topics)
    print(f"{len(chunks)} chunks, {usable} usable for quizzes, {len(topics)} topics")

    rng = random.Random(SEED)
    attempts = []
    started = time.time()
    for topic in topics:
        order = list(topic.chunk_indices)
        rng.shuffle(order)
        for i in order[:ATTEMPTS_PER_TOPIC]:
            chunk = chunks[i]
            t0 = time.time()
            item, reason = quiz.generate_item(
                chunk, answer_fn=answer,
                question_fn=lambda p: complete(p, max_new_tokens=48))
            record = {"topic_id": topic.id, "chunk_index": i,
                      "source_file": chunk.source_file, "page": chunk.page,
                      "rejected_because": reason}
            if item is not None:
                retrieved = retrieve(item.question, index, chunks, k=TOP_K)
                passed = quiz.roundtrip_check(item, answer, lambda q: retrieved)
                record.update({
                    "question": item.question,
                    "answer": item.answer,
                    "passage": item.passage,
                    "section": item.section,
                    "roundtrip_answer": item.roundtrip_answer,
                    "roundtrip_score": item.roundtrip_score,
                    "passed_roundtrip": passed,
                    "source_page_retrieved": any(
                        (c.source_file, c.page) == (chunk.source_file, chunk.page)
                        for c in retrieved),
                })
                if not passed:
                    record["rejected_because"] = "failed round-trip check"
            record["seconds"] = round(time.time() - t0, 2)
            attempts.append(record)
            print(f"  {record['rejected_because'] or 'kept':<36} "
                  f"{record.get('question', '')[:70]}")
    elapsed = time.time() - started

    formed = [a for a in attempts if "question" in a]
    kept = [a for a in formed if a["passed_roundtrip"]]
    per_doc = {}
    for a in attempts:
        d = per_doc.setdefault(a["source_file"], Counter())
        d["attempts"] += 1
        d["kept"] += a["rejected_because"] is None
    summary = {
        "chunks": len(chunks),
        "usable_chunks": usable,
        "topics": len(topics),
        "attempts": len(attempts),
        "outcomes": dict(Counter(a["rejected_because"] or "kept" for a in attempts)),
        "well_formed_rate": round(len(formed) / len(attempts), 3),
        "roundtrip_pass_rate_of_well_formed": round(len(kept) / len(formed), 3),
        "kept_rate": round(len(kept) / len(attempts), 3),
        "topics_with_at_least_one_item": len({a["topic_id"] for a in kept}),
        "source_page_retrieved": {
            "kept": round(mean(a["source_page_retrieved"] for a in kept), 3),
            "rejected": round(mean(a["source_page_retrieved"] for a in formed
                                   if not a["passed_roundtrip"]), 3),
        },
        "seconds_per_attempt": round(elapsed / len(attempts), 2),
        "per_document": {k: dict(v) for k, v in per_doc.items()},
    }
    print(json.dumps(summary, indent=2))

    OUT_PATH.write_text(json.dumps({
        "note": "Quiz generation over the evaluation papers and sample lecture "
                "notes. Round-trip check applied separately so rejected pairs "
                "are kept for comparison.",
        "settings": {"attempts_per_topic": ATTEMPTS_PER_TOPIC, "top_k": TOP_K,
                     "chunking": CHUNKING,
                     "seed": SEED, "grade_threshold": quiz.GRADE_THRESHOLD,
                     "device": os.environ.get("SA_DEVICE", "gpu")},
        "summary": summary,
        "attempts": attempts,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    sheet = [dict(a, id=n) for n, a in enumerate(formed, start=1)]
    rng.shuffle(sheet)
    with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "document", "section", "passage", "question",
                         "reference_answer", *RATINGS, "notes"])
        for a in sheet:
            writer.writerow([a["id"], a["source_file"], a["section"], a["passage"],
                             a["question"], a["answer"], *[""] * len(RATINGS), ""])
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}\n  -> {SHEET_PATH.relative_to(ROOT)} "
          f"({len(sheet)} pairs to rate: {', '.join(RATINGS)})")


def score():
    data = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    formed = [a for a in data["attempts"] if "question" in a]
    by_id = {n: a for n, a in enumerate(formed, start=1)}

    rated = []
    with open(SHEET_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            values = {k: row[k].strip() for k in RATINGS}
            if all(v in ("0", "1") for v in values.values()):
                a = by_id[int(row["id"])]
                rated.append({"kept": a["passed_roundtrip"],
                              **{k: int(v) for k, v in values.items()}})
    if not rated:
        raise SystemExit("No fully rated rows yet — fill in 0/1 for every rating column.")

    groups = {"kept by filter": [r for r in rated if r["kept"]],
              "rejected by filter": [r for r in rated if not r["kept"]],
              "all": rated}
    results = {}
    for name, rows in groups.items():
        if rows:
            results[name] = {"n": len(rows), **{
                k: round(mean(r[k] for r in rows), 3) for k in RATINGS},
                "all_four": round(mean(all(r[k] for k in RATINGS) for r in rows), 3)}
    print(json.dumps(results, indent=2))
    SCORES_PATH.write_text(json.dumps({
        "note": "Human ratings of generated quiz items (blind to filter result).",
        "rating_definitions": RATINGS,
        "results": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  -> {SCORES_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    score() if sys.argv[1:] == ["score"] else generate()
