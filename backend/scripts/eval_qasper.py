"""
backend/scripts/eval_qasper.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Runs the system on QASPER (Dasigi et al., 2021), the benchmark that matches
what this architecture was built to do.

The benchmark measured before this one was the wrong instrument. It asked for
the colour of a node in a diagram, the sum of a column, whether one bar is
taller than another — questions for a vision-language model reading page
images. This is a text retrieval pipeline,
and a fifth of what it lost there was never in the page text at all.

QASPER asks what a reader of a research paper asks. Its 5,049 questions were
written by people who had seen only a paper's title and abstract, then
answered by experts who marked the paragraphs the answer rests on. The answers
are short: an extracted span, a yes or a no, a sentence, or "unanswerable".
That is this system's shape exactly, and it gives the two things needed to
separate the stages:

  Answer F1    token overlap with the reference answer, best over annotators
  Evidence F1  did the system put the right PARAGRAPHS in front of the model

Evidence F1 is the same question as Recall@k, asked in the benchmark's own
terms, so retrieval and answering are scored apart without any extra
machinery.

How a paper is indexed
----------------------
Each paragraph of the full text becomes one unit, carrying its section name,
and goes through the ordinary prose path — the same chunker, embedder and
hybrid retriever the application uses. Because a prose chunk never crosses a
unit boundary, every retrieved chunk maps back to exactly one paragraph, which
is what the evidence metric expects.

Scoring is the benchmark's own evaluator, vendored unchanged. Paragraphs
marked "FLOAT SELECTED" are the captions of figures and tables; they are
dropped from both the gold evidence and ours, which is what the official
--text_evidence_only setting does and the honest setting for a system that
reads no figures.

Run from anywhere:
    SA_EMBEDDER=bge-small python backend/scripts/eval_qasper.py
    ... --papers 0        every paper in the dev set (281, 1005 questions)
    ... --k 5             retrieve more paragraphs
    SA_BACKEND=ollama ... put a larger model through the same pipeline
"""

import json
import os
import random
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
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import DENSE_WEIGHT, retrieve  # noqa: E402
from backend.scripts.vendor.qasper_evaluator import (  # noqa: E402
    evaluate, get_answers_and_evidence)

DATA_DIR = ROOT / "data" / "raw" / "qasper"
EVAL_DIR = ROOT / "data" / "eval"

SEED = 13
CHUNKING = "sentence"
FLOAT_MARKER = "FLOAT SELECTED"
UNANSWERABLE = "Unanswerable"
# What the short style says when it declines, in any of its spellings.
DECLINED = {"unanswerable", "not answerable", "no answer", "n/a", "none", ""}


def arg(flag, default):
    return type(default)(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


PAPERS = arg("--papers", 30)          # 0 means the whole dev set
TOP_K = arg("--k", 3)
SPLIT = arg("--split", "dev")
MODEL_TAG = ("" if generator.MODEL_NAME == "Qwen/Qwen2.5-1.5B-Instruct"
             else "_" + generator.MODEL_NAME.split("/")[-1].lower()
             .replace("-instruct", "").replace(":", "-"))
OUT_PATH = EVAL_DIR / ("qasper"
                       + (f"_{PAPERS}papers" if PAPERS else "_dev")
                       + MODEL_TAG
                       + ("" if TOP_K == 3 else f"_k{TOP_K}")
                       + ("" if generator.ABSTAIN == "check"
                          else f"_abstain-{generator.ABSTAIN}")
                       + ".json")


def paragraphs_of(paper):
    """
    Every paragraph of the full text as one indexed unit, with its section.
    Figure and table captions are kept out: this system does not read them,
    and the official text-evidence-only setting excludes them from the gold.
    """
    units = []
    for section in paper["full_text"]:
        name = (section.get("section_name") or "").strip() or None
        for text in section["paragraphs"]:
            body = " ".join(str(text).split())
            if not body or FLOAT_MARKER in body:
                continue
            units.append({"source_file": "paper", "page": len(units) + 1,
                          "section": name, "text": body})
    return units


def build(units):
    chunks = preprocess(units, chunking=CHUNKING)
    if not chunks:
        return None
    vectors = embed(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return chunks, index, sparse_module.build_index(chunks)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    path = DATA_DIR / f"qasper-{SPLIT}-v0.3.json"
    if not path.exists():
        raise SystemExit(f"{path} is missing — extract the QASPER tarballs first")
    data = json.loads(path.read_text(encoding="utf-8"))

    ids = sorted(data)
    if PAPERS:
        rng = random.Random(SEED)
        rng.shuffle(ids)
        ids = ids[:PAPERS]
    chosen = {i: data[i] for i in ids}
    questions = sum(len(p["qas"]) for p in chosen.values())
    print(f"QASPER {SPLIT}: {len(chosen)} of {len(data)} papers, "
          f"{questions} of {sum(len(p['qas']) for p in data.values())} questions")
    print(f"model {generator.MODEL_NAME}   embedder {model_key()}   k={TOP_K}   "
          f"abstain={generator.ABSTAIN}\n")

    predictions, records = {}, []
    started = time.time()
    for n, (paper_id, paper) in enumerate(chosen.items(), start=1):
        units = paragraphs_of(paper)
        built = build(units)
        if built is None:
            print(f"  [{n}/{len(chosen)}] {paper_id}: no text")
            continue
        chunks, index, keywords = built
        paragraph_of = {u["page"]: u["text"] for u in units}

        for qa in paper["qas"]:
            got = retrieve(qa["question"], index, chunks, k=TOP_K,
                           sparse=keywords, dense_weight=DENSE_WEIGHT)
            evidence, seen = [], set()
            for chunk in got:
                text = paragraph_of.get(getattr(chunk, "page", None))
                if text and text not in seen:
                    seen.add(text)
                    evidence.append(text)
            answer = generator.answer_short(qa["question"],
                                            [str(c) for c in got])
            tidy = " ".join(str(answer).split()).strip().strip(".")
            if tidy.lower() in DECLINED:
                tidy, evidence = UNANSWERABLE, []
            predictions[qa["question_id"]] = {"answer": tidy,
                                              "evidence": evidence}
            records.append({"paper_id": paper_id, "question": qa["question"],
                            "predicted": tidy,
                            "evidence_paragraphs": len(evidence)})
        print(f"  [{n}/{len(chosen)}] {paper['title'][:52]:<52} "
              f"{len(units):>4} paras {len(chunks):>4} chunks "
              f"{len(paper['qas']):>2} q  ({time.time() - started:.0f}s)")

    gold = get_answers_and_evidence(chosen, True)   # text evidence only
    scores = evaluate(gold, predictions)

    declined = sum(1 for p in predictions.values() if p["answer"] == UNANSWERABLE)
    gold_unanswerable = sum(
        1 for refs in gold.values()
        if any(r["answer"] == UNANSWERABLE for r in refs))

    print(f"\n  Answer F1    {scores['Answer F1']:.4f}")
    print(f"  Evidence F1  {scores['Evidence F1']:.4f}")
    print("  Answer F1 by type")
    for kind, value in scores["Answer F1 by type"].items():
        print(f"    {kind:<12} {value:.4f}")
    print(f"\n  the system declined {declined} of {len(predictions)} "
          f"({declined / len(predictions):.0%}); "
          f"{gold_unanswerable} are unanswerable in the gold")

    OUT_PATH.write_text(json.dumps({
        "note": "The system on QASPER (Dasigi et al., 2021), scored with the "
                "benchmark's own evaluator, vendored unchanged. Each paragraph "
                "of a paper is one unit through the ordinary prose path, so a "
                "retrieved chunk maps to exactly one paragraph and Evidence F1 "
                "is Recall@k in the benchmark's terms. Figure and table "
                "captions are excluded from both sides, matching the official "
                "text-evidence-only setting.",
        "split": SPLIT,
        "papers": len(chosen),
        "questions": len(predictions),
        "model": generator.MODEL_NAME,
        "backend": generator.BACKEND,
        "embedder": model_key(),
        "chunking": CHUNKING,
        "k": TOP_K,
        "abstain": generator.ABSTAIN,
        "answer_f1": scores["Answer F1"],
        "evidence_f1": scores["Evidence F1"],
        "answer_f1_by_type": scores["Answer F1 by type"],
        "missing_predictions": scores["Missing predictions"],
        "declined": declined,
        "gold_unanswerable": gold_unanswerable,
        "seconds": round(time.time() - started, 1),
        "predictions": predictions,
        "records": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
