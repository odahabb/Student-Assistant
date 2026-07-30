"""
backend/scripts/draft_candidate_ground_truth.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Produces DRAFT candidate (question, answer, source_file, page) pairs for a
possible retrieval ground-truth expansion, from the four PDFs already in
data/raw/ and from pages not used by the current 8-question set.

The supporting quote is NOT typed by hand: each candidate carries a short
anchor string, and the script lifts the surrounding sentence verbatim out of
the page text. That guarantees the quote matches the source exactly (these PDFs
use the "fi"/"fl" ligatures, which hand-typed quotes reproduce incorrectly) and
gives a page-anchored snippet to spot-check each pair against.

Output is DRAFT material for review:
  - written to data/eval/candidate_ground_truth_DRAFT.json
  - NOT merged into retrieval_ground_truth.json
  - NOT run through the eval script

Questions and answers below are authored; the quotes and page numbers are
machine-verified. Every pair still needs a human read before use.

Run from anywhere:
    python backend/scripts/draft_candidate_ground_truth.py
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "eval" / "candidate_ground_truth_DRAFT.json"

# Pages already used by the official 8-question set — deliberately avoided here.
PAGES_IN_USE = {
    ("embedding.pdf", 1), ("embedding.pdf", 2),
    ("Whisper.pdf", 1), ("Whisper.pdf", 3),
    ("Flant5pdf.pdf", 2), ("Flant5pdf.pdf", 4),
    ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 1),
    ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 3),
}

# (source_file, page, anchor, question, answer)
# `anchor` is a distinctive ASCII substring of the sentence the answer comes
# from; it avoids "fi"/"fl" so it matches the ligature-bearing extracted text.
CANDIDATES = [
    ("embedding.pdf", 4, "built upon the all-MiniLM-L6-v2 architecture",
     "Which base model architecture are the fine-tuned embedding models built on?",
     "all-MiniLM-L6-v2"),
    ("embedding.pdf", 4, "The positive examples, totaling 2,710 pairs",
     "How many positive sentence pairs were included in the fine-tuning dataset?",
     "2,710 pairs"),
    ("embedding.pdf", 6, "with a learning rate of 2e-5",
     "What learning rate was used for the MNRL-only fine-tuning run?",
     "2e-5"),
    ("embedding.pdf", 6, "lower learning rate of 1e-5",
     "What learning rate was used for the dual-loss (MNRL + CosineSimilarityLoss) run?",
     "1e-5"),
    ("embedding.pdf", 7, "28 diverse university syllabus",
     "How many university syllabus files make up the retrieval evaluation corpus?",
     "28 syllabus files"),
    ("embedding.pdf", 7, "grouped into three categories",
     "What three categories are the evaluation questions grouped into?",
     "Course Information, Faculty Information, and Teaching Assistant Information"),

    ("Whisper.pdf", 2, "117,000 hours cover 96 other languages",
     "How many of Whisper's 680,000 training hours are non-English, and how many "
     "languages do they cover?",
     "117,000 hours covering 96 other languages"),
    ("Whisper.pdf", 2, "125,000 hours of X",
     "How many hours of speech translation data does the Whisper training set include?",
     "125,000 hours of X-to-English translation data"),
    ("Whisper.pdf", 5, "1550M",
     "How many parameters does the Whisper Large model have, per the architecture table?",
     "1550M parameters"),
    ("Whisper.pdf", 6, "clean-test WER of 2.5",
     "What LibriSpeech clean-test WER does the best zero-shot Whisper model achieve?",
     "2.5"),
    ("Whisper.pdf", 6, "12 other academic speech recognition datasets",
     "How many additional academic speech recognition datasets are used to study "
     "out-of-distribution behaviour?",
     "12 datasets"),

    ("Flant5pdf.pdf", 1, "75.2%",
     "What five-shot MMLU score does Flan-PaLM 540B achieve?",
     "75.2%"),
    ("Flant5pdf.pdf", 3, "473 datasets, 146 task categories",
     "How many datasets, task categories and total tasks make up the Flan finetuning data?",
     "473 datasets, 146 task categories, and 1,836 total tasks"),
    ("Flant5pdf.pdf", 5, "780M Flan-T5-Large",
     "How many parameters does Flan-T5-Large have, per Table 2?",
     "780M"),
    ("Flant5pdf.pdf", 6, "up to 282 tasks",
     "Beyond how many finetuning tasks does the majority of the performance "
     "improvement stop accruing?",
     "282 tasks"),
    ("Flant5pdf.pdf", 6, "780B tokens",
     "How many tokens are in PaLM's pre-training data compared with instruction "
     "finetuning?",
     "780B pre-training tokens vs 1.4B finetuning tokens (0.2%)"),

    ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 2, "530 Microsoft and NVIDIA",
     "According to the LLM comparison table, how many parameters does MT-NLG have "
     "and which companies produced it?",
     "530 billion parameters, Microsoft and NVIDIA"),
    ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 2,
     "two major parts: the decoder and the encoder",
     "What two major parts make up the transformer architecture?",
     "the encoder and the decoder"),
    ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 4,
     "learns the training data too well",
     "How does the paper define overfitting as a cause of hallucinations?",
     "the model learns the training data too well and cannot generalize to new data"),
    ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 5,
     "Techniques such as dropout",
     "Which regularization technique is given as an example for reducing hallucinations?",
     "dropout"),
]

WINDOW = 180


def page_text(doc, page_number):
    """Whitespace-collapsed text of a 1-indexed page."""
    return " ".join(doc[page_number - 1].get_text().split())


def extract_sentence(text, anchor):
    """
    The sentence containing `anchor`, lifted verbatim. Falls back to a character
    window when sentence boundaries are unhelpful (tables, figure captions).
    """
    pos = text.find(anchor)
    if pos == -1:
        return None, None

    start = max(0, pos - WINDOW)
    end = min(len(text), pos + len(anchor) + WINDOW)
    context = text[start:end]

    left = text.rfind(". ", 0, pos)
    right = text.find(". ", pos + len(anchor))
    if left != -1 and right != -1 and (right - left) < 600:
        sentence = text[left + 2:right + 1]
    else:
        sentence = context
    return sentence, context


def main():
    import fitz

    print("=" * 78)
    print("DRAFT CANDIDATE GROUND-TRUTH EXPANSION")
    print("=" * 78)

    docs = {}
    for name in sorted({c[0] for c in CANDIDATES}):
        docs[name] = fitz.open(str(RAW_DIR / name))

    out, failures = [], []
    for source_file, page, anchor, question, answer in CANDIDATES:
        if (source_file, page) in PAGES_IN_USE:
            failures.append((source_file, page, anchor, "page already used by official set"))
            continue

        doc = docs[source_file]
        if page > len(doc):
            failures.append((source_file, page, anchor, "page out of range"))
            continue

        text = page_text(doc, page)
        sentence, context = extract_sentence(text, anchor)
        if sentence is None:
            failures.append((source_file, page, anchor, "anchor not found on page"))
            continue

        # confirm the anchor appears on this page and on no earlier page, so the
        # page attribution is unambiguous
        also_on = [p for p in range(1, len(doc) + 1)
                   if p != page and anchor in page_text(doc, p)]

        out.append({
            "status": "DRAFT - unverified, needs human review",
            "question": question,
            "answer": answer,
            "source_file": source_file,
            "page": page,
            "supporting_quote": sentence.strip(),
            "quote_context": context.strip(),
            "anchor_used": anchor,
            "quote_verified_verbatim_on_page": True,
            "anchor_also_appears_on_pages": also_on,
        })

    for name, doc in docs.items():
        doc.close()

    # ---- report --------------------------------------------------------------
    by_doc = {}
    for c in out:
        by_doc.setdefault(c["source_file"], []).append(c["page"])

    print(f"\nProduced {len(out)} draft candidates "
          f"(target was 10-15; {len(failures)} rejected)\n")
    for name, pages in by_doc.items():
        print(f"  {name:<52} {len(pages):>2} candidates, pages {sorted(set(pages))}")

    if failures:
        print("\n  REJECTED:")
        for source_file, page, anchor, why in failures:
            print(f"    {source_file} p.{page} ({anchor[:40]!r}): {why}")

    ambiguous = [c for c in out if c["anchor_also_appears_on_pages"]]
    if ambiguous:
        print("\n  NOTE - anchor text also appears on other pages "
              "(page attribution worth double-checking):")
        for c in ambiguous:
            print(f"    {c['source_file']} p.{c['page']} also on "
                  f"{c['anchor_also_appears_on_pages']}: {c['anchor_used'][:50]!r}")

    print()
    print("=" * 78)
    print("CANDIDATES")
    print("=" * 78)
    for i, c in enumerate(out, 1):
        print(f"\n{i:>2}. [{c['source_file']} p.{c['page']}]")
        print(f"    Q: {c['question']}")
        print(f"    A: {c['answer']}")
        print(f"    quote: \"{c['supporting_quote'][:230]}\"")

    payload = {
        "status": "DRAFT - candidate pairs for review. NOT official ground truth.",
        "note": "Generated from the four PDFs already in data/raw/, from pages not "
                "used by the current 8-question set. Questions and answers are "
                "authored; supporting quotes and page numbers are extracted and "
                "verified verbatim from the source page. Not merged into "
                "retrieval_ground_truth.json and not run through eval_recall.py.",
        "pages_excluded_as_already_in_use": [
            {"source_file": f, "page": p} for f, p in sorted(PAGES_IN_USE)],
        "candidate_count": len(out),
        "candidates": out,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
