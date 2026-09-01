"""
backend/scripts/build_slide_ground_truth.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Builds a retrieval ground truth for slide decks, which the four-paper set does
not cover. The decks are the fifteen lecture decks in data/projects/AI —
real course material, not a benchmark, which is the point: the packing rules
in preprocessor._pack_slides were written for material like this.

Labelling by construction, not by judgement
-------------------------------------------
A question is written from ONE slide's text, taken straight from the loader
before any chunker has seen it, and that slide is the label. The label is
therefore a fact about how the question was produced, not an opinion about
where the answer lives — the author of this project never decides which page
is correct, so the evaluation cannot be talked into agreeing with the system.

Two pools come out of the same pass:

  text_layer — slides whose text PyMuPDF reads directly. Used to compare
               chunking strategies (eval_slide_chunking.py), which is fair
               because the questions are written before any chunking.
  picture    — slides where reading the rendered page recovered text the text
               layer does not hold. The question is written from that
               recovered text ALONE, so a system that does not read pictures
               cannot retrieve the answer at all (eval_figure_reading.py).

The question writer is the system's own model, so these questions are as
answerable as the quiz's. quiz.well_formed rejects the malformed ones; no
round-trip filter is applied, because that would need an index and would bias
the set towards whichever chunking built it.

One more filter is needed, and it is mechanical too. A question like "What is
the visible text in the document?" or "What does 4.207 represent?" names
nothing that could distinguish one slide from another, so no retriever could
be expected to find its source and scoring one against the label would measure
nothing. A question is therefore kept only if it contains at least one term
that is rare across the whole corpus (RARE_TERM_SHARE), counted over slides,
and a term that is only a slide number does not count. The threshold is
applied identically to every question before any retrieval runs.

Run from anywhere (slow the first time: it reads every deck twice):
    python backend/scripts/build_slide_ground_truth.py
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from backend.pipeline import quiz  # noqa: E402
from backend.pipeline.chunk import Chunk  # noqa: E402
from backend.pipeline.loader import load_pdf  # noqa: E402

DECK_DIR = ROOT / "data" / "projects" / "AI"
EVAL_DIR = ROOT / "data" / "eval"
OUT_PATH = EVAL_DIR / "slide_ground_truth.json"

SEED = 11
# A slide needs enough on it to ask about; a title and a picture caption do not.
MIN_SLIDE_WORDS = 30
MIN_PICTURE_WORDS = 20
SLIDES_PER_DECK = 3
PICTURES_PER_DECK = 2
# Attempts before giving up on a deck's pool: the question writer rejects
# roughly one attempt in eight.
MAX_ATTEMPTS = 4
# A question must name something that appears on at most this share of the
# corpus's slides, or it cannot point at one slide rather than another.
RARE_TERM_SHARE = 0.02
# Words too common to carry any of that weight, plus the numbering used in
# these decks' slide titles ("4.207"), which names a slide, not an idea.
_STOPWORDS = {
    "a", "about", "according", "all", "an", "and", "answer", "are", "as", "at",
    "be", "been", "between", "both", "but", "by", "can", "did", "do", "does",
    "document", "each", "example", "first", "for", "from", "give", "given",
    "has", "have", "how", "in", "into", "is", "it", "its", "list", "main",
    "make", "many", "may", "much", "must", "name", "new", "next", "not", "of",
    "on", "one", "only", "or", "other", "over", "page", "purpose", "recommended",
    "represent", "said", "same", "second", "shown", "slide", "so", "some",
    "specific", "step", "text", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "three", "to", "two", "type",
    "up", "use", "used", "using", "visible", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "within", "would", "you",
    "your",
}


def decks():
    return sorted(p for p in DECK_DIR.glob("*.pdf") if p.is_file())


def picture_only_text(with_pictures: str, without_pictures: str) -> str:
    """
    The text reading the rendered page added. load_pdf appends it to the text
    layer, so what the two passes do not share is what the picture gave.
    """
    if not without_pictures.strip():
        return with_pictures.strip()
    if with_pictures.startswith(without_pictures.rstrip()):
        return with_pictures[len(without_pictures.rstrip()):].strip()
    return ""


def terms(text: str):
    """Content words, lowercased, without slide numbers or stopwords."""
    words = re.findall(r"[a-z][a-z0-9_]{2,}", str(text).lower())
    return {w for w in words if w not in _STOPWORDS}


def read_deck(path: Path, report: dict):
    """(slide pools, every slide's text) for one deck."""
    name = path.name
    t0 = time.time()
    with_pictures = load_pdf(str(path), figures="auto")
    without_pictures = load_pdf(str(path), figures="off")
    plain = {p["page"]: p["text"] for p in without_pictures}
    print(f"  {name:<44} {len(with_pictures):>3} slides ({time.time() - t0:.0f}s)")

    text_pool, picture_pool = [], []
    for page in with_pictures:
        if page.get("divider"):
            continue
        layer = plain.get(page["page"], "")
        added = picture_only_text(page["text"], layer)
        if added and len(added.split()) >= MIN_PICTURE_WORDS:
            picture_pool.append((page, added))
        elif len(layer.split()) >= MIN_SLIDE_WORDS:
            text_pool.append((page, layer))

    report["slides"] += len(with_pictures)
    report["text_layer_candidates"] += len(text_pool)
    report["picture_candidates"] += len(picture_pool)
    return (name, text_pool, picture_pool), [p["text"] for p in with_pictures]


def questions_for(name, pool, source, wanted, rng, rare, report):
    """Up to `wanted` usable questions from one deck's pool."""
    rng.shuffle(pool)
    entries = []
    for page, text in pool[:wanted * MAX_ATTEMPTS]:
        if len(entries) >= wanted:
            break
        # no retrieve_fn: the round-trip filter would need an index
        item, reason = quiz.generate_item(Chunk(text, name, page["page"],
                                                page.get("section")))
        if item is None:
            report["rejected"][reason] = report["rejected"].get(reason, 0) + 1
            continue
        anchors = sorted(terms(item.question) & rare)
        if not anchors:
            report["rejected"]["names nothing rare"] = (
                report["rejected"].get("names nothing rare", 0) + 1)
            print(f"      [drop] {item.question}")
            continue
        entries.append({
            "question": item.question,
            "answer": item.answer,
            "source_file": name,
            "page": page["page"],
            "source": source,
            "section": page.get("section"),
            "title": page.get("title"),
            "rare_terms": anchors[:6],
            "slide_text": " ".join(text.split()),
        })
        print(f"      [{source}] p.{page['page']}: {item.question}")
    return entries


def main():
    # Slide text carries Unicode the Windows console cannot encode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = random.Random(SEED)
    report = {"slides": 0, "text_layer_candidates": 0, "picture_candidates": 0,
              "rejected": {}}
    paths = decks()
    print(f"{len(paths)} decks in {DECK_DIR.relative_to(ROOT)}\n")

    pools, corpus = [], []
    for path in paths:
        pool, texts = read_deck(path, report)
        pools.append(pool)
        corpus.extend(texts)

    # How many slides in the whole corpus each term appears on.
    document_frequency = Counter()
    for text in corpus:
        document_frequency.update(terms(text))
    cutoff = max(1, int(RARE_TERM_SHARE * len(corpus)))
    rare = {t for t, n in document_frequency.items() if n <= cutoff}
    print(f"\n{len(corpus)} slides, {len(document_frequency)} terms, "
          f"{len(rare)} on at most {cutoff} slide(s)\n")

    entries = []
    for name, text_pool, picture_pool in pools:
        print(f"  {name}")
        entries += questions_for(name, text_pool, "text_layer",
                                 SLIDES_PER_DECK, rng, rare, report)
        entries += questions_for(name, picture_pool, "picture",
                                 PICTURES_PER_DECK, rng, rare, report)

    by_source = {}
    for e in entries:
        by_source[e["source"]] = by_source.get(e["source"], 0) + 1

    OUT_PATH.write_text(json.dumps({
        "note": "Retrieval ground truth over lecture slide decks. Each "
                "question was written by the project's own question model "
                "from a single slide's text, taken from the loader before any "
                "chunking, so the labelled page is a fact about how the "
                "question was made. source='picture' means the question was "
                "written only from text recovered by reading the rendered "
                "page, which the PDF text layer does not contain.",
        "seed": SEED,
        "decks": len(paths),
        "rare_term_share": RARE_TERM_SHARE,
        "counts": {**report, "questions": len(entries), "by_source": by_source},
        "questions": entries,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  {len(entries)} questions ({by_source})")
    print(f"  rejected: {report['rejected']}")
    print(f"  -> {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
