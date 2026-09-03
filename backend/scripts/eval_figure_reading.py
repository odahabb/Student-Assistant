"""
backend/scripts/eval_figure_reading.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Measures what reading a page as a picture buys, and what it costs.

Slides are largely pictures: a diagram, a screenshot of code, a chart. When a
page's text layer is thin, loader._read_page_picture renders the page and
reads it, cheapest tool first — EasyOCR, and Qwen2-VL only when OCR comes back
with almost nothing and the picture covers most of the page
(FIGURE_VISION_IF_FEWER, FIGURE_VISION_MIN_AREA, at most FIGURE_VISION_PER_DOC
pages per document). The comparison of Qwen2-VL against EasyOCR+BLIP on DocVQA
says which reader is better; it says nothing about this cascade, which is what
this script measures.

Two questions, two parts:

  1. What does it cost? A sample of candidate pages is read again with the
     cache bypassed, recording which tool answered, how many words came back
     and how long it took. Escalation to the vision model is rare by design
     and expensive when it happens, so both rates and seconds are reported.

  2. What does it recover? The picture questions in the slide ground truth
     were written only from text the text layer does not contain. Retrieval is
     run over an index built without reading pictures and over one built with
     it. The first is a floor: those answers are not in the index at all, and
     anything it retrieves is a coincidence of wording.

The text_layer questions are run through both indexes as a control: reading
pictures adds chunks to the corpus, and that could in principle push the
answers to ordinary questions down the ranking.

Run from anywhere (the cost sample re-reads pages, so it is slow):
    SA_EMBEDDER=bge-small python backend/scripts/eval_figure_reading.py
    ... --sample 20      (fewer pages in the cost sample)
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
import fitz  # noqa: E402
import numpy as np  # noqa: E402

from backend.pipeline import loader  # noqa: E402
from backend.pipeline import sparse as sparse_module  # noqa: E402
from backend.pipeline.embedder import embed, model_key  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.retriever import DENSE_WEIGHT, retrieve  # noqa: E402

DECK_DIR = ROOT / "data" / "projects" / "AI"
EVAL_DIR = ROOT / "data" / "eval"
GROUND_TRUTH_PATH = EVAL_DIR / "slide_ground_truth.json"
OUT_PATH = EVAL_DIR / "figure_reading.json"

SEED = 5
K_VALUES = [1, 3, 5]
SAMPLE_PAGES = (int(sys.argv[sys.argv.index("--sample") + 1])
                if "--sample" in sys.argv else 40)


def cost_sample(paths, rng):
    """
    Read a sample of candidate pages with the cache bypassed, recording the
    tool that answered, the words recovered and the seconds taken.
    """
    candidates = []
    for path in paths:
        with fitz.open(str(path)) as doc:
            texts = [page.get_text() for page in doc]
            for i, text in enumerate(texts):
                if loader._needs_figure_reading(text, doc[i]):
                    candidates.append((str(path), i,
                                       loader._picture_share(doc[i])))
    rng.shuffle(candidates)
    chosen = candidates[:SAMPLE_PAGES]
    print(f"  {len(candidates)} candidate pages across {len(paths)} decks; "
          f"reading {len(chosen)} of them with the cache off\n")

    # Bypass the cache so the real cost is measured, and do not write the
    # result back. Writing it back is not harmless: the sample allows the
    # vision model on any page large enough, ignoring the per-document budget
    # the application applies, so a page could be cached with different text
    # from the one the application would have stored. An earlier version of
    # this script did write back and cost the corpus a third of its chunks.
    original_read = loader._cached_figure_text
    original_write = loader._cache_figure_text
    loader._cached_figure_text = lambda digest: None
    loader._cache_figure_text = lambda digest, text: None
    records = []
    try:
        for path, i, share in chosen:
            with fitz.open(path) as doc:
                may_describe = share >= loader.FIGURE_VISION_MIN_AREA
                t0 = time.time()
                text, method = loader._read_page_picture(
                    doc[i], i + 1, allow_vision=may_describe)
                records.append({
                    "file": Path(path).name,
                    "page": i + 1,
                    "picture_share": round(share, 3),
                    "vision_allowed": bool(may_describe),
                    "method": method or "nothing found",
                    "words": len(text.split()),
                    "seconds": round(time.time() - t0, 2),
                })
            r = records[-1]
            print(f"    {r['file'][:34]:<34} p.{r['page']:<3} "
                  f"{r['method']:<14} {r['words']:>4} words  {r['seconds']:>6.1f}s")
    finally:
        loader._cached_figure_text = original_read
        loader._cache_figure_text = original_write

    by_method = {}
    for r in records:
        m = by_method.setdefault(r["method"], {"pages": 0, "seconds": [], "words": []})
        m["pages"] += 1
        m["seconds"].append(r["seconds"])
        m["words"].append(r["words"])
    summary = {m: {"pages": v["pages"],
                   "share": round(v["pages"] / len(records), 3),
                   "median_seconds": round(statistics.median(v["seconds"]), 2),
                   "total_seconds": round(sum(v["seconds"]), 1),
                   "median_words": round(statistics.median(v["words"]), 1)}
               for m, v in sorted(by_method.items())}
    return {"candidate_pages": len(candidates), "sampled": len(records),
            "by_method": summary, "pages": records}


def build(pages_by_deck):
    chunks = []
    for pages in pages_by_deck:
        chunks.extend(preprocess(pages))
    embeddings = embed(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return chunks, index, sparse_module.build_index(chunks)


def covers(chunk, source_file, page) -> bool:
    if getattr(chunk, "source_file", None) != source_file:
        return False
    first = getattr(chunk, "page", None)
    if first is None:
        return False
    return first <= page <= (getattr(chunk, "page_end", None) or first)


def evaluate(chunks, index, keywords, questions):
    ranks = []
    for entry in questions:
        got = retrieve(entry["question"], index, chunks, k=max(K_VALUES),
                       sparse=keywords, dense_weight=DENSE_WEIGHT)
        ranks.append(next((i for i, c in enumerate(got, 1)
                           if covers(c, entry["source_file"], entry["page"])), None))
    total = len(questions) or 1
    return {
        "recall": {f"recall_at_{k}": round(sum(1 for r in ranks if r and r <= k) / total, 3)
                   for k in K_VALUES},
        "mrr_top5": round(statistics.mean(1 / r if r else 0 for r in ranks), 4),
        "ranks": ranks,
    }


def main():
    # Slide text carries Unicode the Windows console cannot encode.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    rng = random.Random(SEED)
    truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    picture_questions = [q for q in truth["questions"] if q["source"] == "picture"]
    text_questions = [q for q in truth["questions"] if q["source"] == "text_layer"]
    paths = sorted(p for p in DECK_DIR.glob("*.pdf"))

    print("WHAT IT COSTS")
    if SAMPLE_PAGES:
        cost = cost_sample(paths, rng)
    else:
        # --sample 0 keeps the cost measured by an earlier run and re-runs only
        # the retrieval half, which is cheap and does not touch any model.
        cost = json.loads(OUT_PATH.read_text(encoding="utf-8"))["cost"]
        print(f"  reusing the sample of {cost['sampled']} pages already recorded")
    print(f"\n  {json.dumps(cost['by_method'], indent=2)}")

    print("\nWHAT IT RECOVERS")
    print(f"  {len(picture_questions)} picture questions, "
          f"{len(text_questions)} text-layer questions as a control\n")

    indexes = {}
    for setting in ("off", "auto"):
        t0 = time.time()
        pages_by_deck = [load(p, setting) for p in paths]
        chunks, index, keywords = build(pages_by_deck)
        indexes[setting] = {
            "chunks": len(chunks),
            "pages_with_text": sum(len(p) for p in pages_by_deck),
            "index_seconds": round(time.time() - t0, 1),
            "picture_questions": evaluate(chunks, index, keywords, picture_questions),
            "text_layer_questions": evaluate(chunks, index, keywords, text_questions),
        }

    print(f"{'figures':<9} {'chunks':>7} {'R@1':>6} {'R@3':>6} {'R@5':>6} "
          f"{'MRR':>7}   questions")
    for setting, r in indexes.items():
        for label in ("picture_questions", "text_layer_questions"):
            rec = r[label]["recall"]
            print(f"{setting if label.startswith('picture') else '':<9} "
                  f"{r['chunks'] if label.startswith('picture') else '':>7} "
                  f"{rec['recall_at_1']:>6} {rec['recall_at_3']:>6} "
                  f"{rec['recall_at_5']:>6} {r[label]['mrr_top5']:>7}   {label}")

    OUT_PATH.write_text(json.dumps({
        "note": "Cost and benefit of reading PDF pages as pictures. The cost "
                "sample bypasses the figure cache so each page is read for "
                "real. The retrieval halves compare an index built with "
                "figures='off' against one built with figures='auto'; picture "
                "questions were written only from text the text layer does "
                "not contain, so the 'off' row is a floor, and the text-layer "
                "questions are a control for the extra chunks.",
        "embedder": model_key(),
        "dense_weight": DENSE_WEIGHT,
        "seed": SEED,
        "thresholds": {
            "vision_if_fewer_words": loader.FIGURE_VISION_IF_FEWER,
            "vision_min_picture_share": loader.FIGURE_VISION_MIN_AREA,
            "vision_pages_per_document": loader.FIGURE_VISION_PER_DOC,
            "render_dpi": loader.FIGURE_RENDER_DPI,
        },
        "cost": cost,
        "retrieval": indexes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\n  -> {OUT_PATH.relative_to(ROOT)}")


def load(path, figures):
    return loader.load_pdf(str(path), figures=figures)


if __name__ == "__main__":
    main()
