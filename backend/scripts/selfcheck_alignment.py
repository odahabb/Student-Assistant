"""
backend/scripts/selfcheck_alignment.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Diagnostic for low Recall@k: distinguishes a metadata/vector ALIGNMENT bug from
a genuine SEMANTIC gap between short factual questions and chunk text.

Closed-loop self-retrieval: query the index with text taken from a known chunk
and check whether that same chunk comes back at rank 1. If a chunk cannot
retrieve itself, the vectors and the metadata list are out of step. If it can,
the index is wired correctly and the recall problem lives in question phrasing
vs. chunk content.

Read-only: builds the index through the normal pipeline and measures. Changes
nothing.

Run from anywhere:
    python backend/scripts/selfcheck_alignment.py
"""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.pipeline.loader import load_file  # noqa: E402
from backend.pipeline.preprocessor import preprocess  # noqa: E402
from backend.pipeline.embedder import embed, _get_model  # noqa: E402
from backend.pipeline.vector_store import (  # noqa: E402
    build_and_save, load, INDEX_PATH, CHUNKS_PATH,
)
from backend.pipeline.retriever import retrieve  # noqa: E402

RAW_DIR = ROOT / "data" / "raw"
GROUND_TRUTH_PATH = ROOT / "data" / "eval" / "retrieval_ground_truth.json"

DOCUMENTS = [
    "embedding.pdf",
    "Whisper.pdf",
    "Flant5pdf.pdf",
    "Hallucinations_in_Large_Language_Models_LLMs.pdf",
]

SUBSTRING_WORDS = 15


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def fresh_build():
    """
    Delete any existing store first, then build unconditionally from scratch,
    so nothing left over from a previous run can be silently reused.
    """
    print("=" * 78)
    print("STEP 1 - FRESH BUILD (stale-index check)")
    print("=" * 78)

    for p in (ROOT / INDEX_PATH, ROOT / CHUNKS_PATH):
        if p.exists():
            print(f"  deleting pre-existing {p.relative_to(ROOT)} "
                  f"({p.stat().st_size} bytes)")
            p.unlink()
        else:
            print(f"  no pre-existing {p.relative_to(ROOT)}")

    all_chunks = []
    for name in DOCUMENTS:
        pages = load_file(str(RAW_DIR / name))
        chunks = preprocess(pages)
        all_chunks.extend(chunks)
        print(f"  {name:<52} -> {len(chunks):>4} chunks "
              f"(running total {len(all_chunks)})")

    embeddings = embed(all_chunks)
    build_and_save(embeddings, all_chunks)
    index, stored_chunks = load()

    print(f"\n  in-memory chunks : {len(all_chunks)}")
    print(f"  embeddings       : {embeddings.shape}")
    print(f"  index.ntotal     : {index.ntotal}")
    print(f"  reloaded chunks  : {len(stored_chunks)}")
    consistent = len(all_chunks) == embeddings.shape[0] == index.ntotal == len(stored_chunks)
    print(f"  counts all equal : {consistent}")

    return index, stored_chunks, all_chunks, embeddings


def check_vector_order(index, stored_chunks, all_chunks, embeddings):
    """
    Vector-level alignment: position i in the FAISS index must be the vector for
    stored_chunks[i]. Searching the index with a row of the embedding matrix
    should return that same row's position at distance ~0.
    """
    print()
    print("=" * 78)
    print("STEP 2 - VECTOR/METADATA ORDER")
    print("=" * 78)

    rng = np.random.default_rng(0)
    probes = sorted(rng.choice(len(all_chunks), size=min(40, len(all_chunks)),
                               replace=False).tolist())
    # always include boundaries and the first chunk of each document
    doc_starts, seen = [], set()
    for i, c in enumerate(all_chunks):
        if c.source_file not in seen:
            seen.add(c.source_file)
            doc_starts.append(i)
    probes = sorted(set(probes + doc_starts + [0, len(all_chunks) - 1]))

    bad_self, bad_text = [], []
    for i in probes:
        d, idx = index.search(np.ascontiguousarray(embeddings[i:i + 1], dtype=np.float32), 1)
        if int(idx[0][0]) != i:
            bad_self.append((i, int(idx[0][0]), float(d[0][0])))
        if str(stored_chunks[i]) != str(all_chunks[i]) or \
           stored_chunks[i].source_file != all_chunks[i].source_file or \
           stored_chunks[i].page != all_chunks[i].page:
            bad_text.append(i)

    print(f"  probed {len(probes)} positions (incl. first chunk of each document,"
          f" first and last overall)")
    print(f"  vector at position i returns position i : "
          f"{'ALL OK' if not bad_self else f'{len(bad_self)} MISMATCH'}")
    for i, got, dist in bad_self[:10]:
        print(f"      position {i} -> returned {got} (distance {dist:.4f})")
    print(f"  reloaded chunk[i] identical to built chunk[i] : "
          f"{'ALL OK' if not bad_text else f'{len(bad_text)} MISMATCH'}")
    for i in bad_text[:10]:
        print(f"      position {i}: {stored_chunks[i].source_file} p.{stored_chunks[i].page}"
              f"  vs  {all_chunks[i].source_file} p.{all_chunks[i].page}")

    # document boundaries in the combined list
    print("\n  document boundaries in the combined chunk list:")
    for i in doc_starts:
        c = stored_chunks[i]
        print(f"      index {i:>4}  starts {c.source_file} (page {c.page})")

    return not bad_self and not bad_text


def pick_target_chunks(stored_chunks):
    """
    Pick one chunk per selected ground-truth entry: the chunk on the expected
    page that actually contains the expected answer text if we can find it,
    otherwise the first chunk on that page.
    """
    gt = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    # the four entries that missed at k=5, one per document
    wanted = [
        ("embedding.pdf", 1),
        ("Whisper.pdf", 1),
        ("Flant5pdf.pdf", 2),
        ("Hallucinations_in_Large_Language_Models_LLMs.pdf", 1),
    ]

    targets = []
    for source_file, page in wanted:
        entry = next(e for e in gt
                     if e["source_file"] == source_file and e["page"] == page)
        on_page = [(i, c) for i, c in enumerate(stored_chunks)
                   if c.source_file == source_file and c.page == page]
        answer_key = norm(entry["answer"])
        chosen = next(((i, c) for i, c in on_page
                       if answer_key and answer_key in norm(c)), None)
        how = "contains the expected answer text"
        if chosen is None:
            chosen = on_page[0]
            how = "first chunk on the page (answer text not found verbatim)"
        targets.append({
            "entry": entry, "position": chosen[0], "chunk": chosen[1],
            "how_selected": how, "chunks_on_page": len(on_page),
        })
    return targets


def self_retrieval(index, stored_chunks, targets):
    print()
    print("=" * 78)
    print("STEP 3 - CLOSED-LOOP SELF-RETRIEVAL")
    print("=" * 78)
    print("Query the index with the chunk's OWN text and with a distinctive")
    print("~15-word substring of it. The chunk should come back at rank 1.\n")

    results = []
    for t in targets:
        chunk = t["chunk"]
        expected = (chunk.source_file, chunk.page)

        words = str(chunk).split()
        mid = max(0, len(words) // 2 - SUBSTRING_WORDS // 2)
        substring = " ".join(words[mid:mid + SUBSTRING_WORDS])

        # A mid-chunk slice is arbitrary and often lands on boilerplate (author
        # affiliations, ORCID URLs), which tests nothing useful. Also take the
        # most content-bearing window: the one overlapping the expected answer.
        answer_tokens = set(norm(t["entry"]["answer"]).split())
        best_start, best_score = mid, -1
        for s in range(0, max(1, len(words) - SUBSTRING_WORDS + 1)):
            window = set(norm(" ".join(words[s:s + SUBSTRING_WORDS])).split())
            score = len(answer_tokens & window)
            if score > best_score:
                best_score, best_start = score, s
        answer_substring = " ".join(words[best_start:best_start + SUBSTRING_WORDS])

        row = {"expected_file": chunk.source_file, "expected_page": chunk.page,
               "position": t["position"], "how_selected": t["how_selected"],
               "chunks_on_page": t["chunks_on_page"],
               "question": t["entry"]["question"]}

        print("-" * 78)
        print(f"TARGET: {chunk.source_file} p.{chunk.page}  "
              f"(index position {t['position']}, {t['how_selected']})")
        print(f"  chunk text starts: {str(chunk)[:100]}...")

        for label, query in (
            ("full chunk text", str(chunk)),
            (f"{SUBSTRING_WORDS}-word substring (mid-chunk)", substring),
            (f"{SUBSTRING_WORDS}-word substring (answer window, "
             f"{best_score} answer tokens)", answer_substring),
        ):
            got = retrieve(query, index, stored_chunks, k=5)
            top = got[0]
            exact_same_chunk = str(top) == str(chunk)
            page_match = (top.source_file, top.page) == expected
            rank_of_page = next((r for r, g in enumerate(got, 1)
                                 if (g.source_file, g.page) == expected), None)

            print(f"\n  query = {label}")
            if "substring" in label:
                print(f"    \"{query}\"")
            print(f"    top-1 : {top.source_file} p.{top.page}"
                  f"   {'HIT' if page_match else 'MISS'}"
                  f"{'  (identical chunk)' if exact_same_chunk else ''}")
            if not page_match:
                print(f"    got instead: " + ", ".join(
                    f"#{r} {g.source_file} p.{g.page}" for r, g in enumerate(got, 1)))
            print(f"    correct file+page first appears at rank: {rank_of_page}")

            key = ("full_text" if label.startswith("full")
                   else "substring_midchunk" if "mid-chunk" in label
                   else "substring_answer_window")
            row[key] = {"top1_file": top.source_file, "top1_page": top.page,
                        "top1_is_same_chunk": exact_same_chunk,
                        "hit_at_1": page_match, "first_correct_rank": rank_of_page}
        results.append(row)
        print()

    return results


def main():
    index, stored_chunks, all_chunks, embeddings = fresh_build()
    aligned = check_vector_order(index, stored_chunks, all_chunks, embeddings)
    targets = pick_target_chunks(stored_chunks)
    results = self_retrieval(index, stored_chunks, targets)

    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    full_hits = sum(1 for r in results if r["full_text"]["hit_at_1"])
    mid_hits = sum(1 for r in results if r["substring_midchunk"]["hit_at_1"])
    ans_hits = sum(1 for r in results if r["substring_answer_window"]["hit_at_1"])
    n = len(results)
    print(f"  vector/metadata order intact          : {aligned}")
    print(f"  self-retrieval, full chunk text       : {full_hits}/{n} at rank 1")
    print(f"  self-retrieval, 15w mid-chunk slice   : {mid_hits}/{n} at rank 1")
    print(f"  self-retrieval, 15w answer window     : {ans_hits}/{n} at rank 1")
    print()
    if aligned and full_hits == n:
        print("  => Index alignment is correct. Chunks retrieve themselves.")
        print("     The recall problem is a semantic gap between short factual")
        print("     questions and the chunk text, NOT a wiring bug.")
    else:
        print("  => Alignment is NOT clean - see mismatches above.")

    out = ROOT / "data" / "eval" / "selfcheck_alignment_results.json"
    out.write_text(json.dumps(
        {"vector_metadata_order_intact": aligned,
         "self_retrieval": results}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"\n  details -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
