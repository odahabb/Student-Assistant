"""
backend/pipeline/preprocessor.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 2 of pipeline: PREPROCESSING
Cleans raw text and splits it into overlapping chunks for embedding.

Chunking is page-bounded for PDFs: each page is windowed independently, so no
chunk ever spans a page boundary, and every chunk records the file and page it
came from (see Chunk below).
"""

import re
from typing import Iterator, List, Optional, Sequence, Union

from backend.pipeline.chunk import Chunk
from backend.pipeline.embedder import _get_model

# Re-exported so `from backend.pipeline.preprocessor import Chunk` keeps working
# for callers that think of Chunk as this stage's output type.
__all__ = ["Chunk", "preprocess"]


def _as_pages(source: Union[str, Sequence[dict], dict],
              source_file: Optional[str]) -> List[dict]:
    """
    Normalise preprocess()'s input into a list of {source_file, page, text}
    dicts, so the chunking loop below has one shape to deal with.

    A plain string (image / audio / plain-text input) becomes a single
    page-less entry; loader.load_pdf()'s per-page list passes through with its
    page numbers intact.
    """
    if isinstance(source, str):
        return [{"source_file": source_file, "page": None, "text": source}]

    if isinstance(source, dict):
        source = [source]

    if not isinstance(source, (list, tuple)):
        raise TypeError(
            f"Expected a string or a list of page dicts, got {type(source)}"
        )

    pages = []
    for entry in source:
        if not isinstance(entry, dict):
            raise TypeError(
                f"Expected page dicts with a 'text' key, got {type(entry)}"
            )
        if "text" not in entry:
            raise TypeError("Page dict is missing required key 'text'")
        pages.append({
            "source_file": entry.get("source_file", source_file),
            "page": entry.get("page"),
            "text": entry["text"],
        })
    return pages


def _window(tokenizer, cleaned: str, chunk_tokens: int,
            overlap: int) -> Iterator[str]:
    """
    Split one page's cleaned text into overlapping token windows.

    This is the original chunking loop, unchanged — it just operates on a
    single page's text now instead of the whole document.
    """
    token_ids = tokenizer.encode(cleaned, add_special_tokens=False)
    if not token_ids:
        return

    step = chunk_tokens - overlap
    start = 0
    while start < len(token_ids):
        window = token_ids[start:start + chunk_tokens]
        yield tokenizer.decode(window)
        if start + chunk_tokens >= len(token_ids):
            break
        start += step


def preprocess(text: Union[str, Sequence[dict], dict], chunk_tokens: int = 220,
               overlap: int = 40, source_file: Optional[str] = None) -> List[Chunk]:
    """
    Clean raw text and split it into overlapping chunks sized to fit the
    embedder's 256-token limit (220-token windows, 40-token overlap by default).

    Accepts either:
      - a plain string — image, audio or plain-text input; pass source_file
        explicitly if you want the chunks tagged with a filename, or
      - the per-page list returned by loader.load_pdf():
        [{"source_file": ..., "page": ..., "text": ...}, ...]

    Pages are chunked independently, so a chunk never spans a page boundary.
    Every returned Chunk carries .source_file and .page (both None for plain
    string input with no source_file given).
    """
    pages = _as_pages(text, source_file)

    for page in pages:
        if not isinstance(page["text"], str):
            raise TypeError(f"Expected string, got {type(page['text'])}")

    # Length check is on the document as a whole, as before — a short page in
    # an otherwise-fine PDF isn't an extraction failure.
    combined = " ".join(page["text"] for page in pages)
    if len(combined.strip()) < 20:
        raise ValueError("Input text is too short — OCR likely failed")

    tokenizer = _get_model().tokenizer

    chunks: List[Chunk] = []
    for page in pages:
        # Collapse extra whitespace
        cleaned = re.sub(r'\s+', ' ', page["text"]).strip()
        if not cleaned:
            continue
        for window_text in _window(tokenizer, cleaned, chunk_tokens, overlap):
            chunks.append(Chunk(
                window_text,
                source_file=page["source_file"],
                page=page["page"],
            ))

    return chunks
