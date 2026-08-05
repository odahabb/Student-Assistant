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

import logging
import re
from typing import Iterator, List, Optional, Sequence, Union

from backend.pipeline.chunk import Chunk
from backend.pipeline.embedder import _get_model

log = logging.getLogger(__name__)

# Re-exported so `from backend.pipeline.preprocessor import Chunk` keeps working
# for callers that think of Chunk as this stage's output type.
__all__ = ["Chunk", "preprocess"]


# Page-1 boilerplate stripping
#
# A paper's first page mixes the abstract — which is dense with the facts a
# student actually asks about — with author names, affiliations, ORCID URLs and
# emails. Because chunks are fixed-width token windows, that boilerplate shares
# a window with real content and drags the window's embedding away from the
# topic, so the answer-bearing chunk loses to more topically uniform chunks
# elsewhere in the document.
#
# This strips those lines from page 1 only. It is deliberately conservative:
# every rule needs a positive signal of boilerplate, because dropping real
# content is worse than leaving some boilerplate behind. Titles and section
# headers are protected — prose and titles contain lowercase function words
# ("via", "of", "and"), which author and affiliation lines do not.

_EMAIL_RE = re.compile(r'[^\s@]+@[^\s@]+\.[A-Za-z]{2,}')
_ORCID_RE = re.compile(r'orcid', re.I)
# Case-sensitive on purpose: an affiliation names an institution as a proper
# noun ("University of Iowa"), whereas prose uses the same words in lower case
# ("conducted on 28 university course syllabi", "artificial intelligence
# laboratory of MIT") and must not be stripped.
_INSTITUTION_RE = re.compile(
    r'\b(?:University|Universite|Department|Dept\.|Institute|Institut|'
    r'College|Faculty|School of|Laborator(?:y|ies)|Academy|Hospital|'
    r'Center for|Centre for|Corresponding Author)\b')
# "G. Pradeep Reddy", "Y. V. Pavan Kumar" — initials followed by a surname
_INITIAL_NAME_RE = re.compile(r'\b[A-Z]\.\s*(?:[A-Z]\.\s*)*[A-Z][a-z]+')
_AUTHOR_MARKER_RE = re.compile(r'[∗*†‡§¶]')
_LOWERCASE_WORD_RE = re.compile(r'(?:^|\s)[a-z]{2,}')

# If a rule set somehow matches most of the page, assume the heuristic has
# misfired on an unusual layout and keep the page untouched.
_MAX_REMOVAL_FRACTION = 0.7


def _is_name_run_candidate(line: str) -> bool:
    """
    A line that looks like one entry in a block of author names — 1-5 tokens,
    all capitalised or markers, no lowercase words (e.g. "Shayne Longpre*",
    "Ed H. Chi"). Only stripped when several appear consecutively, so a lone
    section header such as "Related Work" survives.
    """
    stripped = _AUTHOR_MARKER_RE.sub('', line).strip()
    if not stripped or _LOWERCASE_WORD_RE.search(stripped):
        return False
    tokens = stripped.split()
    if not 1 <= len(tokens) <= 4:
        return False
    alpha = [t for t in tokens if any(ch.isalpha() for ch in t)]
    # At most three name tokens ("Ed H. Chi", "Shixiang Shane Gu"). Capping here
    # keeps four-word title case titles such as "Scaling Instruction-Finetuned
    # Language Models" out of the author run that immediately follows them.
    return 2 <= len(alpha) <= 3 and all(t[0].isupper() for t in alpha)


def _is_boilerplate_line(line: str) -> bool:
    """Single-line rules. Each needs an explicit boilerplate signal."""
    stripped = line.strip()
    if not stripped:
        return False

    if _EMAIL_RE.search(stripped):
        return True
    if _ORCID_RE.search(stripped):
        return True

    words = stripped.split()
    ends_sentence = stripped.endswith(('.', ':', ';'))
    # Affiliations are mostly proper nouns; prose is mostly lowercase words.
    lowercase_fraction = (len(_LOWERCASE_WORD_RE.findall(stripped)) / len(words)
                          if words else 0.0)

    # Affiliation line: institution proper noun, short, mostly proper nouns,
    # and not a prose sentence.
    if (len(words) <= 16 and _INSTITUTION_RE.search(stripped)
            and lowercase_fraction < 0.4 and not ends_sentence):
        return True

    # Author line by initials: "G. Pradeep Reddy§, Y. V. Pavan Kumar†, ..."
    if len(words) <= 20 and len(_INITIAL_NAME_RE.findall(stripped)) >= 2:
        return True

    # Author line with no lowercase words at all, plus affiliation markers or
    # numeric footnote keys: "Alec Radford * 1 Jong Wook Kim * 1 Tao Xu 1 ..."
    if not _LOWERCASE_WORD_RE.search(stripped):
        alpha_tokens = [t for t in words if any(ch.isalpha() for ch in t)]
        digit_tokens = [t for t in words if t.isdigit()]
        if len(alpha_tokens) >= 4 and (
                _AUTHOR_MARKER_RE.search(stripped) or len(digit_tokens) >= 2):
            return True

    return False


def _strip_page1_boilerplate(text: str, source_file: Optional[str] = None) -> str:
    """
    Remove author/affiliation/ORCID/email lines from a first page's text.

    Returns the text unchanged if the rules would remove most of the page,
    which would suggest the heuristic has misfired rather than that the page is
    genuinely almost all boilerplate.
    """
    lines = text.splitlines()
    drop = [_is_boilerplate_line(line) for line in lines]

    # The first non-blank line of page 1 is the document title. Never strip it —
    # it is the most useful line on the page for retrieval.
    first_non_blank = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_non_blank is not None:
        drop[first_non_blank] = False

    # Runs of >= 3 consecutive name-like lines: a one-name-per-line author block.
    candidate = [_is_name_run_candidate(line) for line in lines]
    run_start = None
    for i in range(len(lines) + 1):
        if i < len(lines) and candidate[i]:
            run_start = i if run_start is None else run_start
            continue
        if run_start is not None:
            if i - run_start >= 3:
                for j in range(run_start, i):
                    if j != first_non_blank:
                        drop[j] = True
            run_start = None

    non_blank = [i for i, line in enumerate(lines) if line.strip()]
    dropped = [i for i in non_blank if drop[i]]
    if not dropped:
        return text

    if non_blank and len(dropped) / len(non_blank) > _MAX_REMOVAL_FRACTION:
        log.warning(
            f"page-1 boilerplate filter would remove {len(dropped)}/{len(non_blank)} "
            f"lines of {source_file or 'input'} — leaving the page untouched"
        )
        return text

    kept = [line for i, line in enumerate(lines) if not drop[i]]
    log.info(
        f"page-1 boilerplate: removed {len(dropped)}/{len(non_blank)} lines "
        f"from {source_file or 'input'}"
    )
    return "\n".join(kept)


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
               overlap: int = 40, source_file: Optional[str] = None,
               strip_page1_boilerplate: bool = True) -> List[Chunk]:
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

    strip_page1_boilerplate removes author/affiliation/ORCID/email lines from
    page 1 only (see _strip_page1_boilerplate). Pass False to reproduce the
    behaviour from before that filter existed.
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
        page_text = page["text"]
        # Page 1 only — later pages carry no title-block boilerplate.
        if strip_page1_boilerplate and page["page"] == 1:
            page_text = _strip_page1_boilerplate(page_text, page["source_file"])

        # Collapse extra whitespace
        cleaned = re.sub(r'\s+', ' ', page_text).strip()
        if not cleaned:
            continue
        for window_text in _window(tokenizer, cleaned, chunk_tokens, overlap):
            chunks.append(Chunk(
                window_text,
                source_file=page["source_file"],
                page=page["page"],
            ))

    return chunks
