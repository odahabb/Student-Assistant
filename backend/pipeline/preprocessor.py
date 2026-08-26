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
        return [{"source_file": source_file, "page": None, "section": None,
                 "text": source, "kind": "page", "title": None,
                 "divider": False, "start": None, "end": None,
                 "from_image": False}]

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
            "section": entry.get("section"),
            "text": entry["text"],
            # Set by loader.load_pdf for slides and load_audio_segments for
            # recordings; absent for ordinary pages and plain strings.
            "kind": entry.get("kind", "page"),
            "title": entry.get("title"),
            "divider": entry.get("divider", False),
            "start": entry.get("start"),
            "end": entry.get("end"),
            "from_image": entry.get("from_image", False),
        })
    return pages


# Packing units that are smaller than a chunk
#
# A page of prose is bigger than a chunk, so chunking it means splitting. A
# slide and a spoken sentence are far smaller, so chunking them means the
# opposite: packing them together until they are worth embedding, and breaking
# where the medium says one topic ends and the next begins.

SLIDE_TOPIC_BREAK = 0.5      # a new title ends a chunk once it is half full
AUDIO_PAUSE_SECONDS = 2.0    # a pause this long is where a speaker changes topic
AUDIO_PAUSE_BREAK = 0.5      # ... and it ends a chunk once it is half full


def _tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


# Bullets from symbol fonts (Wingdings and friends) arrive as private-use
# characters such as U+F06C. They carry no meaning for the embedder, for BM25
# or for the reader, so they go before anything else sees them.
_SYMBOL_GLYPH = re.compile(r"[-•●▪■]+")


def _collapse(text: str) -> str:
    return re.sub(r"\s+", " ", _SYMBOL_GLYPH.sub(" ", text)).strip()


def _packed_chunk(units: List[dict], section: Optional[str]) -> Chunk:
    """One chunk out of several consecutive slides."""
    text = " ".join(_collapse(u["text"]) for u in units if u["text"].strip())
    pages = [u["page"] for u in units if u["page"] is not None]
    return Chunk(
        text,
        source_file=units[0]["source_file"],
        page=pages[0] if pages else None,
        section=section,
        kind=units[0].get("kind", "page"),
        page_end=pages[-1] if len(pages) > 1 and pages[-1] != pages[0] else None,
        from_image=any(u.get("from_image") for u in units),
    )


def _oversized(tokenizer, unit: dict, section: Optional[str], chunk_tokens: int,
               overlap: int) -> Iterator[Chunk]:
    """A single slide or segment longer than a whole chunk: split it as prose."""
    for window in _sentence_windows(tokenizer, _collapse(unit["text"]),
                                    chunk_tokens, overlap):
        chunk = _packed_chunk([unit], section)
        yield Chunk(window, source_file=chunk.source_file, page=chunk.page,
                    section=section, kind=unit.get("kind", "page"),
                    start=unit.get("start"), end=unit.get("end"),
                    from_image=chunk.from_image)


def _pack_slides(tokenizer, units: List[dict], chunk_tokens: int,
                 overlap: int) -> List[Chunk]:
    """
    Pack a deck's slides into chunks of at most chunk_tokens.

    Slides are packed in order until the chunk is full, or until a new title
    arrives once the chunk is already half full, so a chunk covers one part of
    the talk rather than an arbitrary run of slides. A divider — a slide
    holding only its title — is never embedded on its own: it becomes the
    section label for the slides that follow it, which is what gives a deck
    with no PDF outline and no numbered headings real topics to quiz on.
    """
    chunks: List[Chunk] = []
    buffer: List[dict] = []
    buffered = 0
    label: Optional[str] = None
    started_with: Optional[str] = None

    def flush():
        nonlocal buffer, buffered
        if buffer:
            chunks.append(_packed_chunk(buffer, label or started_with
                                        or buffer[0]["section"]))
        buffer, buffered = [], 0

    for unit in units:
        if not unit["text"].strip():
            continue
        if unit["divider"]:
            flush()
            label = unit["title"]
            started_with = None
            continue

        size = _tokens(tokenizer, _collapse(unit["text"]))
        if size > chunk_tokens:
            flush()
            section = label or unit["title"] or unit["section"]
            chunks.extend(_oversized(tokenizer, unit, section, chunk_tokens, overlap))
            continue

        new_topic = (unit["title"] and started_with and unit["title"] != started_with
                     and buffered >= chunk_tokens * SLIDE_TOPIC_BREAK)
        if buffer and (buffered + size > chunk_tokens or new_topic):
            flush()
        if not buffer:
            started_with = unit["title"]
        buffer.append(unit)
        buffered += size

    flush()
    return chunks


def _pack_audio(tokenizer, units: List[dict], chunk_tokens: int,
                overlap: int) -> List[Chunk]:
    """
    Pack a transcript's segments into chunks of at most chunk_tokens.

    A recording has no headings to cut at, so the breaks come from the
    speaker: a pause of AUDIO_PAUSE_SECONDS or more ends a chunk once it is
    half full, on the assumption that a lecturer pauses between points. Each
    chunk keeps the time it was spoken, so an answer can cite the moment.
    """
    chunks: List[Chunk] = []
    buffer: List[dict] = []
    buffered = 0
    part = 1

    def flush():
        nonlocal buffer, buffered, part
        if buffer:
            start, end = buffer[0]["start"], buffer[-1]["end"]
            def mmss(t):
                return f"{int(t or 0) // 60}:{int(t or 0) % 60:02d}"
            chunks.append(Chunk(
                " ".join(_collapse(u["text"]) for u in buffer),
                source_file=buffer[0]["source_file"],
                page=None, kind="audio",
                section=f"Part {part} ({mmss(start)}-{mmss(end)})",
                start=start, end=end))
            part += 1
        buffer, buffered = [], 0

    for i, unit in enumerate(units):
        if not unit["text"].strip():
            continue
        size = _tokens(tokenizer, _collapse(unit["text"]))
        if size > chunk_tokens:
            flush()
            section = f"Part {part}"
            chunks.extend(_oversized(tokenizer, unit, section, chunk_tokens, overlap))
            part += 1
            continue
        if buffer and buffered + size > chunk_tokens:
            flush()
        buffer.append(unit)
        buffered += size

        following = units[i + 1] if i + 1 < len(units) else None
        pause = ((following["start"] or 0) - (unit["end"] or 0)) if following else 0
        if pause >= AUDIO_PAUSE_SECONDS and buffered >= chunk_tokens * AUDIO_PAUSE_BREAK:
            flush()

    flush()
    return chunks


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


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _sentence_windows(tokenizer, cleaned: str, chunk_tokens: int,
                      overlap: int) -> Iterator[str]:
    """
    Pack whole sentences into windows of at most chunk_tokens tokens, so no
    chunk starts or ends mid-sentence. The next window repeats the trailing
    sentences of the previous one, up to `overlap` tokens. A sentence longer
    than a whole window falls back to _window().
    """
    sentences = [s for s in _SENTENCE_END.split(cleaned) if s.strip()]
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

    i = 0
    while i < len(sentences):
        if lengths[i] > chunk_tokens:
            yield from _window(tokenizer, sentences[i], chunk_tokens, overlap)
            i += 1
            continue
        j, used = i, 0
        while j < len(sentences) and lengths[j] <= chunk_tokens - used:
            used += lengths[j]
            j += 1
        yield tokenizer.decode(tokenizer.encode(" ".join(sentences[i:j]),
                                                add_special_tokens=False))
        if j >= len(sentences):
            break
        # step back over trailing sentences that fit in the overlap budget
        back, carried = j, 0
        while back - 1 > i and carried + lengths[back - 1] <= overlap:
            back -= 1
            carried += lengths[back]
        i = back


# Heading-aware chunking
#
# A page is cut at every heading line — numbered ("2.1. Data Processing",
# "3 Results", "II. CAUSES"), lettered appendix headings ("A. Evaluation
# Datasets") or named ("Lecture 4 - Overfitting") — and each block is packed
# into sentence windows on its own, so no chunk mixes two subsections. This
# works on raw lines, before whitespace is collapsed, because headings are only
# recognisable as whole lines.

_SUBHEADING = re.compile(
    r"^(?:(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,5}\.|[A-H]\.)\s+[A-Z][A-Za-z]"
    r"|(?i:chapter|lecture|week|unit|module|topic|part)\s+[\dIVXivx]+\b)")
_BARE_SECTION_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?$")


def _section_number_ok(line: str) -> bool:
    """Section numbers start between 1 and 20; table cells like "0.1" or "69" don't."""
    first = re.match(r"^(\d+)", line)
    return first is None or 1 <= int(first.group(1)) <= 20


def _is_heading(line: str) -> bool:
    words = line.split()
    return (0 < len(words) <= 12 and "," not in line
            and not line.rstrip().endswith((".", ";"))
            and _section_number_ok(line)
            and bool(_SUBHEADING.match(line)))


def _heading_blocks(page_text: str) -> List[str]:
    """Raw page text split into blocks, each starting at a heading line."""
    lines = [line.strip() for line in page_text.splitlines()]
    blocks, current = [], []
    for i, line in enumerate(lines):
        # "3." on its own line, with the title on the next line
        starts = _is_heading(line) or (
            _BARE_SECTION_NUMBER.match(line) and i + 1 < len(lines)
            and not re.search(r"\d", lines[i + 1])
            and _is_heading(f"{line} {lines[i + 1]}"))
        if starts and any(current):
            blocks.append("\n".join(current))
            current = []
        current.append(line)
    if any(current):
        blocks.append("\n".join(current))
    return blocks


def _heading_windows(tokenizer, page_text: str, chunk_tokens: int,
                     overlap: int) -> Iterator[str]:
    for block in _heading_blocks(page_text):
        cleaned = re.sub(r"\s+", " ", block).strip()
        if cleaned:
            yield from _sentence_windows(tokenizer, cleaned, chunk_tokens, overlap)


# Semantic chunking
#
# The usual RAG sense of the term: embed every sentence, and start a new chunk
# where two neighbouring sentences are least alike — here, at the least similar
# quarter of neighbouring pairs on the page. Segments under MIN_SEGMENT_TOKENS are
# merged into the one before (or after, for the first), and a segment too long
# for one chunk is packed into sentence windows. Sentences are always embedded
# with all-MiniLM-L6-v2, so the chunks are the same whichever model is used for
# retrieval.

SEMANTIC_BREAK_PERCENTILE = 25
MIN_SEGMENT_TOKENS = 40


def _sentence_vectors(sentences: List[str]):
    from backend.pipeline.embedder import embed
    return embed(sentences, model="minilm")


def _semantic_windows(tokenizer, cleaned: str, chunk_tokens: int,
                      overlap: int) -> Iterator[str]:
    sentences = [s for s in _SENTENCE_END.split(cleaned) if s.strip()]
    if len(sentences) < 3:
        yield from _sentence_windows(tokenizer, cleaned, chunk_tokens, overlap)
        return
    lengths = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

    vectors = _sentence_vectors(sentences)
    sims = [float(vectors[i] @ vectors[i + 1]) for i in range(len(sentences) - 1)]
    # the k least similar neighbour pairs, so ties cannot add extra breaks
    k = max(1, len(sims) * SEMANTIC_BREAK_PERCENTILE // 100)
    breaks = set(sorted(range(len(sims)), key=lambda i: sims[i])[:k])

    segments, current = [], [0]
    for i in range(len(sims)):
        if i in breaks:
            segments.append(current)
            current = []
        current.append(i + 1)
    segments.append(current)

    merged = []
    for seg in segments:
        size = sum(lengths[i] for i in seg)
        if merged and size < MIN_SEGMENT_TOKENS:
            merged[-1] = merged[-1] + seg
        elif merged and sum(lengths[i] for i in merged[-1]) < MIN_SEGMENT_TOKENS:
            merged[-1] = merged[-1] + seg
        else:
            merged.append(seg)

    for seg in merged:
        text = " ".join(sentences[i] for i in seg)
        if sum(lengths[i] for i in seg) <= chunk_tokens:
            yield tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))
        else:
            yield from _sentence_windows(tokenizer, text, chunk_tokens, overlap)


CHUNKING_MODES = ("window", "sentence", "heading", "semantic")


def preprocess(text: Union[str, Sequence[dict], dict], chunk_tokens: int = 220,
               overlap: int = 40, source_file: Optional[str] = None,
               strip_page1_boilerplate: bool = True,
               chunking: str = "window") -> List[Chunk]:
    """
    Clean raw text and split it into overlapping chunks sized to fit the
    embedder's 256-token limit (220-token windows, 40-token overlap by default).

    Accepts either:
      - a plain string — image, audio or plain-text input; pass source_file
        explicitly if you want the chunks tagged with a filename, or
      - the per-page list returned by loader.load_pdf():
        [{"source_file": ..., "page": ..., "text": ...}, ...]

    Pages are chunked independently, so a chunk never spans a page boundary.
    Every returned Chunk carries .source_file, .page and .section (page and
    section are None for plain string input).

    strip_page1_boilerplate removes author/affiliation/ORCID/email lines from
    page 1 only (see _strip_page1_boilerplate). Pass False to reproduce the
    behaviour from before that filter existed.

    chunking selects how each page is split, always within chunk_tokens:
      "window"   fixed token windows with overlap (the default)
      "sentence" whole sentences packed into windows (_sentence_windows)
      "heading"  cut at section/subsection headings, then sentence windows
      "semantic" cut where neighbouring sentences are least similar
    The alternatives are compared in backend/scripts/embedder_comparison.py.
    """
    if chunking not in CHUNKING_MODES:
        raise ValueError(f"chunking must be one of {CHUNKING_MODES}")
    pages = _as_pages(text, source_file)

    for page in pages:
        if not isinstance(page["text"], str):
            raise TypeError(f"Expected string, got {type(page['text'])}")

    # Length check is on the document as a whole, as before — a short page in
    # an otherwise-fine PDF isn't an extraction failure.
    combined = " ".join(page["text"] for page in pages)
    if len(combined.strip()) < 20:
        name = pages[0]["source_file"] if pages else None
        raise ValueError(
            f"No readable text in {name or 'this file'} — a scanned document "
            f"or a slide deck of pictures has no text layer to read.")

    tokenizer = _get_model().tokenizer

    # Slides and spoken segments are smaller than a chunk, so they are packed
    # rather than split, and the chunking mode does not apply to them: there
    # is nothing to cut inside a six-word slide. See _pack_slides/_pack_audio.
    kinds = {page["kind"] for page in pages}
    if kinds == {"slide"}:
        return _pack_slides(tokenizer, pages, chunk_tokens, overlap)
    if kinds == {"audio"}:
        return _pack_audio(tokenizer, pages, chunk_tokens, overlap)

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
        if chunking == "heading":
            # needs the raw lines to see headings
            windows = _heading_windows(tokenizer, page_text, chunk_tokens, overlap)
        else:
            splitter = {"window": _window, "sentence": _sentence_windows,
                        "semantic": _semantic_windows}[chunking]
            windows = splitter(tokenizer, cleaned, chunk_tokens, overlap)
        for window_text in windows:
            chunks.append(Chunk(
                window_text,
                source_file=page["source_file"],
                page=page["page"],
                section=page["section"],
            ))

    return chunks
