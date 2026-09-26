"""
backend/pipeline/preprocessor.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 2 of pipeline: PREPROCESSING
Cleans raw text and turns it into the chunks that get embedded.

What a chunk is depends on the medium. A page of prose is larger than a chunk
and is split (preprocess); a slide or a spoken segment is smaller and is
packed together with its neighbours (_pack_slides, _pack_audio). Chunking is
page-bounded either way: no chunk spans a page boundary, and every chunk
records the file, page and section it came from (see Chunk).
"""

import logging
import re
from typing import Iterator, List, Optional, Sequence, Union

from backend.pipeline.chunk import Chunk
from backend.pipeline.embedder import _get_model

log = logging.getLogger(__name__)

# Chunk is re-exported, so it can be imported from this module as well as
# from backend.pipeline.chunk.
__all__ = ["Chunk", "preprocess"]


# Page-1 boilerplate stripping
#
# A paper's first page mixes its abstract with author names, affiliations,
# ORCID URLs and emails, all of which end up in the same token windows as the
# abstract. These rules drop those lines from page 1 only.
#
# Every rule needs a positive signal of boilerplate before it fires, and the
# whole filter stands down when it would remove more than
# _MAX_REMOVAL_FRACTION of the page. Prose and titles carry lower-case
# function words ("via", "of", "and") where author and affiliation lines do
# not, which is what most of the rules below test.

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
    One entry in a block of author names: up to four tokens, no lower-case
    words, and two or three of them alphabetic ("Shayne Longpre*",
    "Ed H. Chi"). _strip_page1_boilerplate drops these only in runs of three
    or more, so a lone heading such as "Related Work" survives.
    """
    stripped = _AUTHOR_MARKER_RE.sub('', line).strip()
    if not stripped or _LOWERCASE_WORD_RE.search(stripped):
        return False
    tokens = stripped.split()
    if not 1 <= len(tokens) <= 4:
        return False
    alpha = [t for t in tokens if any(ch.isalpha() for ch in t)]
    # At most three alphabetic tokens ("Ed H. Chi", "Shixiang Shane Gu"), so
    # a four-word title-case title such as "Scaling Instruction-Finetuned
    # Language Models" is not read as part of the author run below it.
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
    # Affiliations are mostly proper nouns, prose mostly lower-case words.
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
    Remove author, affiliation, ORCID and email lines from a first page.

    The document title — the first non-blank line — is always kept, and the
    text comes back unchanged when the rules would remove more than
    _MAX_REMOVAL_FRACTION of the page's non-blank lines.
    """
    lines = text.splitlines()
    drop = [_is_boilerplate_line(line) for line in lines]

    # The first non-blank line of page 1 is the document title, and is kept
    # whichever rules match it.
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
    Normalise preprocess()'s input into one list of page dicts, so the rest
    of the module has a single shape to work with.

    A plain string (image or plain-text input) becomes a single page-less
    entry. The per-page list from loader.load_pdf and the per-segment list
    from loader.load_audio_segments pass through with their metadata, each
    missing key filled in with its default.
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
            # "slide" from loader.load_pdf, "audio" from
            # load_audio_segments, "page" for everything else.
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
# Slides and spoken segments are packed together until a chunk is full, and
# broken where the medium marks a change of topic: a new slide title, or a
# pause in the recording.

SLIDE_TOPIC_BREAK = 0.5      # share of a chunk past which a new title breaks it
AUDIO_PAUSE_SECONDS = 2.0    # a gap this long counts as a pause
AUDIO_PAUSE_BREAK = 0.5      # share of a chunk past which a pause breaks it


def _tokens(tokenizer, text: str) -> int:
    """How many tokens a piece of text costs in a chunk."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# Bullet glyphs, including the private-use characters (U+F06C and friends)
# that symbol fonts such as Wingdings produce. They are replaced by a space
# before the text is embedded or indexed.
_SYMBOL_GLYPH = re.compile(r"[-•●▪■]+")


def _collapse(text: str) -> str:
    """Text with its bullet glyphs and repeated whitespace removed."""
    return re.sub(r"\s+", " ", _SYMBOL_GLYPH.sub(" ", text)).strip()


def _packed_chunk(units: List[dict], section: Optional[str]) -> Chunk:
    """One chunk built from consecutive slides, covering their page range."""
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
    """Split a slide or segment that is longer than a chunk, as prose is split."""
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

    Slides are added in order until the chunk is full, or until a slide with
    a new title arrives while the chunk is at least SLIDE_TOPIC_BREAK full. A
    divider slide (one holding only its title) produces no chunk of its own:
    it ends the current chunk and its title becomes the section label for
    every chunk after it, until the next divider. A slide longer than a whole
    chunk is split by _oversized.
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

    Segments are added in order until the chunk is full, and a gap of
    AUDIO_PAUSE_SECONDS or more between two segments ends the chunk once it
    is at least AUDIO_PAUSE_BREAK full. Each chunk carries the seconds it
    spans and is sectioned "Part n (mm:ss-mm:ss)", so an answer can cite the
    moment it came from.
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
    Split one page's cleaned text into token windows of chunk_tokens, each
    starting `overlap` tokens before the previous one ended.
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
    Pack whole sentences into windows of at most chunk_tokens, so no chunk
    starts or ends mid-sentence. Each window repeats the trailing sentences
    of the one before it, up to `overlap` tokens. A single sentence longer
    than a window is split by _window instead.
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
        # step back over the trailing sentences that fit in the overlap
        back, carried = j, 0
        while back - 1 > i and carried + lengths[back - 1] <= overlap:
            back -= 1
            carried += lengths[back]
        i = back


# Heading-aware chunking
#
# The page is cut at every heading line — numbered ("2.1. Data Processing",
# "3 Results", "II. CAUSES"), lettered ("A. Evaluation Datasets") or named
# ("Lecture 4 - Overfitting") — and each block is packed into sentence
# windows on its own, so no chunk mixes two subsections. It runs on the raw
# lines, before whitespace is collapsed, since a heading is only recognisable
# as a whole line.

_SUBHEADING = re.compile(
    r"^(?:(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,5}\.|[A-H]\.)\s+[A-Z][A-Za-z]"
    r"|(?i:chapter|lecture|week|unit|module|topic|part)\s+[\dIVXivx]+\b)")
_BARE_SECTION_NUMBER = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,3}\.?$")


def _section_number_ok(line: str) -> bool:
    """True unless the line opens with a number outside the 1-20 a section
    number uses, which is what separates a heading from a table cell."""
    first = re.match(r"^(\d+)", line)
    return first is None or 1 <= int(first.group(1)) <= 20


def _is_heading(line: str) -> bool:
    """Whether a whole line reads as a section or subsection heading."""
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
        # A heading line, or a bare "3." with its title on the line below
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
    """Sentence windows over each heading block of a page in turn."""
    for block in _heading_blocks(page_text):
        cleaned = re.sub(r"\s+", " ", block).strip()
        if cleaned:
            yield from _sentence_windows(tokenizer, cleaned, chunk_tokens, overlap)


# Semantic chunking
#
# Every sentence on the page is embedded, and a new chunk starts where two
# neighbouring sentences are least alike: at the SEMANTIC_BREAK_PERCENTILE
# least similar neighbouring pairs. A segment under MIN_SEGMENT_TOKENS is
# merged into the one before it (or after it, for the first), and a segment
# too long for one chunk is packed into sentence windows. Sentences are
# always embedded with all-MiniLM-L6-v2, so the chunk boundaries do not
# depend on the model used for retrieval.

SEMANTIC_BREAK_PERCENTILE = 25
MIN_SEGMENT_TOKENS = 40


def _sentence_vectors(sentences: List[str]):
    """Unit-norm MiniLM vectors, one per sentence."""
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
    # exactly k break points, so a tie cannot add extra ones
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
    Clean raw text and turn it into chunks sized to fit the embedder's
    256-token limit (220-token windows with 40 tokens of overlap by default).

    Accepts either:
      - a plain string — image or plain-text input; pass source_file to have
        the chunks tagged with a filename, or
      - a list of page dicts, as loader.load_pdf and
        loader.load_audio_segments return.

    Pages are chunked independently, so a chunk never spans a page boundary.
    Every returned Chunk carries .source_file, .page and .section, which are
    None for plain string input.

    strip_page1_boilerplate removes author, affiliation, ORCID and email
    lines from page 1 only (see _strip_page1_boilerplate).

    chunking selects how a page of prose is split, always within chunk_tokens:
      "window"   fixed token windows with overlap (the default)
      "sentence" whole sentences packed into windows (_sentence_windows)
      "heading"  cut at section and subsection headings, then sentence windows
      "semantic" cut where neighbouring sentences are least similar
    A deck of slides and a transcript are packed rather than split, and the
    mode does not apply to them.
    """
    if chunking not in CHUNKING_MODES:
        raise ValueError(f"chunking must be one of {CHUNKING_MODES}")
    pages = _as_pages(text, source_file)

    for page in pages:
        if not isinstance(page["text"], str):
            raise TypeError(f"Expected string, got {type(page['text'])}")

    # The length check is on the document as a whole: one short page in an
    # otherwise readable PDF is not an extraction failure.
    combined = " ".join(page["text"] for page in pages)
    if len(combined.strip()) < 20:
        name = pages[0]["source_file"] if pages else None
        raise ValueError(
            f"No readable text in {name or 'this file'} — a scanned document "
            f"or a slide deck of pictures has no text layer to read.")

    tokenizer = _get_model().tokenizer

    # Slides and spoken segments are smaller than a chunk, so they are packed
    # rather than split and the chunking mode does not apply to them.
    kinds = {page["kind"] for page in pages}
    if kinds == {"slide"}:
        return _pack_slides(tokenizer, pages, chunk_tokens, overlap)
    if kinds == {"audio"}:
        return _pack_audio(tokenizer, pages, chunk_tokens, overlap)

    chunks: List[Chunk] = []
    for page in pages:
        page_text = page["text"]
        # Page 1 only: later pages carry no title-block boilerplate.
        if strip_page1_boilerplate and page["page"] == 1:
            page_text = _strip_page1_boilerplate(page_text, page["source_file"])

        # Collapse runs of whitespace
        cleaned = re.sub(r'\s+', ' ', page_text).strip()
        if not cleaned:
            continue
        if chunking == "heading":
            # reads the raw lines, since a heading is a whole line
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
