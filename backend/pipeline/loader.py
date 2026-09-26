"""
backend/pipeline/loader.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 1 of pipeline: INPUT GATHERING
Handles all input modalities:
  - PDF        → PyMuPDF (fitz)
  - Image      → Qwen2-VL-2B-Instruct (transcription + chart/table description),
                 falling back to EasyOCR + BLIP if Qwen2-VL fails to load/run
  - Audio      → openai-whisper (speech-to-text)
  - Plain text → passthrough
"""

import os
import re
import logging
import warnings
from typing import Callable, List, Optional, Union

from backend.pipeline.device import get_torch_device, get_easyocr_device

logging.basicConfig(level=logging.INFO, format="[loader] %(message)s")
log = logging.getLogger(__name__)


warnings.filterwarnings('ignore', message=".*pin_memory.*accelerator.*")


# Lazy-loaded model singletons. Each model is built on first use and reused
# for the rest of the process, as embedder.py and generator.py do.

_qwen_model = None
_qwen_processor = None

_ocr_reader = None
_blip_processor = None
_blip_model = None


QWEN_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

QWEN_EXTRACTION_PROMPT = (
    "Transcribe all visible text in this document image, and describe any "
    "charts, tables, or figures. If there is a trend or relationship in the "
    "data, briefly explain it."
)

QWEN_MAX_EXTRACTION_ATTEMPTS = 3


def _get_qwen_model():
    """
    Load Qwen2-VL-2B-Instruct, the primary image extractor, on first use.

    Weights are float16 on the Arc GPU and float32 on CPU. Returns
    (processor, model).
    """
    global _qwen_model, _qwen_processor
    if _qwen_model is None:
        import torch
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        log.info("Loading Qwen2-VL-2B-Instruct (first use)...")
        device = get_torch_device()
        _qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_ID,
            torch_dtype=torch.float16 if device == "xpu" else torch.float32,
            low_cpu_mem_usage=True,
        )
        _qwen_model.to(device)
        _qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    return _qwen_processor, _qwen_model


def _is_degenerate_extraction(text: str) -> bool:
    """
    True for output that carries no readable content. Three cases are
    recognised: nothing at all, bare bounding-box coordinates
    ("(10,7),(984,990)"), and a markdown table of pipes and headers with
    almost no words in it.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if re.fullmatch(r'[\s()\d,]+', stripped):
        return True
    pipe_count = stripped.count('|')
    word_count = len(re.findall(r'[A-Za-z]{3,}', stripped))
    if pipe_count > 10 and word_count < 8:
        return True
    return False


def _run_qwen_extraction(path: str) -> str:
    """
    Transcribe and describe an image with Qwen2-VL, and return the text.

    Up to QWEN_MAX_EXTRACTION_ATTEMPTS passes: sampled decoding while output
    is degenerate (see _is_degenerate_extraction), then greedy decoding on
    the last attempt. Returns the final attempt's text either way.
    """
    import torch
    from PIL import Image

    processor, model = _get_qwen_model()
    device = get_torch_device()
    raw_image = Image.open(path).convert("RGB")

    messages = [{
        'role': 'user',
        'content': [{'type': 'image'}, {'type': 'text', 'text': QWEN_EXTRACTION_PROMPT}],
    }]
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt], images=[raw_image],
        padding=True, return_tensors='pt',
    ).to(device)

    extraction = ''
    for attempt in range(QWEN_MAX_EXTRACTION_ATTEMPTS):
        do_sample = attempt < QWEN_MAX_EXTRACTION_ATTEMPTS - 1
        with torch.no_grad():
            kwargs = dict(max_new_tokens=256, eos_token_id=processor.tokenizer.eos_token_id)
            if do_sample:
                kwargs.update(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
            else:
                kwargs.update(do_sample=False)
            output_ids = model.generate(**inputs, **kwargs)
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        extraction = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0].strip()
        if not _is_degenerate_extraction(extraction):
            break
        log.warning(f"  Qwen2-VL produced degenerate output on attempt {attempt+1}, retrying...")

    return extraction


def _get_ocr_reader():
    """Load the EasyOCR English reader on first use."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        log.info("Loading EasyOCR reader (first use)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=get_easyocr_device())
    return _ocr_reader


def _get_blip_model():
    """
    Load BLIP (Salesforce/blip-image-captioning-base) on first use.

    BLIP captions what an image shows, which is what the EasyOCR fallback
    path adds to the transcribed text. Returns (processor, model).
    """
    global _blip_processor, _blip_model
    if _blip_model is None:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        log.info("Loading BLIP image-captioning model (first use)...")
        _blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _blip_model.to(get_torch_device())
    return _blip_processor, _blip_model


# Individual loaders

# Section detection
#
# Every PDF page is tagged with the section it belongs to, which is what the
# quiz layer groups its topics by. Three sources are tried in order:
#   1. the PDF's own outline (bookmarks), top level only;
#   2. headings in the page text — "Lecture 3 - ...", "Chapter 2: ...", or a
#      top-level number ("2. Approach", "II. TRANSFORMER ARCHITECTURE", or a
#      bare "3." line followed by its title). Numbered headings count only as
#      a run counting up from 1, so numbered lines that are not headings
#      (affiliations, list items, figure labels) are ignored;
#   3. fixed groups of PAGE_GROUP_SIZE pages.
# Assignment is per page: a page takes the first heading that appears on it,
# and otherwise the section carried over from the previous page. Text above a
# page's first heading is therefore attributed to the new section.

PAGE_GROUP_SIZE = 5
FRONT_MATTER_TITLE = "Overview"

_NAMED_HEADING = re.compile(
    r"^(?:chapter|lecture|week|unit|module|topic|part)\s+[\dIVXivx]+\b.{0,80}$",
    re.IGNORECASE)
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}|[IVX]{1,5})\.?\s+([A-Z][A-Za-z].{0,70})$")
_BARE_NUMBER = re.compile(r"^(\d{1,2})\.?$")
_REFERENCES_HEADING = re.compile(
    r"^(?:references|bibliography|works cited|acknowledge?ments?)$", re.IGNORECASE)
_ROMAN = {"I": 1, "V": 5, "X": 10}


def _numeral_value(token: str) -> int:
    """The value of an arabic or roman numeral token ("7", "IV")."""
    if token.isdigit():
        return int(token)
    total, prev = 0, 0
    for ch in reversed(token.upper()):
        value = _ROMAN[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total


def _tidy_title(title: str) -> str:
    """A heading's text with its spacing normalised and ALL CAPS title-cased."""
    title = " ".join(title.split()).rstrip(".:")
    if title.isupper():
        title = title.title()
    return title


def _heading_candidates(page_lines: List[List[str]]) -> List[tuple]:
    """(page, kind, number, title) for every line that looks like a heading."""
    found = []
    for page_no, lines in enumerate(page_lines, start=1):
        for i, line in enumerate(lines):
            if len(line.split()) > 12:
                continue
            if _NAMED_HEADING.match(line):
                found.append((page_no, "named", None, _tidy_title(line)))
            elif _REFERENCES_HEADING.match(line):
                found.append((page_no, "references", None, "References"))
            elif (m := _NUMBERED_HEADING.match(line)) and not re.match(r"^\d+\.\d", line):
                style = "arabic" if m.group(1).isdigit() else "roman"
                found.append((page_no, f"numbered_{style}", _numeral_value(m.group(1)),
                              _tidy_title(m.group(2))))
            elif (m := _BARE_NUMBER.match(line)) and i + 1 < len(lines):
                title = lines[i + 1]
                if re.match(r"^[A-Z][A-Za-z]", title) and len(title.split()) <= 8:
                    found.append((page_no, "numbered_arabic", int(m.group(1)),
                                  _tidy_title(title)))
    return found


def _sections_from_headings(page_lines: List[List[str]]) -> List[tuple]:
    """Heading starts as (page, title), or [] when no reliable run is found."""
    candidates = _heading_candidates(page_lines)

    named = [(p, t) for p, kind, _, t in candidates if kind == "named"]
    # One run per numbering style, counting up from 1, so "1. Introduction" is
    # never continued by an unrelated "II. ..." line. The longer run wins.
    runs = {"arabic": [], "roman": []}
    for style, run in runs.items():
        expected = 1
        for p, kind, number, title in candidates:
            if kind == f"numbered_{style}" and number == expected:
                run.append((p, title))
                expected += 1
    numbered = max(runs.values(), key=len)

    starts = named if len(named) >= 2 else (numbered if len(numbered) >= 2 else [])
    if not starts:
        return []
    # A references heading counts only once the body's sections have begun,
    # and the run stops there: numbered lines after it belong to appendix
    # tables and lists rather than to the body.
    first_page = starts[0][0]
    references = [p for p, kind, _, _ in candidates
                  if kind == "references" and p >= first_page]
    if references:
        starts = [(p, t) for p, t in starts if p <= references[0]]
        starts.append((references[0], "References"))
    return sorted(starts, key=lambda s: s[0])


def _page_sections(doc, page_lines: List[List[str]]) -> dict:
    """Map every 1-based page number to a section title."""
    n_pages = len(page_lines)
    starts = [(page, title.strip()) for level, title, page in doc.get_toc()
              if level == 1 and page >= 1 and title.strip()]
    if len(starts) < 2:
        starts = _sections_from_headings(page_lines)

    if not starts:
        return {p: f"Pages {p0}–{min(p0 + PAGE_GROUP_SIZE - 1, n_pages)}"
                for p in range(1, n_pages + 1)
                for p0 in [(p - 1) // PAGE_GROUP_SIZE * PAGE_GROUP_SIZE + 1]}

    sections, current = {}, FRONT_MATTER_TITLE
    for page in range(1, n_pages + 1):
        on_page = [title for p, title in starts if p == page]
        sections[page] = on_page[0] if on_page else current
        if on_page:
            # the last heading on a page carries over to the next one
            current = on_page[-1]
    return sections


# Telling slide decks from ordinary documents
#
# These helpers label each PDF page as a slide or an ordinary page, and give
# a slide its title and a flag for section dividers. load_pdf passes all
# three on to the preprocessor, which packs slides instead of splitting them.

SLIDE_MEDIAN_WORDS = 60      # at or below this median, the pages are slides
SLIDE_MIN_PAGES = 4          # fewer pages than this are read as a document
DIVIDER_MAX_EXTRA_WORDS = 2  # words past its title a divider slide may hold
_BULLET_CHARS = "•●▪■·-*–—"
_BULLET = re.compile(rf"^[\s{re.escape(_BULLET_CHARS)}]+")


def _clean_line(line: str) -> str:
    """A line without its bullet glyph or surrounding whitespace."""
    return re.sub(r"\s+", " ", _BULLET.sub("", line)).strip()


def _document_kind(doc, texts: List[str]) -> str:
    """
    "slides" or "pages" for a whole document.

    Either signal is enough on its own: a median page of at most
    SLIDE_MEDIAN_WORDS words, or more landscape pages than portrait ones. A
    document of fewer than SLIDE_MIN_PAGES pages is always "pages".
    """
    if len(texts) < SLIDE_MIN_PAGES:
        return "pages"
    counts = sorted(len(t.split()) for t in texts)
    median_words = counts[len(counts) // 2]
    try:
        landscape = sum(1 for page in doc if page.rect.width > page.rect.height)
        mostly_landscape = landscape > len(texts) / 2
    except Exception:       # a PDF that won't report page sizes
        mostly_landscape = False
    return "slides" if (median_words <= SLIDE_MEDIAN_WORDS or mostly_landscape) else "pages"


def _slide_title(text: str) -> Optional[str]:
    """
    A slide's title taken from the text layer: its first non-empty line.
    None when that line runs past 12 words or ends like a sentence, which
    means the slide opens with prose rather than a title.
    """
    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    title = lines[0]
    if len(title.split()) > 12 or title.endswith((".", ",", ";")):
        return None
    return title


def _title_from_layout(page) -> Optional[str]:
    """
    A slide's title taken from its layout: every span set in the largest font
    size on the page, read top to bottom and left to right, so a title that
    wraps onto two lines comes back whole. None when the page reports no
    spans, or when the result runs past 12 words or ends like a sentence.
    """
    try:
        spans = [(round(span["size"], 1), span["bbox"][1], span["bbox"][0],
                  span["text"])
                 for block in page.get_text("dict").get("blocks", [])
                 if block.get("type") == 0
                 for line in block.get("lines", [])
                 for span in line.get("spans", []) if span.get("text", "").strip()]
    except Exception:
        return None
    if not spans:
        return None

    # A slide set in a single size throughout is still read as a title; the
    # length check below is what separates one from a slide of running text.
    biggest = max(size for size, _, _, _ in spans)
    title = " ".join(text for size, _, _, text in
                     sorted((s for s in spans if s[0] >= biggest - 0.1),
                            key=lambda s: (s[1], s[2])))
    title = _clean_line(title)
    if not title or len(title.split()) > 12 or title.endswith((".", ",", ";")):
        return None
    return title


def _is_divider(text: str, title: Optional[str]) -> bool:
    """
    True for a slide holding its title and at most DIVIDER_MAX_EXTRA_WORDS
    words besides — a section divider. The preprocessor embeds no chunk for
    one and uses its title to label the slides that follow.
    """
    if not title:
        return False
    rest = _clean_line(re.sub(r"\s+", " ", text)).replace(title, "", 1)
    words = [w for w in re.findall(r"[A-Za-z]{2,}", rest)]
    return len(words) <= DIVIDER_MAX_EXTRA_WORDS and len(rest.split()) <= 4


# Reading the pictures inside a PDF
#
# PyMuPDF reads a page's text layer only, so anything drawn as a picture — a
# diagram, a chart, a screenshot of code, or a slide exported as an image —
# reaches the rest of the pipeline as an empty page. These helpers render
# such a page and read it with EasyOCR, then with Qwen2-VL when OCR finds
# almost nothing. Results are cached on disk by the rendered image's hash.

FIGURE_RENDER_DPI = 150
FIGURE_THIN_WORDS = 12        # a page with fewer words than this is a
FIGURE_IMAGE_AREA = 0.15      # ... candidate if pictures cover this much of it
FIGURE_VISION_IF_FEWER = 3    # OCR words below which the vision model runs
FIGURE_VISION_MIN_AREA = 0.35 # ... and the picture must cover this much
FIGURE_VISION_PER_DOC = 8     # vision-model pages allowed per document
FIGURE_MAX_PAGES = 80         # pages offered to the picture reader at all
FIGURE_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cache", "figures")


class _nothing:
    """A context manager that does nothing, used when no lock was given."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _picture_share(page) -> float:
    """Share of the page covered by pictures, 0.0 when it has none."""
    try:
        area = float(page.rect.width * page.rect.height)
        if area <= 0:
            return 0.0
        covered = 0.0
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 1:      # 1 = image block
                x0, y0, x1, y1 = block["bbox"]
                covered += abs((x1 - x0) * (y1 - y0))
        return min(covered / area, 1.0)
    except Exception:
        return 0.0


def _needs_figure_reading(text: str, page) -> bool:
    """A page worth rendering: little or no text, and a picture on it."""
    words = len(text.split())
    if words == 0:
        return True
    return words < FIGURE_THIN_WORDS and _picture_share(page) >= FIGURE_IMAGE_AREA


def _cached_figure_text(digest: str) -> Optional[str]:
    """Picture text cached for this rendered page, or None."""
    cached = os.path.join(FIGURE_CACHE, f"{digest}.txt")
    if os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _cache_figure_text(digest: str, text: str) -> None:
    """Cache a page's picture text under the rendered image's hash."""
    try:
        os.makedirs(FIGURE_CACHE, exist_ok=True)
        with open(os.path.join(FIGURE_CACHE, f"{digest}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(text)
    except OSError as e:       # a read-only or full disk still loads the file
        log.warning(f"  Could not cache figure text ({e})")


def _ocr_page_image(image_path: str) -> str:
    """EasyOCR over a rendered page, put back into reading order."""
    reader = _get_ocr_reader()
    return _reorder_ocr_by_layout(reader.readtext(image_path))


def _read_page_picture(page, page_number: int, allow_vision: bool = True) -> tuple:
    """
    Read the picture content of one page.

    Returns (text, method), where method is "cache", "ocr", "vision", or ""
    when nothing legible was found. The page is rendered at
    FIGURE_RENDER_DPI and looked up in the cache first. EasyOCR runs next,
    and Qwen2-VL only when allow_vision is set and OCR returned fewer than
    FIGURE_VISION_IF_FEWER words; its output is kept only if it is longer.
    Text under three words is discarded. The cache is written either way, so
    a page that yielded nothing is not read twice.
    """
    import hashlib
    import tempfile

    pixmap = page.get_pixmap(dpi=FIGURE_RENDER_DPI)
    data = pixmap.tobytes("png")
    digest = hashlib.sha1(data).hexdigest()
    cached = _cached_figure_text(digest)
    if cached is not None:
        log.info(f"  Page {page_number}: picture text from cache")
        return cached, "cache"

    handle, image_path = tempfile.mkstemp(suffix=".png")
    os.close(handle)
    try:
        with open(image_path, "wb") as f:
            f.write(data)

        text, method = "", ""
        try:
            text = _ocr_page_image(image_path).strip()
            method = "ocr"
        except Exception as e:
            log.warning(f"  Page {page_number}: OCR failed ({e})")

        if allow_vision and len(text.split()) < FIGURE_VISION_IF_FEWER:
            try:
                described = _run_qwen_extraction(image_path).strip()
                if len(described.split()) > len(text.split()):
                    text, method = described, "vision"
            except Exception as e:
                log.warning(f"  Page {page_number}: vision model failed ({e})")

        text = "" if len(text.split()) < 3 else text
        _cache_figure_text(digest, text)
        log.info(f"  Page {page_number}: read {len(text.split())} words from "
                 f"its picture ({method or 'nothing found'})")
        return text, method if text else ""
    finally:
        try:
            os.remove(image_path)
        except OSError:
            pass


# Individual loaders

def load_pdf(path: str, figures: str = "off", report=None, lock=None) -> List[dict]:
    """
    Extract text from a PDF file page by page using PyMuPDF.

    Returns one dict per page that had extractable text:

        [{"source_file": "lecture_notes.pdf", "page": 1,
          "section": "Lecture 3 - Supervised Learning", "text": "...",
          "kind": "page", "title": None, "divider": False,
          "from_image": False}, ...]

    Page boundaries are kept rather than concatenated, so
    preprocessor.preprocess() chunks within a page and tags every chunk with
    the file and page it came from. Pages with no extractable text are
    skipped. "section" comes from _page_sections(); "kind" is "page" or
    "slide" for the whole document (see _document_kind), and a slide also
    carries its "title" and whether it is a "divider".

    figures="auto" also reads the pictures on pages whose text layer is thin
    or empty (see _read_page_picture) and appends what it finds to the page
    text, marking the page from_image. It is off by default because it costs
    seconds to a minute per page. report(done, total, page) is called as
    those pages are read, for a progress display.

    lock, if given, is held around each picture — one page at a time, not the
    whole document — so a caller sharing the models can run something else
    between pages.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("Run: pip install PyMuPDF")

    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{os.path.basename(path)} is empty (0 bytes)")

    log.info(f"Loading PDF → {path}")
    try:
        doc = fitz.open(path)
    except Exception:
        # PyMuPDF's own message repeats the whole temporary path; this reports
        # the file name and the reason instead.
        raise ValueError(f"{os.path.basename(path)} could not be opened as a "
                         f"PDF — it may be corrupt, or not a PDF at all")

    try:
        if getattr(doc, "needs_pass", False) and not doc.authenticate(""):
            raise ValueError(
                f"{os.path.basename(path)} is password-protected, so its text "
                f"cannot be read")

        source_file = os.path.basename(path)
        texts = []
        for page in doc:
            try:
                texts.append(page.get_text())
            except Exception as e:      # a damaged page does not lose the file
                log.warning(f"  Page {len(texts) + 1} could not be read ({e})")
                texts.append("")
        if not texts:
            raise ValueError(f"{source_file} has no pages")

        kind = "slide" if _document_kind(doc, texts) == "slides" else "page"
        sections = _page_sections(
            doc, [[line.strip() for line in t.splitlines() if line.strip()]
                  for t in texts])

        picture_text = {}
        if figures == "auto":
            candidates = [i for i, t in enumerate(texts)
                          if _needs_figure_reading(t, doc[i])][:FIGURE_MAX_PAGES]
            if candidates:
                log.info(f"  Reading pictures on {len(candidates)} page(s)")
            vision_left = FIGURE_VISION_PER_DOC
            for done, i in enumerate(candidates):
                if report:
                    report(done, len(candidates), i + 1)
                may_describe = (vision_left > 0
                                and _picture_share(doc[i]) >= FIGURE_VISION_MIN_AREA)
                try:
                    with (lock or _nothing()):
                        text, method = _read_page_picture(doc[i], i + 1,
                                                          allow_vision=may_describe)
                    if method == "vision":
                        vision_left -= 1
                except Exception as e:  # a failed picture leaves the page as is
                    log.warning(f"  Page {i+1}: reading its picture failed ({e})")
                    text = ""
                if text:
                    picture_text[i] = text
            if report and candidates:
                report(len(candidates), len(candidates), None)

        pages = []
        for i, page_text in enumerate(texts):
            from_picture = picture_text.get(i, "")
            if from_picture:
                page_text = (page_text.rstrip() + "\n" + from_picture
                             if page_text.strip() else from_picture)
            if not page_text.strip():
                log.warning(f"  Page {i+1}/{len(texts)} had no extractable text")
                continue
            title = None
            if kind == "slide":
                title = _title_from_layout(doc[i]) or _slide_title(page_text)
            pages.append({
                "source_file": source_file,
                "page": i + 1,
                "section": sections[i + 1],
                "text": page_text,
                "kind": kind,
                "title": title,
                "divider": _is_divider(page_text, title) if kind == "slide" else False,
                "from_image": bool(from_picture),
            })

        total_chars = sum(len(p["text"]) for p in pages)
        log.info(
            f"PDF loaded — {len(texts)} {kind}s ({len(pages)} with text"
            + (f", {len(picture_text)} read from pictures" if picture_text else "")
            + f"), {total_chars} characters"
        )
        return pages
    finally:
        doc.close()


def _reorder_ocr_by_layout(results, y_tolerance: int = 15) -> str:
    """
    Reconstruct reading order from EasyOCR's raw detections.

    EasyOCR returns detections in the order its detection model found them,
    not in reading order, which interleaves the cells of a table or a form.
    This groups detections into rows by their y-coordinate and sorts each row
    by x-coordinate. It restores spatial reading order only: no table
    structure is detected or labelled.

    Args:
        results     : output of easyocr.Reader.readtext() — a list of
                      (bounding_box, text, confidence) tuples.
        y_tolerance : largest pixel difference in y-position for two
                      detections to count as the same row.

    Returns:
        Text in top-to-bottom, left-to-right order, one line per row.
    """
    if not results:
        return ""

    # A detection's box is four (x, y) corners: its row position is the mean
    # y of the top two, and its column position the x of the top-left one.
    items = []
    for box, text, confidence in results:
        top_y = (box[0][1] + box[1][1]) / 2.0
        left_x = box[0][0]
        items.append((top_y, left_x, text))

    items.sort(key=lambda item: item[0])  # top to bottom, before grouping

    rows = []
    current_row = [items[0]]
    current_row_y = items[0][0]

    for top_y, left_x, text in items[1:]:
        if abs(top_y - current_row_y) <= y_tolerance:
            current_row.append((top_y, left_x, text))
        else:
            rows.append(current_row)
            current_row = [(top_y, left_x, text)]
            current_row_y = top_y
    rows.append(current_row)

    # Then left to right within each row
    lines = []
    for row in rows:
        row_sorted = sorted(row, key=lambda item: item[1])
        lines.append(" ".join(text for _, _, text in row_sorted))

    return "\n".join(lines)


IMAGE_MIN_PIXELS = 32        # images narrower or shorter than this are refused


def load_image(path: str) -> str:
    """
    Extract the content of an image as text.

    Qwen2-VL-2B-Instruct transcribes the visible text and describes any
    charts, tables and figures in one pass. If it fails to load or run, or
    returns nothing usable, EasyOCR + BLIP answer instead (see
    _load_image_easyocr_blip). The file is opened and checked first, so a
    corrupt or tiny image raises before any model is loaded.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Checked here, so a corrupt or tiny file raises a readable error rather
    # than failing inside a model as "Truncated File Read".
    name = os.path.basename(path)
    try:
        from PIL import Image
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except Exception as e:
        raise ValueError(f"{name} could not be opened as an image "
                         f"(corrupt or truncated file: {e})")
    if width < IMAGE_MIN_PIXELS or height < IMAGE_MIN_PIXELS:
        raise ValueError(f"{name} is only {width}x{height} pixels — too small "
                         f"to hold anything to read")

    log.info(f"Processing image → {path}")

    try:
        extraction = _run_qwen_extraction(path)
        if not extraction.strip():
            log.warning("  Qwen2-VL produced no usable content, falling back to EasyOCR+BLIP")
            return _load_image_easyocr_blip(path)
        return extraction
    except Exception as e:
        log.warning(f"  Qwen2-VL extraction failed ({e}), falling back to EasyOCR+BLIP")
        return _load_image_easyocr_blip(path)


def _load_image_easyocr_blip(path: str) -> str:
    """
    The fallback extractor: EasyOCR's text, in reading order, followed by
    BLIP's caption of what the image shows. Used when Qwen2-VL cannot run.
    """
    # Text, put back into reading order
    reader = _get_ocr_reader()
    ocr_results = reader.readtext(path)
    ocr_text = _reorder_ocr_by_layout(ocr_results)
    log.info(f"OCR complete — {len(ocr_text)} characters extracted")

    # A caption of what the image shows
    from PIL import Image
    processor, model = _get_blip_model()
    raw_image = Image.open(path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(get_torch_device())
    output_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    log.info(f"BLIP caption: {caption}")

    # Both parts, labelled, as one passage
    parts = []
    if ocr_text.strip():
        parts.append(f"Extracted text: {ocr_text}")
    if caption.strip():
        parts.append(f"Image description: {caption}")

    if not parts:
        log.warning(f"  Neither OCR nor BLIP produced usable content for {path}")

    return "\n\n".join(parts)


def load_audio(path: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file into a single string with OpenAI Whisper.

    model_size is one of tiny | base | small | medium | large. The segments
    behind the transcript are available from load_audio_segments.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Run: pip install openai-whisper")

    return " ".join(s["text"].strip()
                    for s in load_audio_segments(path, model_size)).strip()


def load_audio_segments(path: str, model_size: str = "base") -> List[dict]:
    """
    Transcribe an audio file and keep Whisper's own segmentation:

        [{"source_file": "lecture.m4a", "kind": "audio", "text": "...",
          "start": 0.0, "end": 7.4}, ...]

    The preprocessor packs these segments into chunks and breaks at the
    pauses between them (see preprocessor._pack_audio); the timestamps are
    what lets an answer cite the moment it came from. A Whisper build that
    returns no segments falls back to one entry holding the whole transcript.
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("Run: pip install openai-whisper")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    if os.path.getsize(path) == 0:
        raise ValueError(f"{os.path.basename(path)} is empty (0 bytes)")

    log.info(f"Transcribing audio → {path}  (model: {model_size})")
    model = whisper.load_model(model_size)
    try:
        result = model.transcribe(path)
    except Exception as e:
        # Whisper re-raises ffmpeg's whole banner on a bad file; only its last
        # line is kept.
        reason = str(e).strip().splitlines()[-1][:120] if str(e).strip() else ""
        raise ValueError(
            f"{os.path.basename(path)} could not be transcribed — it may not "
            f"contain a readable audio track"
            + (f" ({reason})" if reason else "") + ".")

    source_file = os.path.basename(path)
    segments = [
        {"source_file": source_file, "kind": "audio", "text": s["text"],
         "start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0))}
        for s in result.get("segments", []) if s.get("text", "").strip()
    ]
    if not segments and result.get("text", "").strip():
        # Some Whisper builds return no segments: keep the transcript whole.
        segments = [{"source_file": source_file, "kind": "audio",
                     "text": result["text"], "start": 0.0, "end": 0.0}]

    spoken = sum(len(s["text"].split()) for s in segments)
    log.info(f"Transcription complete — {len(segments)} segments, {spoken} words")
    return segments


def load_text(raw: str) -> str:
    """
    Plain text passthrough: check the input and return it unchanged.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Expected string, got {type(raw)}")
    if not raw.strip():
        raise ValueError("Input text is empty")
    # A file of null bytes or other control characters decodes as "text" but
    # holds nothing to search, so it is refused rather than indexed.
    readable = sum(1 for ch in raw if ch.isprintable() or ch in "\n\r\t")
    if readable < len(raw) * 0.5:
        raise ValueError("This file holds no readable text — it looks binary "
                         "rather than a document")
    log.info(f"Plain text received — {len(raw)} characters")
    return raw


# Unified entry point, called by the rest of the pipeline

SUPPORTED_TYPES = ("pdf", "image", "audio", "text")

def load_input(source: str, input_type: str) -> Union[str, List[dict]]:
    """
    Unified input loader for all modalities.

    Args:
        source      : File path (for pdf/image/audio) or raw string (for text)
        input_type  : One of 'pdf' | 'image' | 'audio' | 'text'

    Returns:
        For 'pdf'   : a list of per-page dicts (see load_pdf).
        For 'audio' : a list of per-segment dicts (see load_audio_segments).
        Otherwise   : the extracted text as a single string.
        All three shapes are accepted by preprocessor.preprocess().

    Raises:
        ValueError        : If input_type is not supported
        FileNotFoundError : If the file path does not exist
        ImportError       : If a required library is not installed
    """
    if input_type not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported input_type '{input_type}'. "
            f"Choose from: {SUPPORTED_TYPES}"
        )

    loaders = {
        "pdf"  : load_pdf,
        "image": load_image,
        "audio": load_audio_segments,
        "text" : load_text,
    }

    return loaders[input_type](source)


# Input type by file extension

EXTENSION_MAP = {
    ".pdf"  : "pdf",
    ".png"  : "image",
    ".jpg"  : "image",
    ".jpeg" : "image",
    ".tiff" : "image",
    ".bmp"  : "image",
    ".mp3"  : "audio",
    ".mp4"  : "audio",
    ".wav"  : "audio",
    ".m4a"  : "audio",
    ".txt"  : "text",
}

def load_file(path: str, figures: str = "off",
              report: Optional[Callable] = None, lock=None) -> Union[str, List[dict]]:
    """
    load_input with the input type taken from the file extension.

    Example:
        pages = load_file("lecture_notes.pdf")   # pdf   -> per-page list
        text  = load_file("scanned_doc.png")     # image -> string

    figures, report and lock apply to PDFs only (see load_pdf) and are
    ignored for the other types, so a caller need not check the extension.
    """
    _, ext = os.path.splitext(path.lower())
    input_type = EXTENSION_MAP.get(ext)

    if input_type is None:
        raise ValueError(
            f"Cannot auto-detect type for extension '{ext}'. "
            f"Use load_input() and specify input_type manually."
        )

    log.info(f"Auto-detected '{ext}' → input_type='{input_type}'")
    if input_type == "text":
        # load_input's "text" type takes the text itself rather than a path.
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return load_text(f.read())
    if input_type == "pdf":
        return load_pdf(path, figures=figures, report=report, lock=lock)
    return load_input(path, input_type)