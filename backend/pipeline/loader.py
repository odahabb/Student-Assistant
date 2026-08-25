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


# Lazy-loaded model singletons
#
# Models are expensive to construct (they load weights from disk). They are
# loaded once, on first use, and reused across calls — mirroring the
# _get_model() singleton pattern already used in embedder.py / retriever.py.

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
    Lazy-load Qwen2-VL-2B-Instruct, the primary image extractor.

    Validated against EasyOCR+BLIP on a 25-sample DocVQA eval (see
    notebooks/easyocr_blip_vs_qwen2vl_eval.ipynb): with the retry/fallback
    logic in _is_degenerate_extraction()/_run_qwen_extraction() below, this
    produces cleaner, more complete extractions in most cases.
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
    Detects two known Qwen2-VL failure modes on this task: bare bounding-box-
    style coordinate output (e.g. "(10,7),(984,990)"), and markdown tables
    with no real data filled in (just headers/pipes). Found during eval —
    see notebooks/easyocr_blip_vs_qwen2vl_eval.ipynb.
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
    Extraction-only — Qwen2-VL describes/transcribes the document. Retries on
    degenerate output, falling back to greedy decoding on the final attempt
    (greedy proved more reliable than sampling on images that trigger the
    bbox/empty-table bug — see eval notebook for the comparison).
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
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        log.info("Loading EasyOCR reader (first use)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=get_easyocr_device())
    return _ocr_reader


def _get_blip_model():
    """
    Lazy-load BLIP (Salesforce/blip-image-captioning-base) for CPU inference.
    Used to caption an image's visual content — this is what lets the
    pipeline handle charts, graphs, and diagrams, where OCR text alone is
    either absent or doesn't carry the image's actual meaning.
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
# The quiz and recommendation layer tracks mastery per section of a document,
# so every PDF page is tagged with the section it belongs to. Sources, in order
# of trust:
#   1. the PDF's own outline (bookmarks), top level only;
#   2. headings in the page text — "Lecture 3 - ...", "Chapter 2: ...", or a
#      top-level number ("2. Approach", "II. TRANSFORMER ARCHITECTURE", or a
#      bare "3." line followed by its title). Numbered headings are only
#      accepted as a run counting up from 1, which filters out numbered lines
#      that are not headings (affiliations, list items, figure labels);
#   3. fixed groups of pages.
# Assignment is per page: a page takes the first heading that appears on it,
# otherwise the section carried over from the previous page. Text on a page
# before its first heading is therefore attributed to the new section — an
# approximation that is fine at the granularity mastery is tracked at.

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
    if token.isdigit():
        return int(token)
    total, prev = 0, 0
    for ch in reversed(token.upper()):
        value = _ROMAN[ch]
        total = total - value if value < prev else total + value
        prev = max(prev, value)
    return total


def _tidy_title(title: str) -> str:
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
    # One run per numbering style, so "1. Introduction" is not followed by an
    # unrelated "II. ..." line; the longer run wins.
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
    # A references heading only counts once the body's sections have begun.
    # Numbered lines after it are appendix tables and lists, not the body's
    # section run, so the run stops there.
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
            # a later heading on the same page carries over to the next page
            current = on_page[-1]
    return sections


# Telling slide decks from ordinary documents
#
# A slide deck needs different treatment from a paper: its "pages" hold a
# handful of words, its titles are the only headings it has, and much of what
# it says is in pictures. These helpers label each page so the preprocessor
# can pack slides together instead of embedding them one by one.

SLIDE_MEDIAN_WORDS = 60      # a deck's slides hold far less text than a page
SLIDE_MIN_PAGES = 4          # too few pages to judge — treat as a document
# A divider's text is never embedded, so the rule has to be strict: anything
# beyond the title and a stray page number means the slide says something.
DIVIDER_MAX_EXTRA_WORDS = 2
_BULLET_CHARS = "•●▪■·-*–—"
_BULLET = re.compile(rf"^[\s{re.escape(_BULLET_CHARS)}]+")


def _clean_line(line: str) -> str:
    """A displayed line without its bullet glyph and surrounding space."""
    return re.sub(r"\s+", " ", _BULLET.sub("", line)).strip()


def _document_kind(doc, texts: List[str]) -> str:
    """
    "slides" or "pages". Slides are recognised by how little text they carry,
    and by the landscape shape almost every deck uses; either signal alone is
    enough, because a text-heavy deck is still a deck and a portrait deck is
    still mostly pictures.
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
    A slide's title: its first line, which is how decks mark their topics.
    Returns None for a slide that starts with a bullet or a sentence rather
    than a title.
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
    A slide's title taken from its largest type, joined across the lines it
    wraps onto. Reading the first line instead truncates the common two-line
    title ("1.101: Bio-inspired / computing"), and font size is the signal the
    format itself uses to mark a title.
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

    # A slide set in one size throughout is usually a title or divider slide,
    # so a single size is not a reason to give up; the length check below is
    # what separates a title from a slide of running text.
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
    True for a slide that carries its title and nothing else — a section
    divider. These are never worth embedding alone; the preprocessor uses
    them to label the slides that follow.
    """
    if not title:
        return False
    rest = _clean_line(re.sub(r"\s+", " ", text)).replace(title, "", 1)
    words = [w for w in re.findall(r"[A-Za-z]{2,}", rest)]
    return len(words) <= DIVIDER_MAX_EXTRA_WORDS and len(rest.split()) <= 4


# Reading the pictures inside a PDF
#
# PyMuPDF only reads a page's text layer, so anything drawn as a picture — a
# diagram, a chart, a screenshot of code, or a slide exported as an image — is
# invisible to the rest of the pipeline. These helpers render such a page and
# send it through the same extraction chain as an uploaded image, cheapest
# method first, because the vision model costs about a minute per page.

FIGURE_RENDER_DPI = 150
FIGURE_THIN_WORDS = 12        # below this, a page may be mostly picture
FIGURE_IMAGE_AREA = 0.15      # ... if pictures cover this share of the page
# When OCR comes back with fewer words than this, the page holds a photograph
# or a diagram with no legible labels, and only the vision model can say what
# is on it. Anything more than that — a title slide, a screenshot, a labelled
# chart — is already searchable text, and a minute of vision model per page
# would buy a description the student can mostly read off the slide anyway.
FIGURE_VISION_IF_FEWER = 3
FIGURE_VISION_MIN_AREA = 0.35   # ... and the picture takes up this much of it
FIGURE_VISION_PER_DOC = 8       # hard cap: one upload cannot run for hours
FIGURE_MAX_PAGES = 80           # pages offered to the picture reader at all
FIGURE_CACHE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "cache", "figures")


class _nothing:
    """A do-nothing stand-in for a lock the caller did not supply."""

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
    cached = os.path.join(FIGURE_CACHE, f"{digest}.txt")
    if os.path.exists(cached):
        with open(cached, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _cache_figure_text(digest: str, text: str) -> None:
    try:
        os.makedirs(FIGURE_CACHE, exist_ok=True)
        with open(os.path.join(FIGURE_CACHE, f"{digest}.txt"), "w",
                  encoding="utf-8") as f:
            f.write(text)
    except OSError as e:       # a read-only or full disk must not stop a load
        log.warning(f"  Could not cache figure text ({e})")


def _ocr_page_image(image_path: str) -> str:
    """EasyOCR alone — about 7 s a page, against about a minute for Qwen2-VL."""
    reader = _get_ocr_reader()
    return _reorder_ocr_by_layout(reader.readtext(image_path))


def _read_page_picture(page, page_number: int, allow_vision: bool = True) -> tuple:
    """
    Read the picture content of one page. Returns (text, method), where method
    is "cache", "ocr", "vision" or "" when nothing legible was found.

    Cheapest first: EasyOCR handles slides whose content is a screenshot or a
    labelled diagram, which is most of them, and Qwen2-VL is called only when
    OCR comes back nearly empty and the caller allows it — the photographs and
    unlabelled diagrams that need describing rather than transcribing.
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
          "section": "Lecture 3 - Supervised Learning", "text": "..."}, ...]

    Page boundaries are deliberately preserved instead of being concatenated
    into a single string, so that preprocessor.preprocess() can chunk *within*
    a page (no chunk spanning two pages) and tag every chunk with the file and
    page it came from. Pages with no extractable text are skipped, as before.
    "section" comes from _page_sections() and is what the quiz layer groups
    pages by.

    Every page also carries "kind" ("page" or "slide"), and slides carry
    "title" and "divider" so the preprocessor can pack a deck by its titles
    instead of embedding one slide at a time.

    figures="auto" additionally reads the pictures on pages whose text layer
    is thin or empty (see _read_page_picture), which is how a diagram, a chart
    or a slide exported as an image gets into the index at all. It is off by
    default because it costs seconds to a minute per page, and because every
    result recorded before it existed was measured without it. report(done,
    total, page) is called as those pages are read, for a progress display.

    lock, if given, is held around each picture — one page at a time, never
    the whole document — so that a caller sharing the models can answer a
    question between pages instead of queueing behind the entire file.
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
        # PyMuPDF's message repeats the whole temporary path; the name and the
        # reason are what the student needs.
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
            except Exception as e:      # a damaged page shouldn't lose the file
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
                except Exception as e:  # never let a picture stop the upload
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

    EasyOCR returns detections in the order its detection model found
    them on the page — NOT in top-to-bottom, left-to-right reading
    order. For a single paragraph this rarely matters, but for tables
    and forms it scrambles rows/columns into a meaningless token soup
    (e.g. a "Revenue | 2023 | 450" row gets split apart and interleaved
    with unrelated cells from other rows).

    This groups detections into rows by y-coordinate (within
    y_tolerance pixels, to absorb natural jitter in scan alignment),
    then sorts each row left-to-right by x-coordinate. This is a
    heuristic, not a table parser: it does not detect or label table
    structure, it only restores spatial reading order. See project
    report for discussion of this scope boundary.

    Args:
        results     : Raw output of easyocr.Reader.readtext() — a list of
                      (bounding_box, text, confidence) tuples.
        y_tolerance : Max pixel difference in row y-position for two
                      detections to be considered part of the same row.

    Returns:
        Text reconstructed in top-to-bottom, left-to-right order, with
        one line per detected row.
    """
    if not results:
        return ""

    # Each detection's box is 4 (x, y) corner points; use the average
    # y of the top two corners as that detection's row position.
    items = []
    for box, text, confidence in results:
        top_y = (box[0][1] + box[1][1]) / 2.0
        left_x = box[0][0]
        items.append((top_y, left_x, text))

    items.sort(key=lambda item: item[0])  # rough top-to-bottom pass

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

    # Within each row, sort left-to-right
    lines = []
    for row in rows:
        row_sorted = sorted(row, key=lambda item: item[1])
        lines.append(" ".join(text for _, _, text in row_sorted))

    return "\n".join(lines)


IMAGE_MIN_PIXELS = 32        # smaller than this holds nothing to read


def load_image(path: str) -> str:
    """
    Extract content from an image, primarily using Qwen2-VL-2B-Instruct to
    transcribe visible text and describe any charts/tables/figures in one
    pass. Falls back to EasyOCR + BLIP if Qwen2-VL fails to load or run
    (e.g. model download failure, out-of-memory) — see
    _load_image_easyocr_blip() below.

    Validated against the EasyOCR+BLIP approach on a 25-sample DocVQA eval
    (notebooks/easyocr_blip_vs_qwen2vl_eval.ipynb): Qwen2-VL extraction, with
    the retry/degenerate-output-detection safety net in _run_qwen_extraction(),
    produced higher exact-match/F1 scores on the downstream RAG pipeline.

    Known limitation (documented, not solved): like BLIP, Qwen2-VL's chart
    descriptions are not guaranteed to recover precise numerical values
    (e.g. exact bar heights) — see project report for discussion of this as
    an accepted scope boundary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    # Open it once here so a corrupt or absurdly small file fails with a
    # sentence, rather than inside a model as "Truncated File Read".
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
    Fallback extraction method — EasyOCR (spatially-reordered text) + BLIP
    (semantic caption), combined. Used only when Qwen2-VL is unavailable.
    """
    # --- EasyOCR branch: spatially-reordered text extraction ---
    reader = _get_ocr_reader()
    ocr_results = reader.readtext(path)
    ocr_text = _reorder_ocr_by_layout(ocr_results)
    log.info(f"OCR complete — {len(ocr_text)} characters extracted")

    # --- BLIP branch: semantic caption ---
    from PIL import Image
    processor, model = _get_blip_model()
    raw_image = Image.open(path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to(get_torch_device())
    output_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    log.info(f"BLIP caption: {caption}")

    # --- Combine ---
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
    Transcribe an audio file to text using OpenAI Whisper.
    model_size options: tiny | base | small | medium | large
    Use 'base' for speed during development, 'small' for better accuracy.
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

    A recording has no pages and no headings, so these segments and the pauses
    between them are the only structure it offers. The preprocessor packs them
    into chunks and breaks at the longest pauses, and the timestamps let an
    answer cite the moment it came from.
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
        # Whisper re-raises ffmpeg's entire banner on a bad file; the student
        # needs the one useful sentence, not forty lines of build flags.
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
        # Some builds return no segments; keep the transcript as one piece.
        segments = [{"source_file": source_file, "kind": "audio",
                     "text": result["text"], "start": 0.0, "end": 0.0}]

    spoken = sum(len(s["text"].split()) for s in segments)
    log.info(f"Transcription complete — {len(segments)} segments, {spoken} words")
    return segments


def load_text(raw: str) -> str:
    """
    Plain text passthrough — validates input and returns as-is.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Expected string, got {type(raw)}")
    if not raw.strip():
        raise ValueError("Input text is empty")
    # A file of null bytes or other control characters reads as "text" but
    # holds nothing anyone can search; indexing it only pollutes the subject.
    readable = sum(1 for ch in raw if ch.isprintable() or ch in "\n\r\t")
    if readable < len(raw) * 0.5:
        raise ValueError("This file holds no readable text — it looks binary "
                         "rather than a document")
    log.info(f"Plain text received — {len(raw)} characters")
    return raw


# Unified entry point — this is what the rest of the pipeline calls

SUPPORTED_TYPES = ("pdf", "image", "audio", "text")

def load_input(source: str, input_type: str) -> Union[str, List[dict]]:
    """
    Unified input loader for all modalities.

    Args:
        source      : File path (for pdf/image/audio) or raw string (for text)
        input_type  : One of 'pdf' | 'image' | 'audio' | 'text'

    Returns:
        For 'pdf'  : a list of per-page dicts (see load_pdf) — page boundaries
                     are preserved so chunking can stay inside a page.
        Otherwise  : extracted text as a single string.
        Both shapes are accepted directly by preprocessor.preprocess().

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


# Auto-detect type from file extension 

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
    Convenience wrapper — detects input type from file extension automatically.

    Example:
        pages = load_file("lecture_notes.pdf")   # auto-detected as pdf → per-page list
        text  = load_file("scanned_doc.png")     # auto-detected as image → string

    figures and report apply to PDFs only (see load_pdf) and are ignored for
    the other types, so callers can pass them without checking the extension.
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
        # load_input's "text" type takes the text itself, not a path.
        if not os.path.exists(path):
            raise FileNotFoundError(f"Text file not found: {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return load_text(f.read())
    if input_type == "pdf":
        return load_pdf(path, figures=figures, report=report, lock=lock)
    return load_input(path, input_type)