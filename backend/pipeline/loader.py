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
from typing import List, Union

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


# Individual loaders

def load_pdf(path: str) -> List[dict]:
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
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("Run: pip install PyMuPDF")

    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    log.info(f"Loading PDF → {path}")
    doc = fitz.open(path)
    source_file = os.path.basename(path)
    texts = [page.get_text() for page in doc]
    sections = _page_sections(
        doc, [[line.strip() for line in t.splitlines() if line.strip()] for t in texts])
    pages = []

    for i, page_text in enumerate(texts):
        if page_text.strip():
            pages.append({
                "source_file": source_file,
                "page": i + 1,
                "section": sections[i + 1],
                "text": page_text,
            })
        else:
            log.warning(f"  Page {i+1}/{len(doc)} had no extractable text")

    total_chars = sum(len(p["text"]) for p in pages)
    log.info(
        f"PDF loaded — {len(doc)} pages ({len(pages)} with text), "
        f"{total_chars} characters"
    )
    doc.close()
    return pages


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

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    log.info(f"Transcribing audio → {path}  (model: {model_size})")
    model = whisper.load_model(model_size)
    result = model.transcribe(path)
    text = result["text"]
    log.info(f"Transcription complete — {len(text)} characters")
    return text


def load_text(raw: str) -> str:
    """
    Plain text passthrough — validates input and returns as-is.
    """
    if not isinstance(raw, str):
        raise TypeError(f"Expected string, got {type(raw)}")
    if not raw.strip():
        raise ValueError("Input text is empty")
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
        "audio": load_audio,
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

def load_file(path: str) -> Union[str, List[dict]]:
    """
    Convenience wrapper — detects input type from file extension automatically.

    Example:
        pages = load_file("lecture_notes.pdf")   # auto-detected as pdf → per-page list
        text  = load_file("scanned_doc.png")     # auto-detected as image → string
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
    return load_input(path, input_type)