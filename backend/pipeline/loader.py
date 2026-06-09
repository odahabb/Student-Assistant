"""
backend/pipeline/loader.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 1 of pipeline: INPUT GATHERING
Handles all input modalities:
  - PDF        → PyMuPDF (fitz)
  - Image      → easyOCR
  - Audio      → openai-whisper (speech-to-text)
  - Plain text → passthrough

Usage:
    from backend.pipeline.loader import load_input

    text = load_input("lecture.pdf",  input_type="pdf")
    text = load_input("slide.png",    input_type="image")
    text = load_input("lecture.mp3",  input_type="audio")
    text = load_input("some string",  input_type="text")
"""

import os
import logging
import warnings

logging.basicConfig(level=logging.INFO, format="[loader] %(message)s")
log = logging.getLogger(__name__)


warnings.filterwarnings('ignore', message=".*pin_memory.*accelerator.*")



# Individual loaders

def load_pdf(path: str) -> str:
    """
    Extract all text from a PDF file page by page using PyMuPDF.
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("Run: pip install PyMuPDF")

    if not os.path.exists(path):
        raise FileNotFoundError(f"PDF not found: {path}")

    log.info(f"Loading PDF → {path}")
    doc = fitz.open(path)
    text_parts = []

    for i, page in enumerate(doc):
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(page_text)
        else:
            log.warning(f"  Page {i+1}/{len(doc)} had no extractable text")

    full_text = "\n\n".join(text_parts)
    log.info(f"PDF loaded — {len(doc)} pages, {len(full_text)} characters")
    doc.close()
    return full_text


def load_image(path: str) -> str:
    """
    Run OCR on an image file (PNG, JPG, TIFF, etc.) using easyOCR.
    Returns extracted text string.
    """
    try:
        import easyocr
    except ImportError:
        raise ImportError("Run: pip install easyocr")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    log.info(f"Running OCR on image → {path}")
    reader = easyocr.Reader(['en'])
    results = reader.readtext(path)
    text = '\n'.join([detection[1] for detection in results])
    log.info(f"OCR complete — {len(text)} characters extracted")
    return text


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

def load_input(source: str, input_type: str) -> str:
    """
    Unified input loader for all modalities.

    Args:
        source      : File path (for pdf/image/audio) or raw string (for text)
        input_type  : One of 'pdf' | 'image' | 'audio' | 'text'

    Returns:
        Extracted text as a single string, ready for preprocessing.

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

def load_file(path: str) -> str:
    """
    Convenience wrapper — detects input type from file extension automatically.

    Example:
        text = load_file("lecture_notes.pdf")   # auto-detected as pdf
        text = load_file("scanned_doc.png")      # auto-detected as image
    """
    _, ext = os.path.splitext(path.lower())
    input_type = EXTENSION_MAP.get(ext)

    if input_type is None:
        raise ValueError(
            f"Cannot auto-detect type for extension '{ext}'. "
            f"Use load_input() and specify input_type manually."
        )

    log.info(f"Auto-detected '{ext}' → input_type='{input_type}'")
    return load_input(path, input_type)


# Quick demo 

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  loader.py — input gathering demo")
    print("="*55)

    # ─ DEMO 1: Plain text (no file needed) ─
    print("\n[1] Text input:")
    sample = "Machine learning (ML) is a subset of artificial intelligence (AI) that teaches computers to learn from data and identify patterns without being explicitly programmed for every specific task. Instead of following hard-coded rules, ML models improve their accuracy over time by analyzing large datasets"
    result = load_input(sample.strip(), input_type="text")
    print(f"    Preview: {result[:60]}...")

    # ─ DEMO 2: Image file (with graceful fallback) ─
    print("\n[2] Image input (OCR):")
    image_path = "data/sample/sample_image.png"
    if os.path.exists(image_path):
        try:
            result = load_input(image_path, input_type="image")
            print(f"    Preview: {result[:60]}...")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print(f"    → File not found: {image_path}")
        print(f"    → Usage: load_input('path/to/image.png', input_type='image')")
        print(f"    → Supported formats: PNG, JPG, TIFF, BMP")

    # ─ DEMO 3: PDF file (with graceful fallback) ─
    print("\n[3] PDF input:")
    pdf_path = "data/sample/sample_document.pdf"
    if os.path.exists(pdf_path):
        try:
            result = load_input(pdf_path, input_type="pdf")
            print(f"    Preview: {result[:60]}...")
        except Exception as e:
            print(f"    Error: {e}")
    else:
        print(f"    → File not found: {pdf_path}")
        print(f"    → Usage: load_input('path/to/document.pdf', input_type='pdf')")

    # ─ DEMO 4: Auto-detect by extension ─
    print("\n[4] Auto-detect by extension:")
    test_files = ["data/sample/sample_document.pdf", "data/sample/sample_image.png", "data/sample/lecture.txt"]
    for fpath in test_files:
        if os.path.exists(fpath):
                result = load_file(fpath)
                print(f"{fpath} → {len(result)} chars"+ "\n")

    print("\n" + "="*55)
    print(" loader.py working correctly")
    print("="*55 + "\n")
