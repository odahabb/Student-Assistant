"""
Fakes and fixture builders shared by the tests: stand-ins for the tokenizer
and the embedding model, and helpers that write small PDFs. Nothing here
loads a model.
"""

import os
import tempfile


class WhitespaceTokenizer:
    """Stands in for the MiniLM tokenizer: one token per word."""

    def __init__(self):
        self.vocab, self.words = {}, []

    def encode(self, text, add_special_tokens=False):
        ids = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.words)
                self.words.append(word)
            ids.append(self.vocab[word])
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.words[i] for i in ids)


class FakeEmbeddingModel:
    def __init__(self):
        self.tokenizer = WhitespaceTokenizer()


def make_deck(slides):
    """
    Write a landscape slide deck and return its path. Each slide is
    (title, [body lines]); the title is set in larger type, as a real deck
    does, and a slide with no body lines is a section divider.
    """
    import fitz

    doc = fitz.open()
    for title, body in slides:
        page = doc.new_page(width=720, height=405)     # 16:9, landscape
        page.insert_text((40, 60), title, fontsize=30)
        y = 140
        for line in body:
            page.insert_text((60, y), line, fontsize=14)
            y += 26
    handle, path = tempfile.mkstemp(suffix=".pdf")
    os.close(handle)
    doc.save(path)
    doc.close()
    return path


def audio_segments(*spans, source_file="lecture.m4a"):
    """Whisper-shaped segments: (text, start, end) each."""
    return [{"source_file": source_file, "kind": "audio", "text": text,
             "start": start, "end": end} for text, start, end in spans]


def make_pdf(pages, toc=None):
    """
    Write a PDF with one text page per entry in `pages` and return its path.
    Each entry is a list of lines. `toc` is an optional PyMuPDF outline.
    """
    import fitz

    doc = fitz.open()
    for lines in pages:
        page = doc.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontsize=10)
            y += 14
    if toc:
        doc.set_toc(toc)
    handle, path = tempfile.mkstemp(suffix=".pdf")
    os.close(handle)
    doc.save(path)
    doc.close()
    return path
