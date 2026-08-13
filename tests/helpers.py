"""
Shared test helpers. Nothing here loads a model.
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
