"""
backend/pipeline/chunk.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

The Chunk type shared by the preprocessing and storage stages.

It lives in its own module so vector_store.py can serialise/deserialise chunks
without importing preprocessor.py, which would drag the embedding model's
dependencies into a module that otherwise only needs faiss and numpy.
"""

from typing import Optional


class Chunk(str):
    """
    A text chunk together with the source it came from.

    Deliberately a `str` subclass: every downstream stage (embedder,
    vector_store, retriever, generator) consumes chunks as plain strings, so
    making Chunk a str lets the metadata ride along without any of those
    modules having to change. `chunk.text` is the same value as `str(chunk)`.

    source_file : filename the chunk came from (e.g. "lecture_notes.pdf"),
                  or None when the caller didn't supply one.
    page        : 1-based page number for PDF input; None for images, audio
                  and plain text, where page numbers don't apply.

    Note: string operations (.strip(), slicing, re.sub, ...) return a plain
    str, not a Chunk — the metadata does not propagate through them.
    """

    def __new__(cls, text: str, source_file: Optional[str] = None,
                page: Optional[int] = None) -> "Chunk":
        obj = super().__new__(cls, text)
        obj.source_file = source_file
        obj.page = page
        return obj

    @property
    def text(self) -> str:
        """The chunk's text. Same value as str(chunk); provided for callers
        that prefer to be explicit about which part of the chunk they want."""
        return str(self)

    def to_record(self) -> dict:
        """JSON-serialisable form, used by vector_store to persist chunks."""
        return {"text": str(self), "source_file": self.source_file, "page": self.page}

    @classmethod
    def from_record(cls, record) -> "Chunk":
        """
        Rebuild a Chunk from its persisted form.

        Accepts a bare string as well, so chunk files written before chunks
        carried metadata still load — they simply come back with source_file
        and page set to None.
        """
        if isinstance(record, str):
            return cls(record)
        return cls(
            record["text"],
            source_file=record.get("source_file"),
            page=record.get("page"),
        )

    def __repr__(self) -> str:
        return (f"Chunk(source_file={self.source_file!r}, page={self.page!r}, "
                f"text={str(self)!r})")
