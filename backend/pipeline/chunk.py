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
    section     : title of the document section the chunk belongs to (see
                  loader._page_sections); None when the input had no sections.
                  The quiz layer groups chunks into topics by it.
    page_end    : last page/slide when a chunk covers several of them (slide
                  decks pack consecutive slides together); None otherwise, and
                  then the chunk covers `page` alone.
    start, end  : position in seconds within an audio recording; None for
                  every other kind of input.
    from_image  : True when some of the text was read out of a picture by the
                  vision model rather than from a text layer, so the interface
                  can say so and the reader can treat it with more caution.

    Note: string operations (.strip(), slicing, re.sub, ...) return a plain
    str, not a Chunk — the metadata does not propagate through them.
    """

    def __new__(cls, text: str, source_file: Optional[str] = None,
                page: Optional[int] = None,
                section: Optional[str] = None,
                page_end: Optional[int] = None,
                start: Optional[float] = None,
                end: Optional[float] = None,
                from_image: bool = False) -> "Chunk":
        obj = super().__new__(cls, text)
        obj.source_file = source_file
        obj.page = page
        obj.section = section
        obj.page_end = page_end
        obj.start = start
        obj.end = end
        obj.from_image = from_image
        return obj

    @property
    def pages(self) -> Optional[str]:
        """"3" or "3-7" — the page or slide range, for citing the chunk."""
        if self.page is None:
            return None
        if self.page_end is None or self.page_end == self.page:
            return str(self.page)
        return f"{self.page}-{self.page_end}"

    @property
    def timecode(self) -> Optional[str]:
        """"12:03-15:40" for audio chunks, None for everything else."""
        if self.start is None:
            return None
        def mmss(t):
            return f"{int(t) // 60}:{int(t) % 60:02d}"
        return f"{mmss(self.start)}-{mmss(self.end if self.end is not None else self.start)}"

    @property
    def text(self) -> str:
        """The chunk's text. Same value as str(chunk); provided for callers
        that prefer to be explicit about which part of the chunk they want."""
        return str(self)

    def to_record(self) -> dict:
        """JSON-serialisable form, used by vector_store to persist chunks."""
        record = {"text": str(self), "source_file": self.source_file,
                  "page": self.page, "section": self.section}
        # Only written when set, so records for ordinary text documents keep
        # the shape they had before slides and audio carried extra metadata.
        if self.page_end is not None:
            record["page_end"] = self.page_end
        if self.start is not None:
            record["start"] = self.start
            record["end"] = self.end
        if self.from_image:
            record["from_image"] = True
        return record

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
            section=record.get("section"),
            page_end=record.get("page_end"),
            start=record.get("start"),
            end=record.get("end"),
            from_image=record.get("from_image", False),
        )

    def __repr__(self) -> str:
        extra = ""
        if self.page_end is not None:
            extra += f", page_end={self.page_end!r}"
        if self.start is not None:
            extra += f", start={self.start!r}, end={self.end!r}"
        if self.from_image:
            extra += ", from_image=True"
        return (f"Chunk(source_file={self.source_file!r}, page={self.page!r}, "
                f"section={self.section!r}{extra}, text={str(self)!r})")
