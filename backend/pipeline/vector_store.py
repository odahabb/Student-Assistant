"""
backend/pipeline/vector_store.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 4 of pipeline: VECTOR STORE
Builds, saves, and loads a FAISS IndexFlatL2 index plus the chunk texts.
"""

import json
import os
from typing import List, Sequence, Tuple

import faiss
import numpy as np

from backend.pipeline.chunk import Chunk

INDEX_PATH = "data/processed/index.faiss"
CHUNKS_PATH = "data/processed/chunks.json"


def build_and_save(embeddings: np.ndarray, chunks: Sequence[str]) -> None:
    """
    Build a FAISS IndexFlatL2 index from embeddings, add all vectors,
    and save the index and the chunks to disk.

    Chunks are stored as {text, source_file, page} records so a chunk's
    provenance survives the round trip — retrieval on a reloaded store can
    still say which file and page an answer came from. Plain strings are
    accepted too and persist with null metadata.
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    records = [
        c.to_record() if isinstance(c, Chunk) else Chunk(c).to_record()
        for c in chunks
    ]

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load() -> Tuple[faiss.Index, List[Chunk]]:
    """
    Load the FAISS index and chunks from disk.

    Returns Chunk objects, which behave as plain strings everywhere downstream
    and additionally carry .source_file / .page. Chunk files written before
    metadata existed (a bare JSON list of strings) still load, with both
    fields set to None.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Vector store not found — run build_and_save() first")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    return index, [Chunk.from_record(r) for r in records]
