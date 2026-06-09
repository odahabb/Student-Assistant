"""
backend/pipeline/vector_store.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 4 of pipeline: VECTOR STORE
Builds, saves, and loads a FAISS IndexFlatL2 index plus the chunk texts.
"""

import json
import os
from typing import List, Tuple

import faiss
import numpy as np

INDEX_PATH = "data/processed/index.faiss"
CHUNKS_PATH = "data/processed/chunks.json"


def build_and_save(embeddings: np.ndarray, chunks: List[str]) -> None:
    """
    Build a FAISS IndexFlatL2 index from embeddings, add all vectors,
    and save the index and chunk texts to disk.
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def load() -> Tuple[faiss.Index, List[str]]:
    """
    Load the FAISS index and chunk texts from disk.
    """
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError("Vector store not found — run build_and_save() first")

    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, chunks
