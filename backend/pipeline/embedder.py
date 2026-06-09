"""
backend/pipeline/embedder.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 3 of pipeline: EMBEDDING
Encodes text chunks into dense vectors using all-MiniLM-L6-v2 (CPU only).
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(chunks: List[str]) -> np.ndarray:
    """
    Encode a list of text chunks into a 2D numpy array of shape (n_chunks, 384).
    """
    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)
