"""
backend/pipeline/embedder.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 3 of pipeline: EMBEDDING
Encodes text chunks into dense vectors using all-MiniLM-L6-v2.
Runs on CPU by default; supports optional Intel Arc GPU / NPU acceleration
via the SA_DEVICE env var (see backend/pipeline/device.py).
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.pipeline.device import get_torch_device, should_use_npu

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is not None:
        return _model

    if should_use_npu():
        try:
            _model = SentenceTransformer(
                MODEL_NAME, backend="openvino",
                model_kwargs={"device": "NPU"},
            )
            return _model
        except Exception as e:
            log.warning(f"NPU embedder load failed ({e}), falling back to torch CPU/GPU")

    _model = SentenceTransformer(MODEL_NAME, device=get_torch_device())
    return _model


def with_section_context(chunk) -> str:
    """
    The text to embed for a chunk when section context is on: the section
    title in front of the chunk ("Model. whisper uses an encoder ..."). Only
    the vector changes — the stored chunk text, and what the generator sees,
    stay the same.
    """
    section = getattr(chunk, "section", None)
    return f"{section}. {chunk}" if section else str(chunk)


def embed(chunks: List[str], section_context: bool = False) -> np.ndarray:
    """
    Encode a list of text chunks into a 2D numpy array of shape (n_chunks, 384).

    section_context=True embeds each chunk with its section title in front
    (with_section_context), an experimental variant compared in
    backend/scripts/retrieval_variants.py.
    """
    model = _get_model()
    texts = [with_section_context(c) for c in chunks] if section_context else chunks
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)
