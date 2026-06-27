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


def embed(chunks: List[str]) -> np.ndarray:
    """
    Encode a list of text chunks into a 2D numpy array of shape (n_chunks, 384).
    """
    model = _get_model()
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)
