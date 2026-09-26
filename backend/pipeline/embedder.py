"""
backend/pipeline/embedder.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 3 of pipeline: EMBEDDING
Encodes text chunks into dense 384-d vectors.
Runs on CPU by default; supports optional Intel Arc GPU / NPU acceleration
via the SA_DEVICE env var (see backend/pipeline/device.py).

Three models are available, selected with the SA_EMBEDDER env var:

  minilm     all-MiniLM-L6-v2
  multi-qa   multi-qa-MiniLM-L6-cos-v1
  bge-small  BAAI/bge-small-en-v1.5, which expects an instruction in front of
             a search query; retriever.py adds it from query_prefix()

All three share the bert-base-uncased tokenizer, so chunk boundaries are the
same whichever is selected. Models are loaded once and cached by key. The
quiz grader always asks for minilm explicitly.
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.pipeline.device import get_torch_device, should_use_npu

log = logging.getLogger(__name__)

MODELS = {
    "minilm": {"name": "all-MiniLM-L6-v2", "query_prefix": ""},
    "multi-qa": {"name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                 "query_prefix": ""},
    "bge-small": {"name": "BAAI/bge-small-en-v1.5",
                  "query_prefix": "Represent this sentence for searching "
                                  "relevant passages: "},
}
DEFAULT_MODEL = "minilm"
MODEL_NAME = MODELS[DEFAULT_MODEL]["name"]

_models: Dict[str, SentenceTransformer] = {}


def model_key(key: Optional[str] = None) -> str:
    """The model key to use: `key` if given, else SA_EMBEDDER, else minilm."""
    key = key or os.environ.get("SA_EMBEDDER", DEFAULT_MODEL)
    if key not in MODELS:
        log.warning(f"Unknown SA_EMBEDDER '{key}' — using {DEFAULT_MODEL}")
        key = DEFAULT_MODEL
    return key


def query_prefix(key: Optional[str] = None) -> str:
    """The instruction a model wants in front of a query, or "" if none."""
    return MODELS[model_key(key)]["query_prefix"]


def _get_model(key: Optional[str] = None) -> SentenceTransformer:
    """The loaded SentenceTransformer for `key`, built on first use."""
    key = model_key(key)
    if key in _models:
        return _models[key]
    name = MODELS[key]["name"]

    if should_use_npu():
        try:
            _models[key] = SentenceTransformer(
                name, backend="openvino",
                model_kwargs={"device": "NPU"},
            )
            return _models[key]
        except Exception as e:
            log.warning(f"NPU embedder load failed ({e}), falling back to torch CPU/GPU")

    _models[key] = SentenceTransformer(name, device=get_torch_device())
    return _models[key]


def with_section_context(chunk) -> str:
    """
    The text to embed for a chunk when section context is on: the section
    title, then the chunk ("Model. whisper uses an encoder ..."). Only the
    vector changes; the stored chunk text is untouched.
    """
    section = getattr(chunk, "section", None)
    return f"{section}. {chunk}" if section else str(chunk)


def embed(chunks: List[str], section_context: bool = False,
          model: Optional[str] = None) -> np.ndarray:
    """
    Encode a list of text chunks into a 2D numpy array of shape
    (n_chunks, 384). Vectors are L2-normalised, so a dot product between two
    of them is their cosine similarity.

    section_context=True embeds each chunk with its section title in front
    (see with_section_context). `model` overrides SA_EMBEDDER.
    """
    encoder = _get_model(model)
    texts = [with_section_context(c) for c in chunks] if section_context else chunks
    embeddings = encoder.encode(texts, convert_to_numpy=True,
                                normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embeddings, dtype=np.float32)
