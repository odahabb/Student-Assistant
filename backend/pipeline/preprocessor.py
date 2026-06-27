"""
backend/pipeline/preprocessor.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 2 of pipeline: PREPROCESSING
Cleans raw text and splits it into overlapping chunks for embedding.
""" 

import re
from typing import List

from backend.pipeline.embedder import _get_model


def preprocess(text: str, chunk_tokens: int = 220, overlap: int = 40) -> List[str]:
    """
    Clean raw text and split it into overlapping chunks sized to fit the
    embedder's 256-token limit (220-token windows, 40-token overlap by default).
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    if len(text.strip()) < 20:
        raise ValueError("Input text is too short — OCR likely failed")

    # Collapse extra whitespace 
    cleaned = re.sub(r'\s+', ' ', text).strip()

    tokenizer = _get_model().tokenizer
    token_ids = tokenizer.encode(cleaned, add_special_tokens=False)

    if not token_ids:
        return []

    step = chunk_tokens - overlap
    chunks = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start:start + chunk_tokens]
        chunks.append(tokenizer.decode(window))
        if start + chunk_tokens >= len(token_ids):
            break
        start += step

    return chunks
