"""
backend/pipeline/preprocessor.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 2 of pipeline: PREPROCESSING
Cleans raw text and splits it into overlapping chunks for embedding.
"""

import re
from typing import List

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def preprocess(text: str) -> List[str]:
    """
    Clean raw text and split it into overlapping chunks (~400 tokens, 50 token overlap).
    Tokens are approximated by whitespace-separated words.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected string, got {type(text)}")

    if len(text.strip()) < 20:
        raise ValueError("Input text is too short — OCR likely failed")

    # Remove non-ASCII characters
    cleaned = re.sub(r'[^\x00-\x7F]+', '', text)
    # Collapse extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    tokens = cleaned.split(' ')

    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start:start + CHUNK_SIZE]
        if not chunk_tokens:
            break
        chunks.append(' '.join(chunk_tokens))
        if start + CHUNK_SIZE >= len(tokens):
            break

    return chunks
