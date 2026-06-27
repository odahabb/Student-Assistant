"""
backend/pipeline/retriever.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 5 of pipeline: RETRIEVAL
Embeds a query and retrieves the top-k most similar chunks from the FAISS index.
"""

from typing import List

import numpy as np

from backend.pipeline.embedder import _get_model


def retrieve(query: str, index, chunks: List[str], k: int = 3) -> List[str]:
    """
    Embed the query and search the FAISS index for the top-k most similar chunks.
    """
    model = _get_model()
    query_embedding = model.encode([query], convert_to_numpy=True, show_progress_bar=False)
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

    _, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
