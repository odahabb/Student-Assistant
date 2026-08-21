"""
backend/pipeline/retriever.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Step 5 of pipeline: RETRIEVAL
Embeds a query and retrieves the top-k most similar chunks from the FAISS index.

Given a BM25 index (sparse.build_index) as well, retrieval is hybrid: every
chunk gets a dense score (cosine similarity) and a keyword score (BM25), each
is min-max scaled to 0..1 over the corpus, and the two are mixed with
DENSE_WEIGHT on the dense side. Without one, retrieval is dense only, as it
was for every result recorded before hybrid retrieval was added.
"""

from typing import List, Optional

import numpy as np

from backend.pipeline.embedder import _get_model, query_prefix

# 0.4 dense / 0.6 keyword, taken unchanged from the retriever it was borrowed
# from rather than tuned on this project's 25 questions.
DENSE_WEIGHT = 0.4


def _scaled(values: np.ndarray) -> np.ndarray:
    lo, hi = float(values.min()), float(values.max())
    return (values - lo) / ((hi - lo) or 1.0)


def retrieve(query: str, index, chunks: List[str], k: int = 3,
             sparse=None, dense_weight: float = DENSE_WEIGHT) -> List[str]:
    """
    Return the top-k chunks for a query: dense only, or hybrid when a BM25
    index built over the same chunks is passed as `sparse`.
    """
    model = _get_model()
    query_embedding = model.encode([query_prefix() + query], convert_to_numpy=True,
                                   normalize_embeddings=True, show_progress_bar=False)
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

    if sparse is None:
        _, indices = index.search(query_embedding, k)
        return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]

    n = len(chunks)
    distances, indices = index.search(query_embedding, n)
    dense = np.zeros(n, dtype=np.float32)
    valid = indices[0] >= 0
    # IndexFlatL2 returns squared distances; on unit vectors d = 2 - 2cos.
    dense[indices[0][valid]] = 1.0 - distances[0][valid] / 2.0
    keyword = np.asarray(sparse.scores(query), dtype=np.float32)
    if keyword.shape[0] != n:
        raise ValueError("BM25 index and chunk list have different lengths")

    fused = dense_weight * _scaled(dense) + (1 - dense_weight) * _scaled(keyword)
    return [chunks[i] for i in np.argsort(-fused, kind="stable")[:k]]
