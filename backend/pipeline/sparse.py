"""
backend/pipeline/sparse.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Keyword (BM25) scoring, the sparse half of hybrid retrieval.

Okapi BM25 with k1 = 1.5, b = 0.75 and the non-negative idf
log(1 + (N - df + 0.5) / (df + 0.5)). Scores are per chunk, in the order the
chunks were indexed, and retriever.retrieve() mixes them with the dense
cosine scores.

Tokenisation lowercases the text, keeps runs of letters and digits, and
strips a trailing plural "s" (but not "ss", "us" or "is"), so "stakeholders"
and "stakeholder" produce the same token.
"""

import math
import re
from collections import Counter
from typing import List, Sequence

K1 = 1.5
B = 0.75


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric tokens, each reduced to its singular form."""
    return [_singular(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]


def index_text(chunk) -> str:
    """The text BM25 indexes for a chunk: its section title, then its text."""
    section = getattr(chunk, "section", None)
    return f"{section}\n{chunk}" if section else str(chunk)


class BM25:
    """Term counts, document lengths and idf weights for one set of chunks."""

    def __init__(self, documents: Sequence[str]):
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        df = Counter()
        for counts in self.docs:
            df.update(counts.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> List[float]:
        """One BM25 score per indexed document, in index order."""
        terms = [t for t in tokenize(query) if t in self.idf]
        out = []
        for counts, length in zip(self.docs, self.lengths):
            norm = K1 * (1 - B + B * length / (self.avg_length or 1))
            score = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    score += self.idf[t] * tf * (K1 + 1) / (tf + norm)
            out.append(score)
        return out


def build_index(chunks: Sequence) -> BM25:
    """Build a BM25 index over the chunks, each read through index_text."""
    return BM25([index_text(c) for c in chunks])
