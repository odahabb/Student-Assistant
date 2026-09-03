"""
backend/pipeline/sparse.py
Multimodal RAG Educational Assistant
Student: Omar Dahab — 23100704

Keyword (BM25) scoring for hybrid retrieval.

Dense embeddings rank passages by meaning, which is exactly where this
project's retrieval failed: the answering chunk lost to other passages of the
same paper that are *about* the same thing. Exact words — a model name, a
number, a term the question repeats from the text — separate those passages,
and BM25 scores exactly that. retriever.retrieve() mixes the two.

Okapi BM25 with the usual k1 = 1.5, b = 0.75, and the non-negative idf
log(1 + (N - df + 0.5) / (df + 0.5)). Implemented here rather than taken from
a library to keep the dependency list unchanged; it is a few lines of counting.

Tokenisation lowercases, keeps letters and digits, and strips a trailing
plural "s" (not "ss", "us" or "is"), so "stakeholders" matches "stakeholder".
The approach, and the 60/40 weighting used in retriever.py, follow a hybrid
retriever built for a separate project. The weighting was swept afterwards on
both of this project's question sets (data/eval/hybrid_weight_sweep.json and
hybrid_weight_sweep_slides.json), which prefer opposite directions, so it was
left where it was.
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
    return [_singular(w) for w in re.findall(r"[a-z0-9]+", str(text).lower())]


def index_text(chunk) -> str:
    """What BM25 indexes: the chunk with its section title in front."""
    section = getattr(chunk, "section", None)
    return f"{section}\n{chunk}" if section else str(chunk)


class BM25:
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
    return BM25([index_text(c) for c in chunks])
