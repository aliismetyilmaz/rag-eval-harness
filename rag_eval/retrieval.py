"""BM25 keyword retrieval, implemented in plain Python.

BM25 keeps the project dependency-free and easy to inspect. An embedding-based
retriever would be the natural next experiment (see README "What I'd do next").
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .corpus import Chunk

STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it its my of on or "
    "our so that the this to we what when which will with you your".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, i: int) -> float:
        tf, length = self.tfs[i], self.lengths[i]
        total = 0.0
        for term in tokenize(query):
            if term not in tf:
                continue
            freq = tf[term]
            norm = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[term] * freq * (self.k1 + 1) / norm
        return total

    def search(self, query: str, k: int) -> list[Chunk]:
        ranked = sorted(range(len(self.chunks)), key=lambda i: self.score(query, i), reverse=True)
        return [self.chunks[i] for i in ranked[:k]]
