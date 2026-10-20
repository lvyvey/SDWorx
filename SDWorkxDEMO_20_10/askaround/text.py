"""Small dependency-free TF-IDF used for document-document similarity and question matching.

Swap `TfidfIndex` for a text-embedding model later: anything with `vector(text)` and
`similarity(a, b)` returning values in [0, 1] works.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, Iterable, List

STOPWORDS = {
    "the", "and", "for", "are", "with", "that", "this", "from", "which", "what", "when", "how",
    "who", "into", "over", "under", "must", "should", "can", "our", "their", "they", "per",
    "all", "any", "not", "but", "has", "have", "been", "was", "were", "will", "its", "apply",
    "where", "why", "does", "did", "about", "also", "then", "than", "only", "such", "these", "those",
    "other", "some", "more", "most", "each", "there", "here", "need", "you", "your", "get", "know",
    "de", "het", "een", "van", "voor", "met", "die", "dat", "zijn", "worden", "wordt", "bij",
}

Vector = Dict[str, float]


def _stem(word: str) -> str:
    # Very light stemming so "payslips" matches "payslip" and "contributions" matches "contribution".
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zà-ÿ0-9]+", (text or "").lower())
    return [_stem(w) for w in words if len(w) > 2 and w not in STOPWORDS]


def cosine(a: Vector, b: Vector) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


class TfidfIndex:
    def __init__(self, documents: Dict[str, str]) -> None:
        self._tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
        n = len(self._tokens)
        df: Counter = Counter()
        for tokens in self._tokens.values():
            df.update(set(tokens))
        # Smoothed idf so words appearing everywhere still count a little.
        self.idf = {term: math.log((1 + n) / (1 + count)) + 1.0 for term, count in df.items()}
        self.vectors = {doc_id: self._weigh(tokens) for doc_id, tokens in self._tokens.items()}

    def _weigh(self, tokens: Iterable[str]) -> Vector:
        counts = Counter(t for t in tokens if t in self.idf)
        total = sum(counts.values())
        return {term: (count / total) * self.idf[term] for term, count in counts.items()} if total else {}

    def vector(self, text: str) -> Vector:
        return self._weigh(tokenize(text))

    def similarity(self, doc_a: str, doc_b: str) -> float:
        return cosine(self.vectors.get(doc_a, {}), self.vectors.get(doc_b, {}))
