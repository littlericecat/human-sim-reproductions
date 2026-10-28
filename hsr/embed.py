"""Sentence embeddings through llama.cpp (all-MiniLM-L6-v2, 23 MB) with an in-memory cache, and a bag-of-words
fallback so unit tests need no model. Cosine similarity is on unit vectors."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Optional

import numpy as np

from hsr.llm import DEFAULT_EMBED


class Embedder:
    dim = 384

    def __init__(self, model_path: str = DEFAULT_EMBED, n_threads: int = 2) -> None:
        self.model_path, self.n_threads = model_path, n_threads
        self._m = None
        self.cache: dict[str, np.ndarray] = {}

    def _load(self):
        if self._m is None:
            from llama_cpp import Llama

            self._m = Llama(model_path=self.model_path, embedding=True, n_ctx=512, n_threads=self.n_threads, verbose=False)
        return self._m

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        v = np.asarray(self._load().embed(text[:1500]), dtype=np.float32)
        if v.ndim > 1:
            v = v.mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        self.cache[text] = v
        return v


class BagOfWordsEmbedder:
    """Hashed unigram+bigram vectors; cosine on these is a crude relevance signal good enough for unit tests."""

    dim = 512

    def __init__(self) -> None:
        self.cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        toks = re.findall(r"[a-z0-9']+", text.lower())
        grams = toks + [a + "_" + b for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for g, c in Counter(grams).items():
            h = int(hashlib.md5(g.encode()).hexdigest(), 16)
            v[h % self.dim] += (1.0 if h & 1 else -1.0) * math.log1p(c)
        v = v / (np.linalg.norm(v) + 1e-9)
        self.cache[text] = v
        return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
