from __future__ import annotations

import hashlib
import math
import re
from array import array
from typing import Protocol, Sequence

_WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
STOPWORDS = frozenset(
    "a an the is are was were be been do does did can could should would will i my me we our you your it its"
    " to of in on for with at by from and or how what when where which who why this that these those there"
    " please need want".split()
)


def stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix) and not word.endswith("ss"):
            base = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(base) > 3 and base[-1] == base[-2] and base[-1] not in "aeiouls":
                base = base[:-1]
            return base
    return word


def content_words(text: str) -> list[str]:
    return [stem(word) for word in _WORD.findall(text.lower()) if word not in STOPWORDS]


class Embedder(Protocol):
    name: str

    def embed(self, text: str) -> Sequence[float]: ...


class HashingEmbedder:
    def __init__(self, dims: int = 512):
        self.dims = dims
        self.name = f"hashing-{dims}"

    def _add(self, vector: list[float], feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        vector[value % self.dims] += weight if (value >> 63) & 1 else -weight

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        words = content_words(text)
        for word in words:
            self._add(vector, f"w:{word}", 1.0)
            padded = f"^{word}$"
            for index in range(len(padded) - 2):
                self._add(vector, f"c:{padded[index:index + 3]}", 0.25)
        for left, right in zip(words, words[1:]):
            self._add(vector, f"b:{left} {right}", 0.5)
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        return [float(value) for value in self._model.encode(text, normalize_embeddings=True)]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def pack(vector: Sequence[float]) -> bytes:
    return array("f", vector).tobytes()


def unpack(blob: bytes) -> array:
    vector = array("f")
    vector.frombytes(blob)
    return vector
