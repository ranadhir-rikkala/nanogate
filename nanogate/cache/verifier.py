from __future__ import annotations

import re
from dataclasses import dataclass

from nanogate.cache.embedding import content_words

_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
_NEGATION = re.compile(r"\b(?:not|no|never|without|cannot|none|neither|nor)\b|n't\b", re.IGNORECASE)
_ACRONYM = re.compile(r"\b[A-Z][A-Z0-9]{1,}\b")
_INNER_CAPS = re.compile(r"\b[a-z]+[A-Z]\w*\b|\b[A-Z][a-z]+[A-Z]\w*\b")
_PROPER = re.compile(r"(?<=[a-z,;] )[A-Z][a-z]{2,}\b")
ANTONYMS: tuple[tuple[str, str], ...] = (
    ("enable", "disable"),
    ("install", "uninstall"),
    ("increase", "decrease"),
    ("add", "remove"),
    ("start", "stop"),
    ("allow", "deny"),
    ("open", "close"),
    ("before", "after"),
    ("upgrade", "downgrade"),
    ("lock", "unlock"),
    ("join", "leave"),
    ("grant", "revoke"),
    ("encrypt", "decrypt"),
    ("create", "delete"),
    ("maximum", "minimum"),
    ("inbound", "outbound"),
    ("upload", "download"),
    ("mount", "unmount"),
)


@dataclass(frozen=True)
class Verdict:
    ok: bool
    score: float
    reason: str | None = None


def _names(text: str) -> set[str]:
    found = set(_ACRONYM.findall(text)) | set(_INNER_CAPS.findall(text)) | set(_PROPER.findall(text))
    return {name.lower() for name in found}


class SlotVerifier:
    def __init__(self, min_overlap: float = 0.5):
        self.min_overlap = min_overlap

    def verify(self, query: str, cached: str) -> Verdict:
        query_words, cached_words = set(content_words(query)), set(content_words(cached))
        union = query_words | cached_words
        overlap = len(query_words & cached_words) / len(union) if union else 0.0

        if set(_NUMBER.findall(query)) != set(_NUMBER.findall(cached)):
            return Verdict(False, overlap, "NUMBER_MISMATCH")
        if len(_NEGATION.findall(query)) % 2 != len(_NEGATION.findall(cached)) % 2:
            return Verdict(False, overlap, "NEGATION_MISMATCH")
        for first, second in ANTONYMS:
            for here, there in ((first, second), (second, first)):
                if here in query_words and there in cached_words and not (
                    there in query_words and here in cached_words
                ):
                    return Verdict(False, overlap, "ANTONYM_MISMATCH")
        if _names(query) != _names(cached):
            return Verdict(False, overlap, "NAME_MISMATCH")
        if query_words - cached_words and cached_words - query_words:
            return Verdict(False, overlap, "TERM_MISMATCH")
        if overlap < self.min_overlap:
            return Verdict(False, overlap, "LOW_OVERLAP")
        return Verdict(True, overlap)
