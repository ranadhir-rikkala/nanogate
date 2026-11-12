from __future__ import annotations

import re
import unicodedata
from typing import Callable, Mapping

from nanogate.models import DataClass, Finding, ScanResult

SECRET_ENTITIES = frozenset(
    {"OPENAI_KEY", "AWS_ACCESS_KEY", "GITHUB_TOKEN", "PRIVATE_KEY", "JWT", "CREDENTIAL_ASSIGNMENT"}
)

DEFAULT_CLASSES: dict[str, DataClass] = {
    "IP_ADDRESS": DataClass.INTERNAL,
    "EMAIL": DataClass.CONFIDENTIAL,
    "PHONE": DataClass.CONFIDENTIAL,
    "EMPLOYEE_ID": DataClass.CONFIDENTIAL,
    "SSN": DataClass.RESTRICTED,
    "CREDIT_CARD": DataClass.RESTRICTED,
    **{entity: DataClass.SECRET for entity in SECRET_ENTITIES},
}

MAX_SCAN_CHARS = 200_000
_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)


def _luhn(match: str) -> bool:
    digits = [int(char) for char in match if char.isdigit()]
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit = digit * 2 - 9 if digit > 4 else digit * 2
        total += digit
    return total % 10 == 0


def _ssn(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def _ipv4(match: str) -> bool:
    return all(0 <= int(part) <= 255 for part in match.split("."))


def _phone(match: str) -> bool:
    return 10 <= sum(char.isdigit() for char in match) <= 15


Detector = tuple[str, re.Pattern[str], Callable[[str], bool] | None]

DETECTORS: tuple[Detector, ...] = (
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----"), None),
    ("OPENAI_KEY", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"), None),
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"), None),
    ("GITHUB_TOKEN", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,})"), None),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), None),
    (
        "CREDENTIAL_ASSIGNMENT",
        re.compile(
            r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
            r"\s*(?:=|:|\bis\b)\s*[\"']?(?=[^\s\"']*\d)[^\s\"']{8,}"
        ),
        None,
    ),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), None),
    ("SSN", re.compile(r"(?<![\d-])\d{3}[- ]\d{2}[- ]\d{4}(?![\d-])"), _ssn),
    ("CREDIT_CARD", re.compile(r"(?<![\d-])\d(?:[ -]?\d){12,18}(?![\d-])"), _luhn),
    ("EMPLOYEE_ID", re.compile(r"\bEMP-\d{6}\b"), None),
    ("IP_ADDRESS", re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])"), _ipv4),
    (
        "PHONE",
        re.compile(r"(?<![\w.-])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)|\d{3})[ .-]?\d{3}[ .-]?\d{4}(?![\w-])"),
        _phone,
    ),
)

_OBFUSCATIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?i)\s*[\[\(\{]\s*at\s*[\]\)\}]\s*"), "@"),
    (re.compile(r"(?i)\s*[\[\(\{]\s*dot\s*[\]\)\}]\s*"), "."),
    (re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)"), "-"),
)
_SPELLED_EMAIL = re.compile(r"(?i)\b[\w+-]+(?: dot [\w+-]+)* at [\w-]+(?: dot [\w-]+)+\b")
_SPACED_DIGITS = re.compile(r"(?<!\S)\d(?: \d){8,}(?!\S)")


def _join_spelled_email(match: re.Match[str]) -> str:
    joined = re.sub(r"(?i) at ", "@", match.group(0), count=1)
    return re.sub(r"(?i) dot ", ".", joined)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    for pattern, replacement in _OBFUSCATIONS:
        text = pattern.sub(replacement, text)
    text = _SPELLED_EMAIL.sub(_join_spelled_email, text)
    return _SPACED_DIGITS.sub(lambda match: match.group(0).replace(" ", ""), text)


def _detect(text: str) -> list[Finding]:
    findings: list[Finding] = []
    for entity, pattern, validate in DETECTORS:
        for match in pattern.finditer(text):
            if validate is not None and not validate(match.group(0)):
                continue
            if any(match.start() < kept.end and kept.start < match.end() for kept in findings):
                continue
            findings.append(Finding(entity, match.start(), match.end()))
    return sorted(findings, key=lambda finding: finding.start)


def _count(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.entity] = counts.get(finding.entity, 0) + 1
    return counts


class Scanner:
    def __init__(self, classes: Mapping[str, DataClass] | None = None, floor: DataClass = DataClass.PUBLIC):
        self.classes = dict(DEFAULT_CLASSES)
        if classes:
            self.classes.update(classes)
        self.floor = floor

    def scan(self, text: str | None) -> ScanResult:
        text = (text or "")[:MAX_SCAN_CHARS]
        findings = _detect(text)
        counts = _count(findings)
        for entity, count in _count(_detect(normalize(text))).items():
            counts[entity] = max(counts.get(entity, 0), count)
        data_class = self.floor
        for entity in counts:
            data_class = max(data_class, self.classes.get(entity, DataClass.CONFIDENTIAL))
        has_secret = any(entity in SECRET_ENTITIES for entity in counts)
        return ScanResult(
            data_class=DataClass.SECRET if has_secret else data_class,
            entities=counts,
            has_secret=has_secret,
            findings=tuple(findings),
        )

    def redact(self, text: str, only: frozenset[str] | None = None) -> tuple[str, int]:
        working = normalize(text) if self._needs_normalizing(text) else text
        findings = [finding for finding in _detect(working) if only is None or finding.entity in only]
        for finding in reversed(findings):
            working = working[: finding.start] + f"[{finding.entity}]" + working[finding.end :]
        return working, len(findings)

    def _needs_normalizing(self, text: str) -> bool:
        return _count(_detect(text)) != _count(_detect(normalize(text)))


def new_entities(output: ScanResult, source: ScanResult) -> dict[str, int]:
    extra: dict[str, int] = {}
    for entity, count in output.entities.items():
        surplus = count - source.entities.get(entity, 0)
        if surplus > 0:
            extra[entity] = surplus
    return extra
