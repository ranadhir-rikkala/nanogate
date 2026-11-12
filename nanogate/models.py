from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Route(str, Enum):
    CACHE = "cache"
    LOCAL_SMALL = "local_small"
    LOCAL_LARGE = "local_large"
    REMOTE = "remote"
    ABSTAIN = "abstain"
    BLOCKED = "blocked"


ANSWERING_ROUTES = (Route.CACHE, Route.LOCAL_SMALL, Route.LOCAL_LARGE, Route.REMOTE)


class DataClass(IntEnum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    SECRET = 4

    @property
    def label(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: str) -> "DataClass":
        return cls[str(value).strip().upper()]


class Reason:
    AUTH_INVALID = "AUTH_INVALID"
    AUTH_REVOKED = "AUTH_REVOKED"
    SCOPE_DENIED = "SCOPE_DENIED"
    IDENTITY_SPOOF_REJECTED = "IDENTITY_SPOOF_REJECTED"
    RATE_LIMITED = "RATE_LIMITED"
    INPUT_TOO_LARGE = "INPUT_TOO_LARGE"
    DLP_FAIL_CLOSED = "DLP_FAIL_CLOSED"
    SECRET_BLOCKED = "SECRET_BLOCKED"
    POLICY_BLOCK = "POLICY_BLOCK"
    ROUTE_NOT_PERMITTED = "ROUTE_NOT_PERMITTED"
    SENSITIVE_LOCAL_ONLY = "SENSITIVE_LOCAL_ONLY"
    INPUT_REDACTED = "INPUT_REDACTED"
    BUDGET_DENY = "BUDGET_DENY"
    CACHE_VERIFIED = "CACHE_VERIFIED"
    CACHE_HARD_NEGATIVE = "CACHE_HARD_NEGATIVE"
    LOCAL_CONFIDENT = "LOCAL_CONFIDENT"
    LOCAL_LARGE_SELECTED = "LOCAL_LARGE_SELECTED"
    LOCAL_UNVERIFIED = "LOCAL_UNVERIFIED"
    REMOTE_ALLOWED = "REMOTE_ALLOWED"
    REMOTE_DISABLED = "REMOTE_DISABLED"
    ROUTER_ABSTAIN = "ROUTER_ABSTAIN"
    OUTPUT_REDACTED = "OUTPUT_REDACTED"
    OUTPUT_BLOCKED = "OUTPUT_BLOCKED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

    PRECEDENCE = (
        AUTH_INVALID,
        AUTH_REVOKED,
        SCOPE_DENIED,
        IDENTITY_SPOOF_REJECTED,
        RATE_LIMITED,
        INPUT_TOO_LARGE,
        DLP_FAIL_CLOSED,
        SECRET_BLOCKED,
        POLICY_BLOCK,
        ROUTE_NOT_PERMITTED,
        BUDGET_DENY,
        OUTPUT_BLOCKED,
        MODEL_UNAVAILABLE,
        ROUTER_ABSTAIN,
        CACHE_VERIFIED,
        REMOTE_ALLOWED,
        LOCAL_LARGE_SELECTED,
        LOCAL_UNVERIFIED,
        LOCAL_CONFIDENT,
        SENSITIVE_LOCAL_ONLY,
        REMOTE_DISABLED,
        OUTPUT_REDACTED,
        INPUT_REDACTED,
        CACHE_HARD_NEGATIVE,
    )

    @classmethod
    def primary(cls, reasons: list[str]) -> str:
        for code in cls.PRECEDENCE:
            if code in reasons:
                return code
        return reasons[0] if reasons else "UNKNOWN"


def degraded(module: str) -> str:
    return f"DEGRADED_{module}"


@dataclass(frozen=True)
class Identity:
    tenant: str
    department: str
    role: str = "app"
    key_id: str = ""


@dataclass(frozen=True)
class Finding:
    entity: str
    start: int
    end: int


@dataclass(frozen=True)
class ScanResult:
    data_class: DataClass = DataClass.PUBLIC
    entities: dict[str, int] = field(default_factory=dict)
    has_secret: bool = False
    findings: tuple[Finding, ...] = ()

    @property
    def entity_count(self) -> int:
        return sum(self.entities.values())

    def summary(self) -> dict[str, Any]:
        return {
            "data_class": self.data_class.label,
            "entities": dict(sorted(self.entities.items())),
            "has_secret": self.has_secret,
        }


@dataclass(frozen=True)
class PolicyDecision:
    allowed_routes: tuple[Route, ...]
    reasons: tuple[str, ...]
    policy_version: str
    max_p_error: float | None = None
    redact_input: bool = False
    cacheable: bool = False
    abstain_on_low_confidence: bool = False

    @property
    def blocked(self) -> bool:
        return not self.allowed_routes

    def permits(self, route: Route) -> bool:
        return route in self.allowed_routes


@dataclass(frozen=True)
class Reservation:
    granted: bool
    reservation_id: str | None = None
    amount_usd: float = 0.0
    remaining_usd: float = 0.0


@dataclass(frozen=True)
class CacheLookup:
    hit: bool
    answer: str | None = None
    similarity: float | None = None
    verifier_score: float | None = None
    reason: str | None = None
    entry_id: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "hit": self.hit,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "verifier_score": None if self.verifier_score is None else round(self.verifier_score, 4),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RouterScore:
    p_error: float
    threshold: float
    top_factors: tuple[str, ...] = ()
    trained: bool = True

    @property
    def confident(self) -> bool:
        return self.p_error <= self.threshold


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    route: Route
    prompt_tokens: int = 0
    completion_tokens: int = 0
    logprobs: tuple[float, ...] = ()
    latency_ms: float = 0.0
    simulated: bool = False


@dataclass
class Stage:
    name: str
    ms: float
    detail: str = ""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
