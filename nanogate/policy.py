from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from nanogate.models import DataClass, Identity, PolicyDecision, Reason, Route, ScanResult, canonical_json

PII_ACTIONS = ("allow", "local_only", "redact", "block")
LOW_CONFIDENCE_ACTIONS = ("flag", "abstain")
MODEL_ALIASES: dict[str, Route | None] = {
    "nanogate-auto": None,
    "nanogate-local": Route.LOCAL_SMALL,
    "nanogate-local-large": Route.LOCAL_LARGE,
    "nanogate-remote": Route.REMOTE,
}
_CONFIGURABLE_ROUTES = (Route.CACHE, Route.LOCAL_SMALL, Route.LOCAL_LARGE, Route.REMOTE)


class PolicyError(ValueError):
    pass


@dataclass(frozen=True)
class DepartmentPolicy:
    routes: tuple[Route, ...]
    remote_max_class: DataClass
    cache_max_class: DataClass
    pii_action: str
    low_confidence: str
    max_p_error: float
    requests_per_minute: int
    max_output_tokens: int
    daily_usd: float
    per_request_usd: float


def _department(raw: Mapping[str, Any], defaults: Mapping[str, Any], where: str) -> DepartmentPolicy:
    merged = {**defaults, **raw}
    budget = {**(defaults.get("budget") or {}), **(raw.get("budget") or {})}
    try:
        routes = tuple(Route(name) for name in merged["routes"])
        policy = DepartmentPolicy(
            routes=tuple(route for route in _CONFIGURABLE_ROUTES if route in routes),
            remote_max_class=DataClass.parse(merged["remote_max_class"]),
            cache_max_class=DataClass.parse(merged["cache_max_class"]),
            pii_action=str(merged["pii_action"]),
            low_confidence=str(merged["low_confidence"]),
            max_p_error=float(merged["max_p_error"]),
            requests_per_minute=int(merged["requests_per_minute"]),
            max_output_tokens=int(merged["max_output_tokens"]),
            daily_usd=float(budget["daily_usd"]),
            per_request_usd=float(budget["per_request_usd"]),
        )
    except (KeyError, ValueError, TypeError) as error:
        raise PolicyError(f"{where}: {error}") from error
    if set(routes) - set(_CONFIGURABLE_ROUTES):
        raise PolicyError(f"{where}: routes may only contain {[r.value for r in _CONFIGURABLE_ROUTES]}")
    if policy.pii_action not in PII_ACTIONS:
        raise PolicyError(f"{where}: pii_action must be one of {PII_ACTIONS}")
    if policy.low_confidence not in LOW_CONFIDENCE_ACTIONS:
        raise PolicyError(f"{where}: low_confidence must be one of {LOW_CONFIDENCE_ACTIONS}")
    if not 0.0 <= policy.max_p_error <= 1.0:
        raise PolicyError(f"{where}: max_p_error must be between 0 and 1")
    if policy.remote_max_class >= DataClass.SECRET or policy.cache_max_class >= DataClass.SECRET:
        raise PolicyError(f"{where}: secrets can never be sent remotely or cached")
    return policy


class PolicyEngine:
    def __init__(self, document: Mapping[str, Any]):
        if not isinstance(document, Mapping) or "tenants" not in document:
            raise PolicyError("policy document needs a 'tenants' section")
        defaults = document.get("defaults") or {}
        self.departments: dict[tuple[str, str], DepartmentPolicy] = {}
        for tenant, tenant_body in (document.get("tenants") or {}).items():
            for name, raw in ((tenant_body or {}).get("departments") or {}).items():
                key = (str(tenant).lower(), str(name).lower())
                self.departments[key] = _department(raw or {}, defaults, f"{key[0]}/{key[1]}")
        try:
            self.classification = {
                str(entity): DataClass.parse(value)
                for entity, value in (document.get("classification") or {}).items()
            }
        except KeyError as error:
            raise PolicyError(f"classification: unknown data class {error}") from error
        digest = hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()[:10]
        self.version = f"{document.get('version', '0')}+{digest}"
        self.document = dict(document)

    @classmethod
    def from_file(cls, path: str | Path) -> "PolicyEngine":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle) or {})

    def department(self, identity: Identity) -> DepartmentPolicy | None:
        return self.departments.get((identity.tenant, identity.department))

    def evaluate(
        self,
        identity: Identity,
        scan: ScanResult,
        pinned: Route | None = None,
        remote_enabled: bool = False,
        redacted: bool = False,
    ) -> PolicyDecision:
        rules = self.department(identity)
        if rules is None:
            return self._block(Reason.POLICY_BLOCK)
        if scan.has_secret:
            return self._block(Reason.SECRET_BLOCKED)

        reasons: list[str] = [Reason.INPUT_REDACTED] if redacted else []
        sensitive = scan.data_class >= DataClass.CONFIDENTIAL
        if sensitive and rules.pii_action == "block":
            return self._block(Reason.POLICY_BLOCK)
        if sensitive and rules.pii_action == "redact" and not redacted:
            return PolicyDecision(
                allowed_routes=rules.routes,
                reasons=(),
                policy_version=self.version,
                max_p_error=rules.max_p_error,
                redact_input=True,
            )

        routes = list(rules.routes)
        if Route.REMOTE in routes:
            too_sensitive = scan.data_class > rules.remote_max_class
            if sensitive and rules.pii_action != "allow":
                too_sensitive = True
            if too_sensitive:
                routes.remove(Route.REMOTE)
                reasons.append(Reason.SENSITIVE_LOCAL_ONLY)
            elif not remote_enabled:
                routes.remove(Route.REMOTE)
                reasons.append(Reason.REMOTE_DISABLED)

        cacheable = Route.CACHE in routes and scan.data_class <= rules.cache_max_class
        if Route.CACHE in routes and not cacheable:
            routes.remove(Route.CACHE)

        if pinned is not None:
            if pinned not in routes:
                return self._block(Reason.ROUTE_NOT_PERMITTED, *reasons)
            routes = [pinned]
            cacheable = False

        return PolicyDecision(
            allowed_routes=tuple(routes),
            reasons=tuple(reasons),
            policy_version=self.version,
            max_p_error=rules.max_p_error,
            cacheable=cacheable,
            abstain_on_low_confidence=rules.low_confidence == "abstain",
        )

    def _block(self, *reasons: str) -> PolicyDecision:
        return PolicyDecision(allowed_routes=(), reasons=tuple(reasons), policy_version=self.version)


def effective_threshold(learned: float, policy_cap: float | None) -> float:
    return learned if policy_cap is None else min(learned, policy_cap)
