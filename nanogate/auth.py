from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
from typing import Mapping

from nanogate.db import Database
from nanogate.models import Identity, Reason

ROLES = ("app", "auditor", "admin")
TENANT_HEADER = "x-nanogate-tenant"
DEPARTMENT_HEADER = "x-nanogate-department"
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_KEY = re.compile(r"^ng_([a-f0-9]{12})_([A-Za-z0-9_-]{20,})$")


class AuthError(Exception):
    def __init__(self, code: str, message: str, status: int = 401):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class KeyStore:
    def __init__(self, db: Database, pepper: str = ""):
        self.db = db
        self.pepper = pepper.encode("utf-8")

    def _hash(self, key: str) -> str:
        return hmac.new(self.pepper, key.encode("utf-8"), hashlib.sha256).hexdigest()

    def create(self, tenant: str, department: str, role: str = "app", label: str = "") -> tuple[str, str]:
        tenant, department = tenant.lower(), department.lower()
        if not _SLUG.match(tenant) or not _SLUG.match(department):
            raise ValueError("tenant and department must match [a-z0-9][a-z0-9_-]{0,63}")
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}")
        key_id = secrets.token_hex(6)
        key = f"ng_{key_id}_{secrets.token_urlsafe(32)}"
        with self.db.write() as conn:
            conn.execute(
                "INSERT INTO api_keys(key_id, key_hash, tenant, department, role, label, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (key_id, self._hash(key), tenant, department, role, label, time.time()),
            )
        return key_id, key

    def resolve(self, key: str | None) -> Identity:
        match = _KEY.match(key or "")
        if not match:
            raise AuthError(Reason.AUTH_INVALID, "Missing or malformed API key.")
        row = self.db.one("SELECT * FROM api_keys WHERE key_id = ?", (match.group(1),))
        if row is None or not hmac.compare_digest(row["key_hash"], self._hash(key)):
            raise AuthError(Reason.AUTH_INVALID, "Unknown API key.")
        if row["revoked_at"] is not None:
            raise AuthError(Reason.AUTH_REVOKED, "This API key has been revoked.")
        return Identity(row["tenant"], row["department"], row["role"], row["key_id"])

    def revoke(self, key_id: str) -> bool:
        with self.db.write() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE key_id = ? AND revoked_at IS NULL",
                (time.time(), key_id),
            )
        return cursor.rowcount == 1

    def list(self) -> list[dict]:
        rows = self.db.query(
            "SELECT key_id, tenant, department, role, label, created_at, revoked_at"
            " FROM api_keys ORDER BY created_at"
        )
        return [dict(row) for row in rows]


def bearer_token(headers: Mapping[str, str]) -> str | None:
    value = headers.get("authorization") or headers.get("Authorization") or ""
    scheme, _, token = value.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def reject_spoofing(identity: Identity, headers: Mapping[str, str]) -> None:
    claims = ((TENANT_HEADER, identity.tenant), (DEPARTMENT_HEADER, identity.department))
    for header, actual in claims:
        claimed = headers.get(header)
        if claimed is not None and claimed.strip().lower() != actual:
            raise AuthError(
                Reason.IDENTITY_SPOOF_REJECTED,
                f"Header {header} does not match the identity bound to this API key.",
                status=403,
            )


def require_role(identity: Identity, *roles: str) -> None:
    if identity.role not in roles:
        raise AuthError(Reason.SCOPE_DENIED, f"Role '{identity.role}' may not do this.", status=403)


class RateLimiter:
    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, per_minute: int) -> bool:
        if per_minute <= 0:
            return True
        now = self._clock()
        with self._lock:
            tokens, stamp = self._buckets.get(key, (float(per_minute), now))
            tokens = min(float(per_minute), tokens + (now - stamp) * per_minute / 60.0)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True
