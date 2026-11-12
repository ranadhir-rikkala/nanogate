from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Callable, Mapping

from nanogate.db import Database
from nanogate.models import canonical_json

GENESIS = "0" * 64
SEALED = "SEALED"
INTEGRITY_FAILED = "INTEGRITY_FAILED"
_INDEXED = ("tenant", "department", "route", "reason", "data_class")


def chain_digest(prev_hash: str, body: Mapping[str, Any]) -> str:
    return hashlib.sha256((prev_hash + canonical_json(body)).encode("utf-8")).hexdigest()


class ReceiptLedger:
    def __init__(self, db: Database, key: str, clock: Callable[[], float] = time.time):
        if not key:
            raise ValueError("a receipt signing key is required")
        self.db = db
        self.key = key.encode("utf-8")
        self.clock = clock

    def _seal(self, seq: int, digest: str) -> str:
        return hmac.new(self.key, f"{seq}:{digest}".encode("utf-8"), hashlib.sha256).hexdigest()

    def append(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        with self.db.write() as conn:
            head = conn.execute("SELECT seq, digest FROM receipts ORDER BY seq DESC LIMIT 1").fetchone()
            seq = 1 if head is None else head["seq"] + 1
            prev_hash = GENESIS if head is None else head["digest"]
            body = dict(fields)
            body.update(receipt_id=f"rcpt_{uuid.uuid4().hex}", seq=seq, created_at=round(self.clock(), 3))
            digest = chain_digest(prev_hash, body)
            seal = self._seal(seq, digest)
            conn.execute(
                "INSERT INTO receipts(seq, receipt_id, created_at, tenant, department, route, reason,"
                " data_class, cost_usd, latency_ms, body, prev_hash, digest, seal)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    seq,
                    body["receipt_id"],
                    body["created_at"],
                    *(str(body.get(name, "")) for name in _INDEXED),
                    float(body.get("cost_usd", 0.0)),
                    float(body.get("latency_ms", 0.0)),
                    canonical_json(body),
                    prev_hash,
                    digest,
                    seal,
                ),
            )
        return {**body, "prev_hash": prev_hash, "digest": digest, "seal": seal}

    def get(self, receipt_id: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT body, prev_hash, digest, seal FROM receipts WHERE receipt_id = ?", (receipt_id,))
        if row is None:
            return None
        return {**json.loads(row["body"]), "prev_hash": row["prev_hash"], "digest": row["digest"], "seal": row["seal"]}

    def recent(self, limit: int = 50, tenant: str | None = None, department: str | None = None) -> list[dict]:
        clauses, params = [], []
        if tenant:
            clauses.append("tenant = ?")
            params.append(tenant)
        if department:
            clauses.append("department = ?")
            params.append(department)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.query(
            f"SELECT body, digest FROM receipts {where} ORDER BY seq DESC LIMIT ?", (*params, int(limit))
        )
        return [{**json.loads(row["body"]), "digest": row["digest"]} for row in rows]

    def head(self) -> dict[str, Any] | None:
        row = self.db.one("SELECT seq, digest, seal FROM receipts ORDER BY seq DESC LIMIT 1")
        return None if row is None else dict(row)

    def verify(self, expected_head: Mapping[str, Any] | None = None) -> dict[str, Any]:
        prev_hash, count = GENESIS, 0
        for row in self.db.conn().execute("SELECT * FROM receipts ORDER BY seq"):
            count += 1
            problem = self._check(row, prev_hash, count)
            if problem:
                return {
                    "status": INTEGRITY_FAILED,
                    "checked": count,
                    "broken_at": row["receipt_id"],
                    "broken_seq": row["seq"],
                    "detail": problem,
                }
            prev_hash = row["digest"]
        if expected_head is not None:
            head = self.head()
            if head is None or head["seq"] != expected_head.get("seq") or head["digest"] != expected_head.get("digest"):
                return {
                    "status": INTEGRITY_FAILED,
                    "checked": count,
                    "broken_at": None,
                    "broken_seq": None,
                    "detail": "chain head does not match the anchored head",
                }
        return {"status": SEALED, "checked": count, "broken_at": None, "broken_seq": None, "detail": None}

    def _check(self, row, prev_hash: str, position: int) -> str | None:
        if row["seq"] != position:
            return "sequence gap: a receipt was removed or reordered"
        if row["prev_hash"] != prev_hash:
            return "previous-hash link does not match the preceding receipt"
        try:
            body = json.loads(row["body"])
        except ValueError:
            return "receipt body is not valid JSON"
        if chain_digest(prev_hash, body) != row["digest"]:
            return "receipt body does not match its digest"
        if not hmac.compare_digest(self._seal(row["seq"], row["digest"]), row["seal"]):
            return "seal was not produced with this gateway's key"
        if body.get("seq") != row["seq"] or body.get("receipt_id") != row["receipt_id"]:
            return "receipt identity does not match its body"
        for name in _INDEXED:
            if str(body.get(name, "")) != row[name]:
                return f"indexed column '{name}' does not match the sealed body"
        return None
