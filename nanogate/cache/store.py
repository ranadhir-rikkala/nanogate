from __future__ import annotations

import hashlib
import time
import uuid
from typing import Callable

from nanogate.cache.embedding import Embedder, HashingEmbedder, cosine, pack, unpack
from nanogate.cache.verifier import SlotVerifier
from nanogate.db import Database
from nanogate.models import CacheLookup, Identity, Reason, canonical_json

CANDIDATES = 3


def namespace(identity: Identity, policy_version: str, model: str, system_prompt: str = "", embedder: str = "") -> str:
    material = canonical_json(
        {
            "tenant": identity.tenant,
            "department": identity.department,
            "policy": policy_version,
            "model": model,
            "system": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
            "embedder": embedder,
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


class SemanticCache:
    def __init__(
        self,
        db: Database,
        embedder: Embedder | None = None,
        verifier: SlotVerifier | None = None,
        similarity: float = 0.70,
        ttl_s: int = 86400,
        clock: Callable[[], float] = time.time,
    ):
        self.db = db
        self.embedder = embedder or HashingEmbedder()
        self.verifier = verifier or SlotVerifier()
        self.similarity = similarity
        self.ttl_s = ttl_s
        self.clock = clock

    def namespace(self, identity: Identity, policy_version: str, model: str, system_prompt: str = "") -> str:
        return namespace(identity, policy_version, model, system_prompt, self.embedder.name)

    def lookup(self, ns: str, prompt: str) -> CacheLookup:
        vector = self.embedder.embed(prompt)
        rows = self.db.query(
            "SELECT entry_id, prompt, answer, vector FROM cache_entries WHERE namespace = ? AND expires_at > ?",
            (ns, self.clock()),
        )
        scored = sorted(((cosine(vector, unpack(row["vector"])), row) for row in rows), key=lambda pair: -pair[0])
        if not scored:
            return CacheLookup(hit=False)
        best = scored[0][0]
        rejected = None
        for score, row in scored[:CANDIDATES]:
            if score < self.similarity:
                break
            verdict = self.verifier.verify(prompt, row["prompt"])
            if verdict.ok:
                self.db.conn().execute("UPDATE cache_entries SET hits = hits + 1 WHERE entry_id = ?", (row["entry_id"],))
                return CacheLookup(True, row["answer"], score, verdict.score, Reason.CACHE_VERIFIED, row["entry_id"])
            rejected = rejected or (score, verdict)
        if rejected is not None:
            return CacheLookup(False, None, rejected[0], rejected[1].score, Reason.CACHE_HARD_NEGATIVE)
        return CacheLookup(hit=False, similarity=best)

    def put(self, ns: str, prompt: str, answer: str) -> str:
        now = self.clock()
        entry_id = uuid.uuid4().hex
        with self.db.write() as conn:
            conn.execute("DELETE FROM cache_entries WHERE namespace = ? AND prompt = ?", (ns, prompt))
            conn.execute(
                "INSERT INTO cache_entries(entry_id, namespace, prompt, answer, vector, created_at, expires_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (entry_id, ns, prompt, answer, pack(self.embedder.embed(prompt)), now, now + self.ttl_s),
            )
        return entry_id

    def purge(self, ns: str | None = None, expired_only: bool = False) -> int:
        clauses, params = [], []
        if ns is not None:
            clauses.append("namespace = ?")
            params.append(ns)
        if expired_only:
            clauses.append("expires_at <= ?")
            params.append(self.clock())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self.db.write() as conn:
            return conn.execute(f"DELETE FROM cache_entries{where}", tuple(params)).rowcount

    def stats(self) -> dict:
        row = self.db.one(
            "SELECT COUNT(*) AS entries, COUNT(DISTINCT namespace) AS namespaces, COALESCE(SUM(hits), 0) AS hits"
            " FROM cache_entries WHERE expires_at > ?",
            (self.clock(),),
        )
        return {**dict(row), "embedder": self.embedder.name, "similarity_threshold": self.similarity}
