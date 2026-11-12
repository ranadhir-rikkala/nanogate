from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE api_keys (
        key_id      TEXT PRIMARY KEY,
        key_hash    TEXT NOT NULL UNIQUE,
        tenant      TEXT NOT NULL,
        department  TEXT NOT NULL,
        role        TEXT NOT NULL,
        label       TEXT NOT NULL DEFAULT '',
        created_at  REAL NOT NULL,
        revoked_at  REAL
    );
    CREATE TABLE budget_ledger (
        reservation_id TEXT PRIMARY KEY,
        tenant         TEXT NOT NULL,
        department     TEXT NOT NULL,
        day            TEXT NOT NULL,
        reserved_usd   REAL NOT NULL,
        actual_usd     REAL,
        state          TEXT NOT NULL,
        created_at     REAL NOT NULL
    );
    CREATE INDEX idx_budget_day ON budget_ledger(tenant, department, day, state);
    CREATE TABLE receipts (
        seq          INTEGER PRIMARY KEY,
        receipt_id   TEXT NOT NULL UNIQUE,
        created_at   REAL NOT NULL,
        tenant       TEXT NOT NULL,
        department   TEXT NOT NULL,
        route        TEXT NOT NULL,
        reason       TEXT NOT NULL,
        data_class   TEXT NOT NULL,
        cost_usd     REAL NOT NULL,
        latency_ms   REAL NOT NULL,
        body         TEXT NOT NULL,
        prev_hash    TEXT NOT NULL,
        digest       TEXT NOT NULL,
        seal         TEXT NOT NULL
    );
    CREATE INDEX idx_receipts_scope ON receipts(tenant, department, created_at);
    CREATE TABLE cache_entries (
        entry_id    TEXT PRIMARY KEY,
        namespace   TEXT NOT NULL,
        prompt      TEXT NOT NULL,
        answer      TEXT NOT NULL,
        vector      BLOB NOT NULL,
        created_at  REAL NOT NULL,
        expires_at  REAL NOT NULL,
        hits        INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX idx_cache_ns ON cache_entries(namespace, expires_at);
    CREATE TABLE egress_log (
        ticket_id   TEXT PRIMARY KEY,
        created_at  REAL NOT NULL,
        tenant      TEXT NOT NULL,
        department  TEXT NOT NULL,
        data_class  TEXT NOT NULL,
        mode        TEXT NOT NULL,
        chars       INTEGER NOT NULL
    );
    """,
)


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._migrate_lock = threading.Lock()
        self._migrate()

    def conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            return existing
        conn = sqlite3.connect(self.path, timeout=10.0, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=10000")
        self._local.conn = conn
        return conn

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn().execute(sql, params).fetchall()

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn().execute(sql, params).fetchone()

    def close(self) -> None:
        existing = getattr(self._local, "conn", None)
        if existing is not None:
            existing.close()
            self._local.conn = None

    def _migrate(self) -> None:
        with self._migrate_lock:
            conn = self.conn()
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            for index in range(version, len(MIGRATIONS)):
                conn.executescript("BEGIN;" + MIGRATIONS[index] + f"PRAGMA user_version={index + 1};COMMIT;")
