from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from nanogate.db import Database
from nanogate.models import Identity, Reservation, Route

OPEN, SETTLED, RELEASED = "open", "settled", "released"


@dataclass(frozen=True)
class Rate:
    prompt_per_1k: float
    completion_per_1k: float
    basis: str


class Pricing:
    def __init__(self, document: Mapping[str, Any]):
        self.version = str(document.get("version", "0"))
        self.rates: dict[Route, Rate] = {}
        for name, body in (document.get("routes") or {}).items():
            self.rates[Route(name)] = Rate(
                float(body.get("usd_per_1k_prompt_tokens", 0.0)),
                float(body.get("usd_per_1k_completion_tokens", 0.0)),
                str(body.get("basis", "assumed")),
            )

    @classmethod
    def from_file(cls, path: str | Path) -> "Pricing":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle) or {})

    def cost(self, route: Route, prompt_tokens: int, completion_tokens: int) -> float:
        rate = self.rates.get(route)
        if rate is None:
            return 0.0
        return round(
            prompt_tokens / 1000.0 * rate.prompt_per_1k + completion_tokens / 1000.0 * rate.completion_per_1k, 8
        )

    def ceiling(self, routes: Iterable[Route], prompt_tokens: int, max_output_tokens: int) -> float:
        return max((self.cost(route, prompt_tokens, max_output_tokens) for route in routes), default=0.0)


class BudgetLedger:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock

    def _day(self) -> str:
        return datetime.fromtimestamp(self.clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def _committed(self, conn, identity: Identity, day: str) -> float:
        row = conn.execute(
            "SELECT COALESCE(SUM(CASE state WHEN 'settled' THEN actual_usd WHEN 'open' THEN reserved_usd"
            " ELSE 0 END), 0) FROM budget_ledger WHERE tenant = ? AND department = ? AND day = ?",
            (identity.tenant, identity.department, day),
        ).fetchone()
        return float(row[0])

    def reserve(self, identity: Identity, amount_usd: float, daily_limit_usd: float) -> Reservation:
        amount_usd = max(0.0, float(amount_usd))
        day = self._day()
        with self.db.write() as conn:
            committed = self._committed(conn, identity, day)
            remaining = daily_limit_usd - committed
            if committed >= daily_limit_usd or amount_usd > remaining + 1e-12:
                return Reservation(granted=False, amount_usd=amount_usd, remaining_usd=max(0.0, remaining))
            reservation_id = uuid.uuid4().hex
            conn.execute(
                "INSERT INTO budget_ledger(reservation_id, tenant, department, day, reserved_usd, state, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (reservation_id, identity.tenant, identity.department, day, amount_usd, OPEN, self.clock()),
            )
        return Reservation(True, reservation_id, amount_usd, remaining - amount_usd)

    def settle(self, reservation: Reservation, actual_usd: float) -> bool:
        return self._close(reservation, SETTLED, max(0.0, float(actual_usd)))

    def release(self, reservation: Reservation) -> bool:
        return self._close(reservation, RELEASED, 0.0)

    def _close(self, reservation: Reservation, state: str, actual_usd: float) -> bool:
        if not reservation.granted or reservation.reservation_id is None:
            return False
        with self.db.write() as conn:
            cursor = conn.execute(
                "UPDATE budget_ledger SET state = ?, actual_usd = ? WHERE reservation_id = ? AND state = ?",
                (state, actual_usd, reservation.reservation_id, OPEN),
            )
        return cursor.rowcount == 1

    def spent_today(self, identity: Identity) -> float:
        return self._committed(self.db.conn(), identity, self._day())

    def by_department(self, day: str | None = None) -> list[dict]:
        rows = self.db.query(
            "SELECT tenant, department,"
            " COALESCE(SUM(CASE WHEN state = 'settled' THEN actual_usd ELSE 0 END), 0) AS spent_usd,"
            " COALESCE(SUM(CASE WHEN state = 'open' THEN reserved_usd ELSE 0 END), 0) AS held_usd,"
            " COUNT(*) AS requests"
            " FROM budget_ledger WHERE day = ? GROUP BY tenant, department ORDER BY tenant, department",
            (day or self._day(),),
        )
        return [dict(row) for row in rows]
