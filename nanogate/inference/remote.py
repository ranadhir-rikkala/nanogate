from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

from nanogate.db import Database
from nanogate.inference.base import Messages, estimate_tokens, message_text, prompt_tokens
from nanogate.models import Completion, Identity, PolicyDecision, Route, ScanResult

MODES = ("disabled", "simulated")
SIMULATED_NOTICE = "[simulated remote tier: nothing left this machine]"


class EgressDenied(Exception):
    pass


@dataclass(frozen=True)
class EgressTicket:
    ticket_id: str
    tenant: str
    department: str
    data_class: str


class EgressGuard:
    def __init__(self, db: Database, mode: str = "disabled"):
        if mode not in MODES:
            raise ValueError(f"remote mode must be one of {MODES}")
        self.db = db
        self.mode = mode
        self._open: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.mode != "disabled"

    def issue(self, identity: Identity, scan: ScanResult, decision: PolicyDecision, chars: int) -> EgressTicket:
        if not self.enabled:
            raise EgressDenied("remote tier is disabled")
        if not decision.permits(Route.REMOTE):
            raise EgressDenied("policy does not permit the remote route for this request")
        if scan.has_secret:
            raise EgressDenied("requests containing credentials never leave the device")
        ticket = EgressTicket(uuid.uuid4().hex, identity.tenant, identity.department, scan.data_class.label)
        with self.db.write() as conn:
            conn.execute(
                "INSERT INTO egress_log(ticket_id, created_at, tenant, department, data_class, mode, chars)"
                " VALUES (?,?,?,?,?,?,?)",
                (ticket.ticket_id, time.time(), ticket.tenant, ticket.department, ticket.data_class, self.mode, chars),
            )
        self._open.add(ticket.ticket_id)
        return ticket

    def redeem(self, ticket: EgressTicket) -> None:
        if ticket.ticket_id not in self._open:
            raise EgressDenied("egress ticket is unknown or already used")
        self._open.discard(ticket.ticket_id)

    def log(self, limit: int = 50) -> list[dict]:
        rows = self.db.query("SELECT * FROM egress_log ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(row) for row in rows]


class RemoteSimulator:
    model = "remote-simulator"

    def __init__(self, guard: EgressGuard):
        self.guard = guard

    async def complete(self, messages: Messages, ticket: EgressTicket) -> Completion:
        self.guard.redeem(ticket)
        chars = sum(len(message_text(message)) for message in messages)
        text = f"{SIMULATED_NOTICE} A hosted tier would have received {chars} characters."
        return Completion(
            text=text,
            model=self.model,
            route=Route.REMOTE,
            prompt_tokens=prompt_tokens(messages),
            completion_tokens=estimate_tokens(text),
            simulated=True,
        )
