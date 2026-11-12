from __future__ import annotations

import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from nanogate.auth import RateLimiter
from nanogate.budget import BudgetLedger, Pricing
from nanogate.cache import SemanticCache
from nanogate.dlp import Scanner, new_entities
from nanogate.inference import (
    Backend,
    EgressDenied,
    EgressGuard,
    ModelUnavailable,
    RemoteSimulator,
    message_text,
    prompt_tokens,
)
from nanogate.models import (
    CacheLookup,
    Completion,
    Identity,
    PolicyDecision,
    Reason,
    Reservation,
    Route,
    RouterScore,
    ScanResult,
    Stage,
    degraded,
)
from nanogate.policy import PolicyEngine, effective_threshold
from nanogate.receipts import ReceiptLedger
from nanogate.router import RouterModel, build_features

ABSTAIN_TEXT = "NanoGate could not produce an answer it is confident in, and policy does not allow escalating this request."
BLOCKED_TEXT = "This request was blocked by NanoGate policy."
OUTPUT_BLOCKED_TEXT = "The generated answer was withheld because it contained a credential."
UNAVAILABLE_TEXT = "No model tier permitted for this request is currently available."

STATUS = {
    Reason.RATE_LIMITED: 429,
    Reason.BUDGET_DENY: 429,
    Reason.INPUT_TOO_LARGE: 413,
    Reason.DLP_FAIL_CLOSED: 503,
    Reason.MODEL_UNAVAILABLE: 503,
    Reason.SECRET_BLOCKED: 403,
    Reason.POLICY_BLOCK: 403,
    Reason.ROUTE_NOT_PERMITTED: 403,
}


@dataclass
class ChatRequest:
    messages: list[dict[str, Any]]
    pinned: Route | None = None
    requested_model: str = "nanogate-auto"
    max_tokens: int | None = None
    temperature: float = 0.0


@dataclass
class Outcome:
    request_id: str
    text: str
    route: Route
    reason: str
    reasons: list[str]
    status: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    receipt: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == 200

    def headers(self) -> dict[str, str]:
        return {
            "x-nanogate-request-id": self.request_id,
            "x-nanogate-receipt-id": self.receipt["receipt_id"],
            "x-nanogate-route": self.route.value,
            "x-nanogate-reason": self.reason,
            "x-nanogate-policy-version": str(self.receipt.get("policy_version", "")),
            "x-nanogate-data-class": str(self.receipt.get("data_class", "")),
            "x-nanogate-cost-usd": f"{self.cost_usd:.6f}",
        }


@dataclass
class _Trace:
    identity: Identity
    request: ChatRequest
    request_id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex}")
    started: float = field(default_factory=time.perf_counter)
    reasons: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    stages: list[Stage] = field(default_factory=list)
    scan: ScanResult = field(default_factory=ScanResult)
    decision: PolicyDecision | None = None
    reservation: Reservation | None = None
    lookup: CacheLookup | None = None
    score: RouterScore | None = None
    threshold: float | None = None
    completion: Completion | None = None
    egress: str = "none"
    input_redactions: int = 0
    output_redactions: int = 0
    prompt_tokens: int = 0

    def add(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)


class Gateway:
    def __init__(
        self,
        *,
        scanner: Scanner,
        policy: PolicyEngine,
        pricing: Pricing,
        budget: BudgetLedger,
        receipts: ReceiptLedger,
        cache: SemanticCache,
        backend: Backend,
        router: RouterModel,
        guard: EgressGuard,
        limiter: RateLimiter | None = None,
        max_prompt_chars: int = 32000,
    ):
        self.scanner = scanner
        self.policy = policy
        self.pricing = pricing
        self.budget = budget
        self.receipts = receipts
        self.cache = cache
        self.backend = backend
        self.router = router
        self.guard = guard
        self.remote = RemoteSimulator(guard)
        self.limiter = limiter or RateLimiter()
        self.max_prompt_chars = max_prompt_chars
        self.counters: Counter[tuple[str, str]] = Counter()
        self.latency_ms_total = 0.0

    async def handle(self, identity: Identity, request: ChatRequest) -> Outcome:
        trace = _Trace(identity, request)
        try:
            text, route = await self._decide(trace)
        except BaseException:
            self._release(trace)
            raise
        return self._finish(trace, text, route)

    def reject(self, identity: Identity | None, reason: str, status: int) -> Outcome:
        trace = _Trace(identity or Identity("unknown", "unknown", role="none"), ChatRequest(messages=[]))
        trace.add(reason)
        outcome = self._finish(trace, BLOCKED_TEXT, Route.BLOCKED)
        outcome.status = status
        return outcome

    def _timed(self, trace: _Trace, name: str, started: float, detail: str = "") -> None:
        trace.stages.append(Stage(name, round((time.perf_counter() - started) * 1000.0, 3), detail))

    async def _decide(self, trace: _Trace) -> tuple[str, Route]:
        identity, request = trace.identity, trace.request
        rules = self.policy.department(identity)

        if rules is not None and not self.limiter.allow(identity.key_id or f"{identity.tenant}/{identity.department}", rules.requests_per_minute):
            trace.add(Reason.RATE_LIMITED)
            return BLOCKED_TEXT, Route.BLOCKED

        messages = [dict(message) for message in request.messages]
        full_text = "\n".join(message_text(message) for message in messages)
        if len(full_text) > self.max_prompt_chars:
            trace.add(Reason.INPUT_TOO_LARGE)
            return BLOCKED_TEXT, Route.BLOCKED

        started = time.perf_counter()
        try:
            trace.scan = self.scanner.scan(full_text)
        except Exception:
            trace.degraded.append(degraded("dlp"))
            trace.add(Reason.DLP_FAIL_CLOSED)
            return BLOCKED_TEXT, Route.BLOCKED
        original_scan = trace.scan
        self._timed(trace, "classify", started, trace.scan.data_class.label)

        started = time.perf_counter()
        decision = self.policy.evaluate(identity, trace.scan, request.pinned, self.guard.enabled)
        if decision.redact_input:
            for message in messages:
                if message.get("role") != "assistant":
                    message["content"], count = self.scanner.redact(message_text(message))
                    trace.input_redactions += count
            trace.scan = self.scanner.scan("\n".join(message_text(message) for message in messages))
            decision = self.policy.evaluate(identity, trace.scan, request.pinned, self.guard.enabled, redacted=True)
        trace.decision = decision
        for reason in decision.reasons:
            trace.add(reason)
        self._timed(trace, "policy", started, ",".join(route.value for route in decision.allowed_routes) or "blocked")
        if decision.blocked:
            return BLOCKED_TEXT, Route.BLOCKED

        max_tokens = min(request.max_tokens or rules.max_output_tokens, rules.max_output_tokens)
        trace.prompt_tokens = prompt_tokens(messages)
        started = time.perf_counter()
        ceiling = self.pricing.ceiling(decision.allowed_routes, trace.prompt_tokens, max_tokens)
        remote_affordable = ceiling <= rules.per_request_usd
        trace.reservation = self.budget.reserve(identity, min(ceiling, rules.per_request_usd), rules.daily_usd)
        self._timed(trace, "budget", started, f"hold {trace.reservation.amount_usd:.4f}")
        if not trace.reservation.granted:
            trace.add(Reason.BUDGET_DENY)
            return BLOCKED_TEXT, Route.BLOCKED

        user_turns = [message for message in messages if message.get("role") == "user"]
        single_turn = len(user_turns) == 1 and all(message.get("role") != "assistant" for message in messages)
        question = message_text(user_turns[-1]) if user_turns else ""
        system_prompt = "\n".join(message_text(m) for m in messages if m.get("role") == "system")
        namespace = self.cache.namespace(identity, decision.policy_version, self.backend.model_name(False), system_prompt)

        if decision.permits(Route.CACHE) and single_turn:
            started = time.perf_counter()
            try:
                trace.lookup = self.cache.lookup(namespace, question)
            except Exception:
                trace.degraded.append(degraded("cache"))
                trace.lookup = CacheLookup(hit=False)
            self._timed(trace, "cache", started, trace.lookup.reason or "miss")
            if trace.lookup.hit:
                trace.add(Reason.CACHE_VERIFIED)
                return self._screen(trace, trace.lookup.answer or "", Route.CACHE, original_scan)
            if trace.lookup.reason:
                trace.add(trace.lookup.reason)

        confident = False
        for large in (False, True):
            route = Route.LOCAL_LARGE if large else Route.LOCAL_SMALL
            if not decision.permits(route) or (large and not self.backend.has_large):
                continue
            started = time.perf_counter()
            try:
                completion = await self.backend.complete(
                    messages, large=large, max_tokens=max_tokens, temperature=request.temperature
                )
            except ModelUnavailable:
                self._timed(trace, route.value, started, "unavailable")
                trace.degraded.append(degraded("inference"))
                continue
            self._timed(trace, route.value, started, completion.model)
            trace.completion = completion
            confident = self._score(trace, question, completion)
            if confident:
                trace.add(Reason.LOCAL_LARGE_SELECTED if large else Reason.LOCAL_CONFIDENT)
                break

        if not confident and decision.permits(Route.REMOTE) and remote_affordable:
            started = time.perf_counter()
            try:
                ticket = self.guard.issue(identity, trace.scan, decision, chars=len(full_text))
                trace.completion = await self.remote.complete(messages, ticket)
                trace.egress = self.guard.mode
                trace.add(Reason.REMOTE_ALLOWED)
                confident = True
            except EgressDenied:
                trace.add(Reason.REMOTE_DISABLED)
            self._timed(trace, "remote", started, trace.egress)

        if trace.completion is None:
            trace.add(Reason.MODEL_UNAVAILABLE)
            return UNAVAILABLE_TEXT, Route.ABSTAIN
        if not confident:
            if decision.abstain_on_low_confidence and trace.score is not None and trace.score.trained:
                trace.add(Reason.ROUTER_ABSTAIN)
                return ABSTAIN_TEXT, Route.ABSTAIN
            trace.add(Reason.LOCAL_UNVERIFIED)

        text, route = self._screen(trace, trace.completion.text, trace.completion.route, original_scan)
        writable = (
            confident
            and route in (Route.LOCAL_SMALL, Route.LOCAL_LARGE)
            and decision.cacheable
            and single_turn
            and not trace.input_redactions
            and not trace.output_redactions
        )
        if writable:
            try:
                self.cache.put(namespace, question, text)
            except Exception:
                trace.degraded.append(degraded("cache"))
        return text, route

    def _score(self, trace: _Trace, question: str, completion: Completion) -> bool:
        similarity = trace.lookup.similarity if trace.lookup else None
        try:
            trace.score = self.router.score(build_features(question, completion, trace.scan, similarity))
        except Exception:
            trace.score = RouterModel().score({})
        if not trace.score.trained and degraded("router") not in trace.degraded:
            trace.degraded.append(degraded("router"))
        trace.threshold = effective_threshold(trace.score.threshold, trace.decision.max_p_error)
        return trace.score.trained and trace.score.p_error <= trace.threshold

    def _screen(self, trace: _Trace, text: str, route: Route, source: ScanResult) -> tuple[str, Route]:
        started = time.perf_counter()
        try:
            output = self.scanner.scan(text)
        except Exception:
            trace.degraded.append(degraded("dlp"))
            trace.add(Reason.OUTPUT_BLOCKED)
            return OUTPUT_BLOCKED_TEXT, Route.BLOCKED
        if output.has_secret:
            trace.add(Reason.OUTPUT_BLOCKED)
            self._timed(trace, "output_scan", started, "blocked")
            return OUTPUT_BLOCKED_TEXT, Route.BLOCKED
        extra = new_entities(output, source)
        if extra:
            text, trace.output_redactions = self.scanner.redact(text, only=frozenset(extra))
            trace.add(Reason.OUTPUT_REDACTED)
        self._timed(trace, "output_scan", started, "redacted" if extra else "clean")
        return text, route

    def _release(self, trace: _Trace) -> None:
        if trace.reservation is not None and trace.reservation.granted:
            self.budget.release(trace.reservation)

    def _finish(self, trace: _Trace, text: str, route: Route) -> Outcome:
        completion = trace.completion
        answered = route in (Route.CACHE, Route.LOCAL_SMALL, Route.LOCAL_LARGE, Route.REMOTE)
        used_prompt = completion.prompt_tokens if completion else trace.prompt_tokens
        used_completion = completion.completion_tokens if completion and route != Route.CACHE else 0
        billed = completion.route if completion and route != Route.CACHE else route
        cost = self.pricing.cost(billed, used_prompt, used_completion) if completion and route != Route.CACHE else 0.0
        if trace.reservation is not None and trace.reservation.granted:
            self.budget.settle(trace.reservation, cost)

        reason = Reason.primary(trace.reasons)
        latency_ms = round((time.perf_counter() - trace.started) * 1000.0, 3)
        rate = self.pricing.rates.get(billed)
        score = trace.score
        receipt = self.receipts.append(
            {
                "request_id": trace.request_id,
                "tenant": trace.identity.tenant,
                "department": trace.identity.department,
                "key_id": trace.identity.key_id,
                "requested_model": trace.request.requested_model,
                "data_class": trace.scan.data_class.label,
                "entities": dict(sorted(trace.scan.entities.items())),
                "input_redactions": trace.input_redactions,
                "output_redactions": trace.output_redactions,
                "policy_version": trace.decision.policy_version if trace.decision else self.policy.version,
                "allowed_routes": [r.value for r in trace.decision.allowed_routes] if trace.decision else [],
                "route": route.value,
                "reason": reason,
                "reasons": list(trace.reasons),
                "degraded": list(trace.degraded),
                "cache": trace.lookup.summary() if trace.lookup else None,
                "router": None
                if score is None
                else {
                    "p_error": round(score.p_error, 6),
                    "threshold_learned": score.threshold,
                    "threshold_policy": trace.decision.max_p_error if trace.decision else None,
                    "threshold_applied": trace.threshold,
                    "top_factors": list(score.top_factors),
                    "trained": score.trained,
                },
                "model": completion.model if completion and route != Route.CACHE else None,
                "prompt_tokens": used_prompt,
                "completion_tokens": used_completion,
                "cost_usd": cost,
                "cost_basis": rate.basis if rate else "none",
                "egress": trace.egress,
                "latency_ms": latency_ms,
                "stages": [{"name": stage.name, "ms": stage.ms, "detail": stage.detail} for stage in trace.stages],
            }
        )
        self.counters[(route.value, reason)] += 1
        self.latency_ms_total += latency_ms
        status = 200 if answered or reason == Reason.ROUTER_ABSTAIN else STATUS.get(reason, 403)
        if reason == Reason.OUTPUT_BLOCKED:
            status = 200
        return Outcome(
            request_id=trace.request_id,
            text=text,
            route=route,
            reason=reason,
            reasons=list(trace.reasons),
            status=status,
            model=(completion.model if completion and answered and route != Route.CACHE else "nanogate"),
            prompt_tokens=used_prompt,
            completion_tokens=used_completion,
            cost_usd=cost,
            receipt=receipt,
        )
