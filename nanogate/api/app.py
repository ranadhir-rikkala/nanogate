from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse

from nanogate import __version__
from nanogate.auth import AuthError, KeyStore, RateLimiter, bearer_token, reject_spoofing, require_role
from nanogate.budget import BudgetLedger, Pricing
from nanogate.cache import SemanticCache
from nanogate.config import Settings
from nanogate.db import Database
from nanogate.dlp import Scanner
from nanogate.inference import EgressGuard, build_backend
from nanogate.models import Identity
from nanogate.pipeline import ChatRequest, Gateway, Outcome
from nanogate.policy import MODEL_ALIASES, PolicyEngine
from nanogate.receipts import ReceiptLedger
from nanogate.router import RouterModel

WEB_DIR = Path(__file__).parent.parent / "web"


@dataclass
class Services:
    settings: Settings
    db: Database
    keys: KeyStore
    gateway: Gateway


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict[str, str] | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def load_or_create_secret(configured: str, path: Path) -> str:
    if configured:
        return configured
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    value = secrets.token_hex(32)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(value)
    return value


def build_services(settings: Settings) -> Services:
    db = Database(settings.db_path)
    state_dir = Path(settings.db_path).parent
    policy = PolicyEngine.from_file(settings.policy_path)
    guard = EgressGuard(db, mode=settings.remote_mode)
    gateway = Gateway(
        scanner=Scanner(policy.classification),
        policy=policy,
        pricing=Pricing.from_file(settings.rates_path),
        budget=BudgetLedger(db),
        receipts=ReceiptLedger(db, load_or_create_secret(settings.receipt_key, state_dir / "receipt.key")),
        cache=SemanticCache(db, similarity=settings.cache_similarity, ttl_s=settings.cache_ttl_s),
        backend=build_backend(settings),
        router=RouterModel.from_file(settings.router_artifact),
        guard=guard,
        limiter=RateLimiter(),
        max_prompt_chars=settings.max_prompt_chars,
    )
    keys = KeyStore(db, load_or_create_secret(settings.key_pepper, state_dir / "key.pepper"))
    return Services(settings, db, keys, gateway)


def chat_body(outcome: Outcome, requested_model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{outcome.request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": requested_model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": outcome.text}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "total_tokens": outcome.prompt_tokens + outcome.completion_tokens,
        },
        "nanogate": {
            "route": outcome.route.value,
            "reason": outcome.reason,
            "reasons": outcome.reasons,
            "served_by": outcome.model,
            "receipt_id": outcome.receipt["receipt_id"],
        },
    }


def stream_chunks(outcome: Outcome, requested_model: str):
    base = {"id": f"chatcmpl-{outcome.request_id}", "object": "chat.completion.chunk", "created": int(time.time()), "model": requested_model}
    pieces = outcome.text.split(" ")
    deltas = [{"role": "assistant", "content": ""}] + [
        {"content": piece if index == 0 else f" {piece}"} for index, piece in enumerate(pieces)
    ]
    for delta in deltas:
        yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]})}\n\n"
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    services = services or build_services(settings or Settings.from_env())
    gateway, keys = services.gateway, services.keys
    app = FastAPI(title="NanoGate", version=__version__, docs_url="/docs", redoc_url=None)
    app.state.services = services

    @app.exception_handler(ApiError)
    async def api_error(_: Request, error: ApiError) -> JSONResponse:
        body = {"error": {"message": error.message, "type": "nanogate_error", "code": error.code}}
        return JSONResponse(body, status_code=error.status, headers=error.headers)

    def authenticate(request: Request) -> Identity:
        try:
            identity = keys.resolve(bearer_token(request.headers))
            reject_spoofing(identity, request.headers)
            return identity
        except AuthError as error:
            outcome = gateway.reject(None, error.code, error.status)
            raise ApiError(error.status, error.code, error.message, outcome.headers()) from error

    def with_roles(*roles: str):
        def dependency(identity: Identity = Depends(authenticate)) -> Identity:
            try:
                require_role(identity, *roles)
            except AuthError as error:
                outcome = gateway.reject(identity, error.code, error.status)
                raise ApiError(error.status, error.code, error.message, outcome.headers()) from error
            return identity

        return dependency

    caller = with_roles("app", "admin")
    reader = with_roles("auditor", "admin")
    admin = with_roles("admin")

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        model = await gateway.backend.health()
        checks = {
            "model": model,
            "router_trained": gateway.router.trained,
            "remote_mode": gateway.guard.mode,
            "policy_version": gateway.policy.version,
        }
        return JSONResponse({"ready": bool(model.get("ok")), **checks}, status_code=200 if model.get("ok") else 503)

    @app.get("/metrics")
    async def metrics() -> PlainTextResponse:
        lines = ["# TYPE nanogate_requests_total counter"]
        for (route, reason), count in sorted(gateway.counters.items()):
            lines.append(f'nanogate_requests_total{{route="{route}",reason="{reason}"}} {count}')
        lines += ["# TYPE nanogate_latency_ms_total counter", f"nanogate_latency_ms_total {gateway.latency_ms_total:.3f}"]
        return PlainTextResponse("\n".join(lines) + "\n")

    @app.get("/v1/models")
    async def models(_: Identity = Depends(caller)) -> dict[str, Any]:
        return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "nanogate"} for name in MODEL_ALIASES]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, identity: Identity = Depends(caller)):
        try:
            body = await request.json()
        except ValueError as error:
            raise ApiError(400, "invalid_json", "Request body must be JSON.") from error
        messages = body.get("messages") if isinstance(body, dict) else None
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and "role" in m for m in messages):
            raise ApiError(400, "invalid_messages", "'messages' must be a non-empty list of {role, content} objects.")
        model = str(body.get("model") or "nanogate-auto")
        if model not in MODEL_ALIASES:
            raise ApiError(400, "unknown_model", f"Unknown model '{model}'. Use one of: {', '.join(MODEL_ALIASES)}.")
        limit = body.get("max_completion_tokens") or body.get("max_tokens")
        try:
            chat_request = ChatRequest(
                messages=messages,
                pinned=MODEL_ALIASES[model],
                requested_model=model,
                max_tokens=int(limit) if limit else None,
                temperature=float(body.get("temperature") or 0.0),
            )
        except (TypeError, ValueError) as error:
            raise ApiError(400, "invalid_parameter", "max_tokens and temperature must be numbers.") from error
        outcome = await gateway.handle(identity, chat_request)
        if not outcome.ok:
            raise ApiError(outcome.status, outcome.reason, outcome.text, outcome.headers())
        if body.get("stream"):
            return StreamingResponse(stream_chunks(outcome, model), media_type="text/event-stream", headers=outcome.headers())
        return JSONResponse(chat_body(outcome, model), headers=outcome.headers())

    @app.get("/api/overview")
    async def overview(identity: Identity = Depends(reader)) -> dict[str, Any]:
        db = services.db
        scope = (identity.tenant,)
        totals = db.one(
            "SELECT COUNT(*) AS requests, COALESCE(SUM(cost_usd), 0) AS cost_usd, COALESCE(AVG(latency_ms), 0) AS avg_latency_ms"
            " FROM receipts WHERE tenant = ?",
            scope,
        )
        by_route = db.query("SELECT route, COUNT(*) AS n FROM receipts WHERE tenant = ? GROUP BY route ORDER BY n DESC", scope)
        by_reason = db.query("SELECT reason, COUNT(*) AS n FROM receipts WHERE tenant = ? GROUP BY reason ORDER BY n DESC", scope)
        by_class = db.query("SELECT data_class, COUNT(*) AS n FROM receipts WHERE tenant = ? GROUP BY data_class", scope)
        return {
            "tenant": identity.tenant,
            **dict(totals),
            "by_route": {row["route"]: row["n"] for row in by_route},
            "by_reason": {row["reason"]: row["n"] for row in by_reason},
            "by_data_class": {row["data_class"]: row["n"] for row in by_class},
            "budget_today": [row for row in gateway.budget.by_department() if row["tenant"] == identity.tenant],
            "cache": gateway.cache.stats(),
            "router": {"trained": gateway.router.trained, "threshold": gateway.router.threshold},
            "remote_mode": gateway.guard.mode,
            "policy_version": gateway.policy.version,
            "chain_head": gateway.receipts.head(),
        }

    @app.get("/api/receipts")
    async def receipts(limit: int = 50, department: str | None = None, identity: Identity = Depends(reader)) -> dict[str, Any]:
        rows = gateway.receipts.recent(max(1, min(limit, 500)), tenant=identity.tenant, department=department)
        return {"receipts": rows}

    @app.get("/api/receipts/{receipt_id}")
    async def receipt(receipt_id: str, identity: Identity = Depends(reader)) -> dict[str, Any]:
        found = gateway.receipts.get(receipt_id)
        if found is None or found["tenant"] != identity.tenant:
            raise ApiError(404, "not_found", "No such receipt.")
        return found

    @app.get("/api/chain/verify")
    async def verify(_: Identity = Depends(reader)) -> dict[str, Any]:
        return {**gateway.receipts.verify(), "head": gateway.receipts.head()}

    @app.get("/api/policy")
    async def policy(identity: Identity = Depends(reader)) -> dict[str, Any]:
        tenant = (gateway.policy.document.get("tenants") or {}).get(identity.tenant, {})
        return {
            "version": gateway.policy.version,
            "defaults": gateway.policy.document.get("defaults", {}),
            "classification": gateway.policy.document.get("classification", {}),
            "departments": tenant.get("departments", {}),
        }

    @app.post("/api/policy/test")
    async def policy_test(request: Request, identity: Identity = Depends(reader)) -> dict[str, Any]:
        body = await request.json()
        model = str(body.get("model") or "nanogate-auto")
        if model not in MODEL_ALIASES:
            raise ApiError(400, "unknown_model", f"Unknown model '{model}'.")
        subject = Identity(identity.tenant, str(body.get("department") or identity.department).lower())
        scan = gateway.scanner.scan(str(body.get("text") or ""))
        decision = gateway.policy.evaluate(subject, scan, MODEL_ALIASES[model], gateway.guard.enabled)
        return {
            "scan": scan.summary(),
            "allowed_routes": [route.value for route in decision.allowed_routes],
            "blocked": decision.blocked,
            "reasons": list(decision.reasons),
            "would_redact_input": decision.redact_input,
            "cacheable": decision.cacheable,
            "max_p_error": decision.max_p_error,
            "policy_version": decision.policy_version,
        }

    @app.get("/api/egress")
    async def egress(identity: Identity = Depends(reader)) -> dict[str, Any]:
        return {"mode": gateway.guard.mode, "log": [row for row in gateway.guard.log(200) if row["tenant"] == identity.tenant]}

    @app.get("/api/keys")
    async def list_keys(identity: Identity = Depends(admin)) -> dict[str, Any]:
        return {"keys": [row for row in keys.list() if row["tenant"] == identity.tenant]}

    @app.post("/api/keys")
    async def create_key(request: Request, identity: Identity = Depends(admin)) -> JSONResponse:
        body = await request.json()
        try:
            key_id, key = keys.create(
                identity.tenant, str(body.get("department", "")), str(body.get("role", "app")), str(body.get("label", ""))
            )
        except ValueError as error:
            raise ApiError(400, "invalid_key_request", str(error)) from error
        return JSONResponse({"key_id": key_id, "key": key, "note": "Shown once. Store it now."}, status_code=201)

    @app.delete("/api/keys/{key_id}")
    async def revoke_key(key_id: str, identity: Identity = Depends(admin)) -> dict[str, Any]:
        owned = any(row["key_id"] == key_id and row["tenant"] == identity.tenant for row in keys.list())
        if not owned or not keys.revoke(key_id):
            raise ApiError(404, "not_found", "No active key with that id in your tenant.")
        return {"revoked": key_id}

    @app.delete("/api/cache")
    async def purge_cache(_: Identity = Depends(admin)) -> dict[str, Any]:
        return {"removed": gateway.cache.purge()}

    @app.get("/", include_in_schema=False)
    async def dashboard():
        index = WEB_DIR / "index.html"
        if not index.is_file():
            return PlainTextResponse("NanoGate is running. Dashboard assets are not installed.")
        return FileResponse(index)

    return app
