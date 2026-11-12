from __future__ import annotations

import asyncio
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from nanogate.config import Settings
from nanogate.models import Completion, Identity, Route
from nanogate.pipeline import ChatRequest
from nanogate.policy import MODEL_ALIASES
from nanogate.router import FEATURES, RouterModel


def _fixed_router(confident: bool) -> RouterModel:
    width = len(FEATURES)
    return RouterModel(
        {
            "features": list(FEATURES),
            "mean": [0.0] * width,
            "scale": [1.0] * width,
            "weights": [0.0] * width,
            "bias": -10.0 if confident else 10.0,
            "calibration": {"x": [], "y": []},
            "threshold": 0.25,
        }
    )


class PlantedBackend:
    has_large = True

    def __init__(self):
        self.reply = "Here is a routine answer."
        self.calls = 0

    def model_name(self, large: bool) -> str:
        return "attack-large" if large else "attack-small"

    async def complete(self, messages, *, large, max_tokens, temperature) -> Completion:
        self.calls += 1
        return Completion(
            text=self.reply,
            model=self.model_name(large),
            route=Route.LOCAL_LARGE if large else Route.LOCAL_SMALL,
            prompt_tokens=8,
            completion_tokens=8,
        )

    async def health(self) -> dict[str, Any]:
        return {"ok": True, "backend": "attack"}


def _check(case: dict[str, Any], outcome, backend: PlantedBackend, egress_rows: int) -> str | None:
    expect = case["expect"]
    if "route" in expect and outcome.route.value != expect["route"]:
        return f"route was {outcome.route.value}, expected {expect['route']}"
    if "route_in" in expect and outcome.route.value not in expect["route_in"]:
        return f"route was {outcome.route.value}, expected one of {expect['route_in']}"
    if "reason" in expect and outcome.reason != expect["reason"]:
        return f"reason was {outcome.reason}, expected {expect['reason']}"
    if "has_reason" in expect and expect["has_reason"] not in outcome.reasons:
        return f"{expect['has_reason']} missing from {outcome.reasons}"
    if "data_class" in expect and outcome.receipt["data_class"] != expect["data_class"]:
        return f"data class was {outcome.receipt['data_class']}, expected {expect['data_class']}"
    if expect.get("model_not_called") and backend.calls:
        return "a model was called for a request that should have been stopped first"
    if expect.get("no_egress") and (egress_rows or outcome.receipt["egress"] != "none"):
        return "request reached the remote tier"
    for needle in expect.get("absent_from_response", []):
        if needle in outcome.text:
            return "response leaked a value that should have been withheld"
    for needle in case.get("sensitive_values", []):
        if needle in json.dumps(outcome.receipt):
            return "receipt stored a raw sensitive value"
    return None


async def _run(cases: list[dict[str, Any]], settings: Settings) -> dict[str, Any]:
    from nanogate.api import build_services

    failures = []
    with tempfile.TemporaryDirectory() as scratch:
        for index, case in enumerate(cases):
            isolated = replace(
                settings,
                db_path=str(Path(scratch) / f"case{index}.db"),
                remote_mode=case.get("remote_mode", "simulated"),
                router_artifact=str(Path(scratch) / "absent.json"),
                receipt_key="attack-suite",
                key_pepper="attack-suite",
            )
            services = build_services(isolated)
            backend = PlantedBackend()
            services.gateway.backend = backend
            services.gateway.router = _fixed_router(True)
            identity = Identity(case.get("tenant", "acme"), case.get("department", "it"), key_id=f"case{index}")
            for setup in case.get("setup", []):
                backend.reply = setup.get("model_reply", backend.reply)
                seeder = Identity(setup.get("tenant", identity.tenant), setup.get("department", identity.department))
                await services.gateway.handle(seeder, ChatRequest(messages=[{"role": "user", "content": setup["prompt"]}]))
            backend.calls = 0
            backend.reply = case.get("model_reply", "Here is a routine answer.")
            services.gateway.router = _fixed_router(not case.get("model_unsure", False))
            model = case.get("model", "nanogate-auto")
            outcome = await services.gateway.handle(
                identity,
                ChatRequest(
                    messages=[{"role": "user", "content": case["prompt"]}],
                    pinned=MODEL_ALIASES[model],
                    requested_model=model,
                ),
            )
            why = _check(case, outcome, backend, len(services.gateway.guard.log()))
            if why is None and services.gateway.receipts.verify()["status"] != "SEALED":
                why = "receipt chain did not verify"
            if why:
                failures.append({"id": case["id"], "why": why})
            services.db.close()
    return {"total": len(cases), "passed": len(cases) - len(failures), "failures": failures}


def run_suite(path: str | Path, settings: Settings) -> dict[str, Any]:
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return asyncio.run(_run(cases, settings))
