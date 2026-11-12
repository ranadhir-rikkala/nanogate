import pytest

from nanogate.db import Database
from nanogate.models import Identity


@pytest.fixture
def db(tmp_path):
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture
def it_user():
    return Identity(tenant="acme", department="it", role="app", key_id="k_it")


@pytest.fixture
def hr_user():
    return Identity(tenant="acme", department="hr", role="app", key_id="k_hr")


from pathlib import Path

from nanogate.auth import RateLimiter
from nanogate.budget import BudgetLedger, Pricing
from nanogate.cache import SemanticCache
from nanogate.dlp import Scanner
from nanogate.inference import EgressGuard, ModelUnavailable
from nanogate.inference.base import estimate_tokens, prompt_tokens
from nanogate.models import Completion, Route
from nanogate.pipeline import Gateway
from nanogate.policy import PolicyEngine
from nanogate.receipts import ReceiptLedger
from nanogate.router import FEATURES, RouterModel

ROOT = Path(__file__).parent.parent
SURE, UNSURE = (-0.05, -0.1), (-2.0, -2.5)
ROUTER_ARTIFACT = {
    "features": list(FEATURES),
    "mean": [0.0] * len(FEATURES),
    "scale": [1.0] * len(FEATURES),
    "weights": [-4.0 if name == "logprob_mean" else 0.0 for name in FEATURES],
    "bias": -3.0,
    "calibration": {"x": [], "y": []},
    "threshold": 0.25,
}


class ScriptedBackend:
    def __init__(self, small=("Open Settings and choose Reset.", SURE), large=("Large tier answer.", SURE), has_large=True):
        self.small, self.large, self.has_large = small, large, has_large
        self.calls = []

    def model_name(self, large):
        return "scripted-large" if large else "scripted-small"

    async def complete(self, messages, *, large, max_tokens, temperature):
        self.calls.append(("large" if large else "small", messages))
        script = self.large if large else self.small
        if script is None:
            raise ModelUnavailable("down")
        text, logprobs = script
        return Completion(
            text=text,
            model=self.model_name(large),
            route=Route.LOCAL_LARGE if large else Route.LOCAL_SMALL,
            prompt_tokens=prompt_tokens(messages),
            completion_tokens=estimate_tokens(text),
            logprobs=logprobs,
        )

    async def health(self):
        return {"ok": True, "backend": "scripted"}


def build_gateway(db, backend=None, remote_mode="disabled", trained=True, **overrides):
    policy = PolicyEngine.from_file(ROOT / "config" / "policy.yaml")
    parts = dict(
        scanner=Scanner(policy.classification),
        policy=policy,
        pricing=Pricing.from_file(ROOT / "config" / "rates.yaml"),
        budget=BudgetLedger(db),
        receipts=ReceiptLedger(db, key="test-key"),
        cache=SemanticCache(db),
        backend=backend or ScriptedBackend(),
        router=RouterModel(ROUTER_ARTIFACT if trained else None),
        guard=EgressGuard(db, mode=remote_mode),
        limiter=RateLimiter(),
    )
    parts.update(overrides)
    return Gateway(**parts)
