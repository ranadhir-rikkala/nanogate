import json

import httpx
import pytest

from nanogate.config import Settings
from nanogate.dlp import Scanner
from nanogate.inference import (
    SIMULATED_NOTICE,
    EgressDenied,
    EgressGuard,
    ModelUnavailable,
    OfflineBackend,
    OpenAICompatBackend,
    RemoteSimulator,
    build_backend,
    message_text,
)
from nanogate.models import PolicyDecision, Route

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "How do I reset the VPN?"}]
SETTINGS = Settings(backend="openai", local_model="small-model", large_model="large-model", large_base_url="http://large/v1")
scanner = Scanner()


def backend_with(handler):
    return OpenAICompatBackend(SETTINGS, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


async def test_local_call_parses_text_usage_and_logprobs():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "small-model",
                "choices": [
                    {
                        "message": {"content": "Open Settings."},
                        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.7}]},
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    completion = await backend_with(handler).complete(MESSAGES, large=False, max_tokens=64, temperature=0.0)
    assert seen["url"] == "http://127.0.0.1:8000/v1/chat/completions"
    assert seen["body"]["logprobs"] is True and seen["body"]["model"] == "small-model"
    assert completion.text == "Open Settings." and completion.route == Route.LOCAL_SMALL
    assert (completion.prompt_tokens, completion.completion_tokens) == (12, 3)
    assert completion.logprobs == (-0.1, -0.7)


async def test_large_tier_uses_its_own_endpoint():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    completion = await backend_with(handler).complete(MESSAGES, large=True, max_tokens=64, temperature=0.0)
    assert seen["url"] == "http://large/v1/chat/completions"
    assert completion.route == Route.LOCAL_LARGE and completion.model == "large-model"
    assert completion.completion_tokens > 0


@pytest.mark.parametrize(
    "handler",
    [
        lambda request: httpx.Response(503, text="loading"),
        lambda request: httpx.Response(200, json={"choices": []}),
        lambda request: (_ for _ in ()).throw(httpx.ConnectError("refused")),
    ],
)
async def test_failures_become_model_unavailable(handler):
    with pytest.raises(ModelUnavailable):
        await backend_with(handler).complete(MESSAGES, large=False, max_tokens=64, temperature=0.0)


async def test_unconfigured_tier_is_unavailable():
    backend = OpenAICompatBackend(Settings(backend="openai", local_model="small"))
    assert backend.has_large is False
    with pytest.raises(ModelUnavailable):
        await backend.complete(MESSAGES, large=True, max_tokens=8, temperature=0.0)


async def test_offline_backend_is_labelled_and_never_echoes_the_prompt():
    completion = await OfflineBackend().complete(MESSAGES, large=False, max_tokens=8, temperature=0.0)
    assert completion.text.startswith("[offline backend")
    assert "VPN" not in completion.text
    assert (await OfflineBackend().health())["ok"] is True


def test_backend_factory():
    assert isinstance(build_backend(Settings()), OfflineBackend)
    assert isinstance(build_backend(SETTINGS), OpenAICompatBackend)
    with pytest.raises(ValueError):
        build_backend(Settings(backend="cloud"))


def test_message_text_handles_content_parts():
    assert message_text({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == "a\nb"
    assert message_text({"content": None}) == ""


def decision(*routes):
    return PolicyDecision(allowed_routes=tuple(routes), reasons=(), policy_version="v1")


async def test_simulated_remote_requires_a_ticket_and_logs_egress(db, it_user):
    guard = EgressGuard(db, mode="simulated")
    scan = scanner.scan("How do I reset the VPN?")
    ticket = guard.issue(it_user, scan, decision(Route.LOCAL_SMALL, Route.REMOTE), chars=23)
    completion = await RemoteSimulator(guard).complete(MESSAGES, ticket)
    assert completion.simulated is True and completion.text.startswith(SIMULATED_NOTICE)
    assert guard.log()[0]["mode"] == "simulated" and guard.log()[0]["chars"] == 23
    with pytest.raises(EgressDenied):
        await RemoteSimulator(guard).complete(MESSAGES, ticket)


def test_egress_is_refused_without_policy_permission_or_when_disabled(db, it_user):
    scan = scanner.scan("hello")
    with pytest.raises(EgressDenied):
        EgressGuard(db, mode="disabled").issue(it_user, scan, decision(Route.REMOTE), chars=5)
    with pytest.raises(EgressDenied):
        EgressGuard(db, mode="simulated").issue(it_user, scan, decision(Route.LOCAL_SMALL), chars=5)
    secret = scanner.scan("sk-abcdefghijklmnopqrstuvwx1234")
    with pytest.raises(EgressDenied):
        EgressGuard(db, mode="simulated").issue(it_user, secret, decision(Route.REMOTE), chars=5)
    assert EgressGuard(db, mode="simulated").log() == []
    with pytest.raises(ValueError):
        EgressGuard(db, mode="live")
