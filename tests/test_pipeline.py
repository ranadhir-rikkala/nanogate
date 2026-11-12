import pytest

from conftest import SURE, UNSURE, ScriptedBackend, build_gateway
from nanogate.inference import SIMULATED_NOTICE
from nanogate.models import Identity, Reason, Route
from nanogate.pipeline import ChatRequest
from nanogate.receipts import SEALED

QUESTION = "How do I reset the VPN client?"


def ask(text=QUESTION, **kwargs):
    return ChatRequest(messages=[{"role": "user", "content": text}], **kwargs)


async def test_confident_local_answer_is_returned_and_receipted(db, it_user):
    gateway = build_gateway(db)
    outcome = await gateway.handle(it_user, ask())
    assert outcome.status == 200 and outcome.route == Route.LOCAL_SMALL
    assert outcome.reason == Reason.LOCAL_CONFIDENT
    receipt = outcome.receipt
    assert receipt["router"]["p_error"] < receipt["router"]["threshold_applied"] == 0.25
    assert receipt["egress"] == "none" and receipt["cost_usd"] == 0.0
    assert [stage["name"] for stage in receipt["stages"]] == [
        "classify", "policy", "budget", "cache", "local_small", "output_scan",
    ]
    assert outcome.headers()["x-nanogate-route"] == "local_small"
    assert gateway.receipts.verify()["status"] == SEALED


async def test_second_ask_is_served_from_the_verified_cache(db, it_user):
    backend = ScriptedBackend()
    gateway = build_gateway(db, backend)
    first = await gateway.handle(it_user, ask())
    second = await gateway.handle(it_user, ask("how can I reset my VPN client"))
    assert second.route == Route.CACHE and second.reason == Reason.CACHE_VERIFIED
    assert second.text == first.text and len(backend.calls) == 1
    assert second.receipt["model"] is None and second.completion_tokens == 0


async def test_cache_never_crosses_departments_or_tenants(db, it_user, hr_user):
    backend = ScriptedBackend()
    gateway = build_gateway(db, backend)
    await gateway.handle(it_user, ask())
    for other in (hr_user, Identity("globex", "it")):
        assert (await gateway.handle(other, ask())).route == Route.LOCAL_SMALL
    assert len(backend.calls) == 3


async def test_hard_negative_goes_to_the_model_not_the_cache(db, it_user):
    backend = ScriptedBackend()
    gateway = build_gateway(db, backend)
    await gateway.handle(it_user, ask("Can contractors access the wiki?"))
    outcome = await gateway.handle(it_user, ask("Can contractors not access the wiki?"))
    assert outcome.route == Route.LOCAL_SMALL and len(backend.calls) == 2
    assert Reason.CACHE_HARD_NEGATIVE in outcome.reasons


async def test_low_confidence_escalates_to_the_large_tier(db, it_user):
    backend = ScriptedBackend(small=("Maybe?", UNSURE))
    outcome = await build_gateway(db, backend).handle(it_user, ask())
    assert outcome.route == Route.LOCAL_LARGE and outcome.reason == Reason.LOCAL_LARGE_SELECTED
    assert [call[0] for call in backend.calls] == ["small", "large"]
    assert Reason.REMOTE_DISABLED in outcome.reasons


async def test_remote_is_used_only_when_enabled_and_is_labelled_simulated(db, it_user):
    backend = ScriptedBackend(small=("Maybe?", UNSURE), large=("Hmm.", UNSURE))
    gateway = build_gateway(db, backend, remote_mode="simulated")
    outcome = await gateway.handle(it_user, ask())
    assert outcome.route == Route.REMOTE and outcome.reason == Reason.REMOTE_ALLOWED
    assert outcome.text.startswith(SIMULATED_NOTICE)
    assert outcome.receipt["egress"] == "simulated" and outcome.cost_usd > 0
    assert outcome.receipt["cost_basis"] == "assumed"
    assert len(gateway.guard.log()) == 1
    assert round(gateway.budget.spent_today(it_user), 8) == outcome.cost_usd


async def test_sensitive_request_never_leaves_even_when_unsure(db, it_user):
    backend = ScriptedBackend(small=("Maybe?", UNSURE), large=("Hmm.", UNSURE))
    gateway = build_gateway(db, backend, remote_mode="simulated")
    outcome = await gateway.handle(it_user, ask("Reset the badge for jane.doe@acme.com"))
    assert outcome.route == Route.LOCAL_LARGE and outcome.reason == Reason.LOCAL_UNVERIFIED
    assert Reason.SENSITIVE_LOCAL_ONLY in outcome.reasons
    assert outcome.receipt["egress"] == "none" and gateway.guard.log() == []
    assert outcome.receipt["entities"] == {"EMAIL": 1}
    assert "jane" not in str(outcome.receipt)
    assert gateway.cache.stats()["entries"] == 0


async def test_hr_abstains_instead_of_guessing(db, hr_user):
    backend = ScriptedBackend(small=("Maybe?", UNSURE), large=("Hmm.", UNSURE))
    outcome = await build_gateway(db, backend, remote_mode="simulated").handle(hr_user, ask("What is the PTO policy?"))
    assert outcome.route == Route.ABSTAIN and outcome.reason == Reason.ROUTER_ABSTAIN
    assert outcome.status == 200 and "Hmm" not in outcome.text


async def test_secret_is_blocked_before_any_model_or_cache(db, it_user):
    backend = ScriptedBackend()
    gateway = build_gateway(db, backend)
    outcome = await gateway.handle(it_user, ask("debug this: sk-abcdefghijklmnopqrstuvwx1234"))
    assert outcome.status == 403 and outcome.route == Route.BLOCKED
    assert outcome.reason == Reason.SECRET_BLOCKED and backend.calls == []
    assert outcome.receipt["stages"][-1]["name"] == "policy"
    assert gateway.budget.spent_today(it_user) == 0.0


async def test_redact_policy_strips_pii_before_the_model_sees_it(db):
    backend = ScriptedBackend()
    outcome = await build_gateway(db, backend).handle(
        Identity("acme", "sales"), ask("Draft a note to jane.doe@acme.com about renewal")
    )
    sent = backend.calls[0][1][0]["content"]
    assert "[EMAIL]" in sent and "jane" not in sent
    assert outcome.receipt["input_redactions"] == 1 and Reason.INPUT_REDACTED in outcome.reasons
    assert outcome.receipt["data_class"] == "public"


async def test_pii_block_policy_and_pinned_route_denial(db, hr_user):
    gateway = build_gateway(db, remote_mode="simulated")
    blocked = await gateway.handle(Identity("acme", "legal"), ask("ssn 123-45-6789"))
    assert blocked.reason == Reason.POLICY_BLOCK and blocked.status == 403
    pinned = await gateway.handle(hr_user, ask(pinned=Route.REMOTE, requested_model="nanogate-remote"))
    assert pinned.reason == Reason.ROUTE_NOT_PERMITTED and pinned.status == 403


async def test_output_with_a_credential_is_withheld(db, it_user):
    backend = ScriptedBackend(small=("Use AKIAIOSFODNN7EXAMPLE to log in.", SURE))
    gateway = build_gateway(db, backend)
    outcome = await gateway.handle(it_user, ask())
    assert outcome.route == Route.BLOCKED and outcome.reason == Reason.OUTPUT_BLOCKED
    assert "AKIA" not in outcome.text and gateway.cache.stats()["entries"] == 0


async def test_new_pii_in_the_output_is_redacted_but_echoed_pii_is_kept(db, it_user):
    backend = ScriptedBackend(small=("Ask bob@acme.com or jane@acme.com.", SURE))
    outcome = await build_gateway(db, backend).handle(it_user, ask("Who besides jane@acme.com can help?"))
    assert outcome.reason == Reason.LOCAL_CONFIDENT and Reason.OUTPUT_REDACTED in outcome.reasons
    assert outcome.text.count("[EMAIL]") >= 1 and "bob@" not in outcome.text


async def test_budget_exhaustion_denies(db, it_user):
    gateway = build_gateway(db)
    gateway.budget.settle(gateway.budget.reserve(it_user, 2.5, 2.5), 2.5)
    outcome = await gateway.handle(it_user, ask())
    assert outcome.status == 429 and outcome.reason == Reason.BUDGET_DENY


async def test_rate_limit_and_oversized_input(db):
    gateway = build_gateway(db, max_prompt_chars=50)
    legal = Identity("acme", "legal", key_id="k1")
    big = await gateway.handle(legal, ask("x" * 51))
    assert big.status == 413 and big.reason == Reason.INPUT_TOO_LARGE
    results = [await gateway.handle(legal, ask("hi")) for _ in range(30)]
    assert results[-1].status == 429 and results[-1].reason == Reason.RATE_LIMITED


async def test_model_outage_degrades_then_reports_unavailable(db, it_user):
    large_only = await build_gateway(db, ScriptedBackend(small=None)).handle(it_user, ask())
    assert large_only.route == Route.LOCAL_LARGE and "DEGRADED_inference" in large_only.receipt["degraded"]
    down = await build_gateway(db, ScriptedBackend(small=None, large=None)).handle(it_user, ask("another question"))
    assert down.status == 503 and down.reason == Reason.MODEL_UNAVAILABLE and down.route == Route.ABSTAIN


async def test_untrained_router_never_claims_confidence(db, it_user, hr_user):
    gateway = build_gateway(db, trained=False)
    outcome = await gateway.handle(it_user, ask())
    assert outcome.reason == Reason.LOCAL_UNVERIFIED and outcome.route == Route.LOCAL_LARGE
    assert "DEGRADED_router" in outcome.receipt["degraded"]
    assert gateway.cache.stats()["entries"] == 0
    assert (await gateway.handle(hr_user, ask())).reason == Reason.LOCAL_UNVERIFIED


async def test_broken_cache_fails_open_and_broken_dlp_fails_closed(db, it_user):
    def boom(*args):
        raise RuntimeError("boom")

    gateway = build_gateway(db)
    gateway.cache.lookup = boom
    outcome = await gateway.handle(it_user, ask())
    assert outcome.status == 200 and "DEGRADED_cache" in outcome.receipt["degraded"]

    closed = build_gateway(db)
    closed.scanner.scan = boom
    outcome = await closed.handle(it_user, ask())
    assert outcome.status == 503 and outcome.reason == Reason.DLP_FAIL_CLOSED


async def test_multi_turn_conversations_skip_the_cache(db, it_user):
    backend = ScriptedBackend()
    gateway = build_gateway(db, backend)
    messages = [
        {"role": "user", "content": QUESTION},
        {"role": "assistant", "content": "Which OS?"},
        {"role": "user", "content": "the usual one"},
    ]
    for _ in range(2):
        outcome = await gateway.handle(it_user, ChatRequest(messages=messages))
        assert outcome.route == Route.LOCAL_SMALL
    assert gateway.cache.stats()["entries"] == 0


async def test_rejections_are_receipted_and_every_request_is_on_the_chain(db, it_user):
    gateway = build_gateway(db)
    await gateway.handle(it_user, ask())
    rejected = gateway.reject(None, Reason.AUTH_INVALID, 401)
    assert rejected.status == 401 and rejected.receipt["tenant"] == "unknown"
    assert gateway.receipts.verify() == {"status": SEALED, "checked": 2, "broken_at": None, "broken_seq": None, "detail": None}
    assert gateway.counters[("local_small", Reason.LOCAL_CONFIDENT)] == 1
