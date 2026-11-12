import copy
from pathlib import Path

import pytest
import yaml

from nanogate.dlp import Scanner
from nanogate.models import DataClass, Identity, Reason, Route
from nanogate.policy import PolicyEngine, PolicyError, effective_threshold

DOCUMENT = yaml.safe_load((Path(__file__).parent.parent / "config" / "policy.yaml").read_text())
engine = PolicyEngine(DOCUMENT)
scanner = Scanner(engine.classification)

PUBLIC = scanner.scan("How do I reset the VPN client?")
PII = scanner.scan("Email jane@acme.com about her badge")
RESTRICTED = scanner.scan("ssn 123-45-6789")
SECRET = scanner.scan("my key is sk-abcdefghijklmnopqrstuvwx1234")


def who(department, tenant="acme"):
    return Identity(tenant, department)


def test_shipped_policy_loads():
    assert PolicyEngine.from_file(Path(__file__).parent.parent / "config" / "policy.yaml").version == engine.version
    assert ("acme", "hr") in engine.departments


def test_public_request_may_escalate_when_remote_is_on():
    decision = engine.evaluate(who("it"), PUBLIC, remote_enabled=True)
    assert decision.allowed_routes == (Route.CACHE, Route.LOCAL_SMALL, Route.LOCAL_LARGE, Route.REMOTE)
    assert decision.cacheable and decision.reasons == ()


def test_remote_kill_switch_removes_route_and_says_so():
    decision = engine.evaluate(who("it"), PUBLIC, remote_enabled=False)
    assert Route.REMOTE not in decision.allowed_routes
    assert decision.reasons == (Reason.REMOTE_DISABLED,)


def test_sensitive_data_stays_local_even_with_remote_on():
    decision = engine.evaluate(who("it"), PII, remote_enabled=True)
    assert decision.allowed_routes == (Route.LOCAL_SMALL, Route.LOCAL_LARGE)
    assert decision.reasons == (Reason.SENSITIVE_LOCAL_ONLY,)
    assert decision.cacheable is False


def test_sensitivity_outranks_remote_disabled():
    decision = engine.evaluate(who("it"), RESTRICTED, remote_enabled=False)
    assert decision.reasons == (Reason.SENSITIVE_LOCAL_ONLY,)


def test_secret_is_blocked_everywhere():
    for department in ("it", "hr", "sales", "legal"):
        decision = engine.evaluate(who(department), SECRET, remote_enabled=True)
        assert decision.blocked and decision.reasons == (Reason.SECRET_BLOCKED,)


def test_unknown_department_or_tenant_fails_closed():
    assert engine.evaluate(who("finance"), PUBLIC).reasons == (Reason.POLICY_BLOCK,)
    assert engine.evaluate(who("hr", tenant="globex"), PUBLIC).blocked


def test_hr_never_has_a_remote_route():
    decision = engine.evaluate(who("hr"), PUBLIC, remote_enabled=True)
    assert Route.REMOTE not in decision.allowed_routes
    assert decision.abstain_on_low_confidence is True
    assert decision.max_p_error == 0.15


def test_block_action_refuses_pii():
    assert engine.evaluate(who("legal"), PUBLIC).blocked is False
    assert engine.evaluate(who("legal"), PII).reasons == (Reason.POLICY_BLOCK,)


def test_redact_action_asks_for_redaction_then_allows_remote():
    first = engine.evaluate(who("sales"), PII, remote_enabled=True)
    assert first.redact_input is True
    redacted_text, _ = scanner.redact("Email jane@acme.com about her badge")
    second = engine.evaluate(who("sales"), scanner.scan(redacted_text), remote_enabled=True, redacted=True)
    assert Route.REMOTE in second.allowed_routes
    assert Reason.INPUT_REDACTED in second.reasons


def test_pinned_route_must_already_be_permitted():
    pinned = engine.evaluate(who("it"), PUBLIC, pinned=Route.LOCAL_LARGE)
    assert pinned.allowed_routes == (Route.LOCAL_LARGE,) and pinned.cacheable is False
    denied = engine.evaluate(who("hr"), PUBLIC, pinned=Route.REMOTE, remote_enabled=True)
    assert denied.blocked and Reason.ROUTE_NOT_PERMITTED in denied.reasons
    sensitive = engine.evaluate(who("it"), PII, pinned=Route.REMOTE, remote_enabled=True)
    assert sensitive.blocked
    assert set(sensitive.reasons) == {Reason.ROUTE_NOT_PERMITTED, Reason.SENSITIVE_LOCAL_ONLY}


def test_version_changes_when_any_rule_changes():
    changed = copy.deepcopy(DOCUMENT)
    changed["tenants"]["acme"]["departments"]["hr"]["max_p_error"] = 0.1
    assert PolicyEngine(changed).version != engine.version
    assert PolicyEngine(copy.deepcopy(DOCUMENT)).version == engine.version


@pytest.mark.parametrize(
    "path, value",
    [
        (("defaults", "pii_action"), "shrug"),
        (("defaults", "routes"), ["cache", "abstain"]),
        (("defaults", "remote_max_class"), "secret"),
        (("defaults", "max_p_error"), 1.5),
        (("classification", "EMAIL"), "spicy"),
    ],
)
def test_invalid_documents_are_rejected(path, value):
    broken = copy.deepcopy(DOCUMENT)
    broken[path[0]][path[1]] = value
    with pytest.raises(PolicyError):
        PolicyEngine(broken)


def test_classification_feeds_the_scanner():
    assert engine.classification["SSN"] == DataClass.RESTRICTED
    assert RESTRICTED.data_class == DataClass.RESTRICTED


def test_effective_threshold_takes_the_stricter_value():
    assert effective_threshold(0.3, 0.15) == 0.15
    assert effective_threshold(0.1, 0.15) == 0.1
    assert effective_threshold(0.3, None) == 0.3
