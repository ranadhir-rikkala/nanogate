import pytest

from nanogate.auth import AuthError, KeyStore, RateLimiter, bearer_token, reject_spoofing, require_role
from nanogate.models import Reason


def test_key_round_trip_and_hash_only_storage(db):
    store = KeyStore(db, pepper="p")
    key_id, key = store.create("Acme", "IT", label="service desk")
    identity = store.resolve(key)
    assert (identity.tenant, identity.department, identity.role, identity.key_id) == ("acme", "it", "app", key_id)
    stored = db.one("SELECT * FROM api_keys WHERE key_id = ?", (key_id,))
    assert key not in " ".join(str(value) for value in tuple(stored))


def test_unknown_malformed_and_tampered_keys_are_rejected(db):
    store = KeyStore(db, pepper="p")
    _, key = store.create("acme", "it")
    for bad in (None, "", "sk-not-ours", key[:-1] + ("A" if key[-1] != "A" else "B")):
        with pytest.raises(AuthError) as error:
            store.resolve(bad)
        assert error.value.code == Reason.AUTH_INVALID


def test_pepper_change_invalidates_keys(db):
    _, key = KeyStore(db, pepper="one").create("acme", "it")
    with pytest.raises(AuthError):
        KeyStore(db, pepper="two").resolve(key)


def test_revoked_key_stops_working(db):
    store = KeyStore(db)
    key_id, key = store.create("acme", "hr")
    assert store.revoke(key_id) is True
    assert store.revoke(key_id) is False
    with pytest.raises(AuthError) as error:
        store.resolve(key)
    assert error.value.code == Reason.AUTH_REVOKED


def test_invalid_scope_values_are_refused(db):
    store = KeyStore(db)
    with pytest.raises(ValueError):
        store.create("acme; DROP", "it")
    with pytest.raises(ValueError):
        store.create("acme", "it", role="root")


def test_header_cannot_override_key_identity(it_user):
    reject_spoofing(it_user, {"x-nanogate-tenant": "ACME", "x-nanogate-department": "it"})
    with pytest.raises(AuthError) as error:
        reject_spoofing(it_user, {"x-nanogate-department": "hr"})
    assert error.value.code == Reason.IDENTITY_SPOOF_REJECTED
    assert error.value.status == 403


def test_role_check(it_user):
    require_role(it_user, "app", "admin")
    with pytest.raises(AuthError) as error:
        require_role(it_user, "admin")
    assert error.value.code == Reason.SCOPE_DENIED


def test_bearer_parsing():
    assert bearer_token({"authorization": "Bearer abc"}) == "abc"
    assert bearer_token({"authorization": "Basic abc"}) is None
    assert bearer_token({}) is None


def test_rate_limiter_refills_over_time():
    now = [0.0]
    limiter = RateLimiter(clock=lambda: now[0])
    assert [limiter.allow("k", 2) for _ in range(3)] == [True, True, False]
    now[0] = 30.0
    assert limiter.allow("k", 2) is True
    assert limiter.allow("k", 2) is False
    assert limiter.allow("other", 2) is True
