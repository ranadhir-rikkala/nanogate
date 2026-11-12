import pytest

from nanogate.cache import HashingEmbedder, SemanticCache, SlotVerifier
from nanogate.cache.embedding import cosine, pack, unpack
from nanogate.models import Identity, Reason

verifier = SlotVerifier()
embedder = HashingEmbedder()


@pytest.fixture
def cache(db):
    return SemanticCache(db)


@pytest.fixture
def ns(cache, it_user):
    return cache.namespace(it_user, "v1", "small")


def test_embedding_is_deterministic_and_normalised():
    first, second = embedder.embed("Reset the VPN client"), embedder.embed("Reset the VPN client")
    assert first == second
    assert abs(cosine(first, first) - 1.0) < 1e-6
    assert cosine(first, embedder.embed("lunch menu today")) < 0.3
    assert embedder.embed("") == [0.0] * embedder.dims


def test_vector_round_trips_through_storage():
    vector = embedder.embed("hello world")
    restored = unpack(pack(vector))
    assert len(restored) == len(vector)
    assert abs(cosine(vector, restored) - 1.0) < 1e-5


@pytest.mark.parametrize(
    "query, cached",
    [
        ("How do I reset the VPN client?", "how can I reset my VPN client"),
        ("Where do I submit an expense report?", "where should expense reports be submitted"),
        ("How do I reset the VPN client?", "What are the steps for resetting the VPN client?"),
    ],
)
def test_verifier_accepts_paraphrases(query, cached):
    assert verifier.verify(query, cached).ok


@pytest.mark.parametrize(
    "query, cached, reason",
    [
        ("What is the PTO policy for 2025?", "What is the PTO policy for 2026?", "NUMBER_MISMATCH"),
        ("Can contractors access the wiki?", "Can contractors not access the wiki?", "NEGATION_MISMATCH"),
        ("How do I enable MFA on my account?", "How do I disable MFA on my account?", "ANTONYM_MISMATCH"),
        ("How do I install the driver on macOS?", "How do I install the driver on Windows?", "NAME_MISMATCH"),
        ("How do I reset the VPN client?", "How do I reset the VPN client on Windows?", "NAME_MISMATCH"),
        ("How do I request a new laptop?", "How do I request a new monitor?", "TERM_MISMATCH"),
    ],
)
def test_verifier_rejects_hard_negatives(query, cached, reason):
    verdict = verifier.verify(query, cached)
    assert not verdict.ok and verdict.reason == reason


def test_hit_after_put_for_a_paraphrase(cache, ns):
    cache.put(ns, "How do I reset the VPN client?", "Open Settings > VPN > Reset.")
    result = cache.lookup(ns, "how can I reset my VPN client")
    assert result.hit and result.answer == "Open Settings > VPN > Reset."
    assert result.reason == Reason.CACHE_VERIFIED
    assert cache.stats()["hits"] == 1


def test_similar_but_different_question_is_a_hard_negative(cache, ns):
    cache.put(ns, "Can contractors access the wiki?", "Yes, with a sponsor.")
    result = cache.lookup(ns, "Can contractors not access the wiki?")
    assert result.hit is False and result.answer is None
    assert result.reason == Reason.CACHE_HARD_NEGATIVE
    assert result.similarity >= cache.similarity


def test_unrelated_question_is_a_plain_miss(cache, ns):
    cache.put(ns, "How do I reset the VPN client?", "Open Settings.")
    result = cache.lookup(ns, "What is on the lunch menu?")
    assert result.hit is False and result.reason is None
    assert cache.lookup(cache.namespace(Identity("acme", "it"), "v1", "other"), "anything").hit is False


def test_namespace_isolates_tenant_department_policy_model_and_system_prompt(cache, it_user):
    base = cache.namespace(it_user, "v1", "small", "You are helpful.")
    variants = {
        cache.namespace(Identity("globex", "it"), "v1", "small", "You are helpful."),
        cache.namespace(Identity("acme", "hr"), "v1", "small", "You are helpful."),
        cache.namespace(it_user, "v2", "small", "You are helpful."),
        cache.namespace(it_user, "v1", "large", "You are helpful."),
        cache.namespace(it_user, "v1", "small", "You are a pirate."),
    }
    assert base not in variants and len(variants) == 5
    assert base == cache.namespace(Identity("acme", "it", role="admin", key_id="x"), "v1", "small", "You are helpful.")


def test_another_tenant_never_sees_the_entry(cache, ns):
    prompt = "How do I reset the VPN client?"
    cache.put(ns, prompt, "acme answer")
    other = cache.namespace(Identity("globex", "it"), "v1", "small")
    assert cache.lookup(other, prompt).hit is False


def test_entries_expire(db, it_user):
    now = [1000.0]
    cache = SemanticCache(db, ttl_s=60, clock=lambda: now[0])
    ns = cache.namespace(it_user, "v1", "small")
    cache.put(ns, "How do I reset the VPN client?", "answer")
    now[0] += 61
    assert cache.lookup(ns, "How do I reset the VPN client?").hit is False
    assert cache.purge(expired_only=True) == 1


def test_put_replaces_the_same_prompt_and_purge_is_scoped(cache, ns, hr_user):
    cache.put(ns, "How do I reset the VPN client?", "old")
    cache.put(ns, "How do I reset the VPN client?", "new")
    assert cache.stats()["entries"] == 1
    assert cache.lookup(ns, "How do I reset the VPN client?").answer == "new"
    other = cache.namespace(hr_user, "v1", "small")
    cache.put(other, "What is the PTO policy?", "20 days")
    assert cache.purge(ns) == 1
    assert cache.stats()["entries"] == 1
