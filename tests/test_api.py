import json

import pytest
from fastapi.testclient import TestClient

from conftest import SURE, ScriptedBackend, build_gateway
from nanogate.api import Services, build_services, create_app
from nanogate.auth import KeyStore
from nanogate.config import Settings
from nanogate.receipts import SEALED

QUESTION = {"model": "nanogate-auto", "messages": [{"role": "user", "content": "How do I reset the VPN client?"}]}


@pytest.fixture
def env(db):
    keys = KeyStore(db, pepper="p")
    gateway = build_gateway(db, ScriptedBackend(), remote_mode="simulated")
    client = TestClient(create_app(services=Services(Settings(), db, keys, gateway)))
    tokens = {
        "it": keys.create("acme", "it")[1],
        "hr": keys.create("acme", "hr")[1],
        "admin": keys.create("acme", "it", role="admin")[1],
        "auditor": keys.create("acme", "it", role="auditor")[1],
        "globex_admin": keys.create("globex", "it", role="admin")[1],
    }
    return client, tokens, gateway, keys


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_chat_completion_is_openai_shaped_with_decision_headers(env):
    client, tokens, _, _ = env
    response = client.post("/v1/chat/completions", json=QUESTION, headers=auth(tokens["it"]))
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion" and body["model"] == "nanogate-auto"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Open Settings and choose Reset."}
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    assert response.headers["x-nanogate-route"] == "local_small"
    assert response.headers["x-nanogate-reason"] == "LOCAL_CONFIDENT"
    assert response.headers["x-nanogate-receipt-id"] == body["nanogate"]["receipt_id"]


def test_streaming_emits_sse_chunks_that_rebuild_the_answer(env):
    client, tokens, _, _ = env
    response = client.post("/v1/chat/completions", json={**QUESTION, "stream": True}, headers=auth(tokens["it"]))
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(event) for event in events[:-1]]
    text = "".join(chunk["choices"][0]["delta"].get("content", "") for chunk in chunks)
    assert text == "Open Settings and choose Reset."
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_missing_bad_and_revoked_keys(env):
    client, tokens, gateway, keys = env
    assert client.post("/v1/chat/completions", json=QUESTION).status_code == 401
    bad = client.post("/v1/chat/completions", json=QUESTION, headers=auth("ng_000000000000_" + "x" * 30))
    assert bad.status_code == 401 and bad.json()["error"]["code"] == "AUTH_INVALID"
    assert bad.headers["x-nanogate-route"] == "blocked"
    key_id = [row for row in keys.list() if row["department"] == "hr"][0]["key_id"]
    keys.revoke(key_id)
    revoked = client.post("/v1/chat/completions", json=QUESTION, headers=auth(tokens["hr"]))
    assert revoked.json()["error"]["code"] == "AUTH_REVOKED"
    assert gateway.receipts.verify()["checked"] == 3


def test_spoofed_department_header_is_refused(env):
    client, tokens, _, _ = env
    headers = {**auth(tokens["it"]), "X-NanoGate-Department": "hr"}
    response = client.post("/v1/chat/completions", json=QUESTION, headers=headers)
    assert response.status_code == 403 and response.json()["error"]["code"] == "IDENTITY_SPOOF_REJECTED"


def test_secret_prompt_is_rejected_with_reason(env):
    client, tokens, _, _ = env
    body = {"messages": [{"role": "user", "content": "use sk-abcdefghijklmnopqrstuvwx1234"}]}
    response = client.post("/v1/chat/completions", json=body, headers=auth(tokens["it"]))
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "SECRET_BLOCKED"
    assert response.headers["x-nanogate-data-class"] == "secret"


@pytest.mark.parametrize(
    "body, code",
    [
        ({"messages": []}, "invalid_messages"),
        ({"messages": "hi"}, "invalid_messages"),
        ({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}, "unknown_model"),
        ({"messages": [{"role": "user", "content": "hi"}], "max_tokens": "lots"}, "invalid_parameter"),
    ],
)
def test_request_validation(env, body, code):
    client, tokens, _, _ = env
    response = client.post("/v1/chat/completions", json=body, headers=auth(tokens["it"]))
    assert response.status_code == 400 and response.json()["error"]["code"] == code


def test_pinned_remote_model_is_denied_for_hr(env):
    client, tokens, _, _ = env
    response = client.post("/v1/chat/completions", json={**QUESTION, "model": "nanogate-remote"}, headers=auth(tokens["hr"]))
    assert response.status_code == 403 and response.json()["error"]["code"] == "ROUTE_NOT_PERMITTED"


def test_roles_gate_the_admin_surface(env):
    client, tokens, _, _ = env
    assert client.get("/api/overview", headers=auth(tokens["it"])).status_code == 403
    assert client.get("/api/overview", headers=auth(tokens["auditor"])).status_code == 200
    assert client.get("/api/keys", headers=auth(tokens["auditor"])).status_code == 403
    assert client.post("/v1/chat/completions", json=QUESTION, headers=auth(tokens["auditor"])).status_code == 403
    assert client.get("/v1/models", headers=auth(tokens["it"])).json()["data"][0]["id"] == "nanogate-auto"


def test_overview_receipts_and_chain_verification(env):
    client, tokens, _, _ = env
    for _ in range(2):
        client.post("/v1/chat/completions", json=QUESTION, headers=auth(tokens["it"]))
    overview = client.get("/api/overview", headers=auth(tokens["admin"])).json()
    assert overview["requests"] == 2 and overview["by_route"] == {"local_small": 1, "cache": 1}
    assert overview["cache"]["hits"] == 1 and overview["remote_mode"] == "simulated"
    listed = client.get("/api/receipts?limit=1", headers=auth(tokens["auditor"])).json()["receipts"]
    assert len(listed) == 1 and listed[0]["route"] == "cache"
    detail = client.get(f"/api/receipts/{listed[0]['receipt_id']}", headers=auth(tokens["auditor"])).json()
    assert detail["digest"] == listed[0]["digest"] and len(detail["seal"]) == 64
    verified = client.get("/api/chain/verify", headers=auth(tokens["auditor"])).json()
    assert verified["status"] == SEALED and verified["head"]["seq"] == verified["checked"]


def test_tenants_cannot_read_each_others_receipts_or_keys(env):
    client, tokens, _, _ = env
    client.post("/v1/chat/completions", json=QUESTION, headers=auth(tokens["it"]))
    receipt_id = client.get("/api/receipts", headers=auth(tokens["admin"])).json()["receipts"][0]["receipt_id"]
    other = auth(tokens["globex_admin"])
    assert client.get("/api/receipts", headers=other).json()["receipts"] == []
    assert client.get(f"/api/receipts/{receipt_id}", headers=other).status_code == 404
    assert client.get("/api/overview", headers=other).json()["requests"] == 0
    assert {row["tenant"] for row in client.get("/api/keys", headers=other).json()["keys"]} == {"globex"}
    acme_key = client.get("/api/keys", headers=auth(tokens["admin"])).json()["keys"][0]["key_id"]
    assert client.delete(f"/api/keys/{acme_key}", headers=other).status_code == 404


def test_admin_creates_and_revokes_keys(env):
    client, tokens, _, _ = env
    created = client.post("/api/keys", json={"department": "sales", "label": "crm"}, headers=auth(tokens["admin"]))
    assert created.status_code == 201
    new_key = created.json()["key"]
    assert client.post("/v1/chat/completions", json=QUESTION, headers=auth(new_key)).status_code == 200
    assert client.delete(f"/api/keys/{created.json()['key_id']}", headers=auth(tokens["admin"])).status_code == 200
    assert client.post("/v1/chat/completions", json=QUESTION, headers=auth(new_key)).status_code == 401
    assert client.post("/api/keys", json={"department": "it", "role": "root"}, headers=auth(tokens["admin"])).status_code == 400
    assert "key" not in json.dumps(client.get("/api/keys", headers=auth(tokens["admin"])).json()["keys"][0]).replace("key_id", "")


def test_policy_view_and_dry_run(env):
    client, tokens, _, _ = env
    view = client.get("/api/policy", headers=auth(tokens["auditor"])).json()
    assert "hr" in view["departments"] and "globex" not in json.dumps(view)
    dry = client.post(
        "/api/policy/test", json={"department": "hr", "text": "ssn 123-45-6789"}, headers=auth(tokens["auditor"])
    ).json()
    assert dry["scan"]["data_class"] == "restricted" and "remote" not in dry["allowed_routes"]
    assert dry["cacheable"] is False


def test_health_ready_metrics_and_cache_purge(env):
    client, tokens, _, _ = env
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["ready"] is True and ready["router_trained"] is True
    client.post("/v1/chat/completions", json=QUESTION, headers=auth(tokens["it"]))
    assert 'nanogate_requests_total{route="local_small",reason="LOCAL_CONFIDENT"} 1' in client.get("/metrics").text
    assert client.delete("/api/cache", headers=auth(tokens["admin"])).json() == {"removed": 1}
    assert client.get("/api/egress", headers=auth(tokens["admin"])).json() == {"mode": "simulated", "log": []}


def test_services_build_from_settings_and_persist_generated_secrets(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "var" / "gate.db"), policy_path="config/policy.yaml", rates_path="config/rates.yaml",
        router_artifact=str(tmp_path / "none.json"),
    )
    first = build_services(settings)
    _, key = first.keys.create("acme", "it")
    client = TestClient(create_app(services=first))
    response = client.post("/v1/chat/completions", json=QUESTION, headers=auth(key))
    assert response.status_code == 200 and response.headers["x-nanogate-reason"] == "LOCAL_UNVERIFIED"
    assert response.json()["choices"][0]["message"]["content"].startswith("[offline backend")
    first.db.close()
    second = build_services(settings)
    assert second.keys.resolve(key).department == "it"
    assert second.gateway.receipts.verify()["status"] == SEALED
    assert oct((tmp_path / "var" / "receipt.key").stat().st_mode & 0o777) == "0o600"
