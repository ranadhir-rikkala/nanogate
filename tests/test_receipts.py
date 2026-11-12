import json

import pytest

from nanogate.receipts import GENESIS, INTEGRITY_FAILED, SEALED, ReceiptLedger, chain_digest


def fields(n, route="local_small"):
    return {
        "tenant": "acme",
        "department": "it",
        "route": route,
        "reason": "LOCAL_CONFIDENT",
        "data_class": "public",
        "cost_usd": 0.0,
        "latency_ms": 12.5,
        "request_id": f"req_{n}",
    }


@pytest.fixture
def ledger(db):
    ledger = ReceiptLedger(db, key="test-key")
    for n in range(5):
        ledger.append(fields(n))
    return ledger


def test_chain_links_from_genesis(ledger):
    receipts = list(reversed(ledger.recent(10)))
    assert [receipt["seq"] for receipt in receipts] == [1, 2, 3, 4, 5]
    first = ledger.get(receipts[0]["receipt_id"])
    second = ledger.get(receipts[1]["receipt_id"])
    assert first["prev_hash"] == GENESIS
    assert second["prev_hash"] == first["digest"]
    assert ledger.verify() == {"status": SEALED, "checked": 5, "broken_at": None, "broken_seq": None, "detail": None}


def test_empty_chain_verifies(db):
    assert ReceiptLedger(db, key="k").verify()["status"] == SEALED


def test_editing_a_body_is_detected_at_that_receipt(ledger, db):
    target = db.one("SELECT receipt_id, body FROM receipts WHERE seq = 3")
    body = json.loads(target["body"])
    body["route"] = "remote"
    db.conn().execute("UPDATE receipts SET body = ? WHERE seq = 3", (json.dumps(body),))
    result = ledger.verify()
    assert result["status"] == INTEGRITY_FAILED
    assert result["broken_at"] == target["receipt_id"] and result["broken_seq"] == 3


def test_recomputing_one_digest_breaks_the_next_link(ledger, db):
    row = db.one("SELECT * FROM receipts WHERE seq = 2")
    body = json.loads(row["body"])
    body["cost_usd"] = 99.0
    forged = chain_digest(row["prev_hash"], body)
    db.conn().execute(
        "UPDATE receipts SET body = ?, digest = ?, seal = ? WHERE seq = 2",
        (json.dumps(body, sort_keys=True, separators=(",", ":")), forged, ledger._seal(2, forged)),
    )
    result = ledger.verify()
    assert result["status"] == INTEGRITY_FAILED and result["broken_seq"] == 3


def test_full_rebuild_without_the_key_fails_on_the_seal(ledger, db):
    prev = GENESIS
    for row in db.query("SELECT * FROM receipts ORDER BY seq"):
        body = json.loads(row["body"])
        body["department"] = "hr"
        digest = chain_digest(prev, body)
        db.conn().execute(
            "UPDATE receipts SET body = ?, prev_hash = ?, digest = ?, department = 'hr' WHERE seq = ?",
            (json.dumps(body, sort_keys=True, separators=(",", ":")), prev, digest, row["seq"]),
        )
        prev = digest
    result = ledger.verify()
    assert result["status"] == INTEGRITY_FAILED and result["broken_seq"] == 1
    assert "seal" in result["detail"]


def test_deleting_a_middle_receipt_is_detected(ledger, db):
    db.conn().execute("DELETE FROM receipts WHERE seq = 2")
    result = ledger.verify()
    assert result["status"] == INTEGRITY_FAILED and "sequence" in result["detail"]


def test_truncating_the_tail_is_caught_by_an_anchored_head(ledger, db):
    anchored = ledger.head()
    assert ledger.verify(expected_head=anchored)["status"] == SEALED
    db.conn().execute("DELETE FROM receipts WHERE seq = 5")
    assert ledger.verify()["status"] == SEALED
    assert ledger.verify(expected_head=anchored)["status"] == INTEGRITY_FAILED


def test_indexed_columns_cannot_drift_from_the_sealed_body(ledger, db):
    db.conn().execute("UPDATE receipts SET route = 'cache' WHERE seq = 4")
    result = ledger.verify()
    assert result["status"] == INTEGRITY_FAILED and "route" in result["detail"]


def test_a_different_key_does_not_verify(ledger, db):
    assert ReceiptLedger(db, key="other").verify()["status"] == INTEGRITY_FAILED


def test_recent_filters_by_scope(ledger):
    ledger.append({**fields(9), "department": "hr"})
    assert len(ledger.recent(10, department="hr")) == 1
    assert len(ledger.recent(10, tenant="acme")) == 6
    assert ledger.recent(2)[0]["seq"] == 6


def test_key_is_required(db):
    with pytest.raises(ValueError):
        ReceiptLedger(db, key="")
