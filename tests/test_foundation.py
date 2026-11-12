from nanogate.config import Settings
from nanogate.db import Database
from nanogate.models import DataClass, Reason, canonical_json


def test_data_class_is_ordered():
    assert DataClass.PUBLIC < DataClass.INTERNAL < DataClass.SECRET
    assert DataClass.parse("Restricted") is DataClass.RESTRICTED
    assert DataClass.CONFIDENTIAL.label == "confidential"


def test_primary_reason_follows_precedence():
    reasons = [Reason.SENSITIVE_LOCAL_ONLY, Reason.LOCAL_CONFIDENT, Reason.SECRET_BLOCKED]
    assert Reason.primary(reasons) == Reason.SECRET_BLOCKED
    assert Reason.primary([Reason.REMOTE_DISABLED, Reason.LOCAL_CONFIDENT]) == Reason.LOCAL_CONFIDENT


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})
    assert " " not in canonical_json({"a": {"b": 1}})


def test_settings_read_environment_with_defaults():
    settings = Settings.from_env({"NANOGATE_PORT": "9090", "NANOGATE_REMOTE_MODE": "simulated", "NANOGATE_DB": ""})
    assert settings.port == 9090
    assert settings.remote_mode == "simulated"
    assert settings.db_path == "var/nanogate.db"


def test_migrations_apply_once(tmp_path):
    path = tmp_path / "a" / "gate.db"
    first = Database(path)
    first.close()
    second = Database(path)
    tables = {row["name"] for row in second.query("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"api_keys", "budget_ledger", "receipts", "cache_entries", "egress_log"} <= tables
    assert second.one("PRAGMA user_version")[0] == 1


def test_write_rolls_back_on_error(db):
    try:
        with db.write() as conn:
            conn.execute(
                "INSERT INTO egress_log VALUES ('t1', 0, 'acme', 'it', 'public', 'simulated', 3)"
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert db.one("SELECT COUNT(*) FROM egress_log")[0] == 0
