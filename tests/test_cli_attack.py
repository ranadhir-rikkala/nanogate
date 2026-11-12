import json
from pathlib import Path

import pytest

from nanogate import cli
from nanogate.attack import run_suite
from nanogate.config import Settings
from nanogate.router import FEATURES

ROOT = Path(__file__).parent.parent


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    for name in ("NANOGATE_DB", "NANOGATE_ROUTER_ARTIFACT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NANOGATE_DB", str(tmp_path / "var" / "gate.db"))
    monkeypatch.setenv("NANOGATE_ROUTER_ARTIFACT", str(tmp_path / "var" / "router.json"))
    return tmp_path


def test_attack_suite_holds():
    report = run_suite(ROOT / "bench" / "attacks.json", Settings(policy_path=str(ROOT / "config/policy.yaml"), rates_path=str(ROOT / "config/rates.yaml")))
    assert report["failures"] == []
    assert report["total"] >= 30


def test_init_issues_keys_once_with_private_permissions(workdir, capsys):
    assert cli.main(["init"]) == 0
    issued = json.loads((workdir / "var" / "bootstrap_keys.json").read_text())
    assert "app:acme/hr" in issued and "admin:acme" in issued and "auditor:globex" in issued
    assert oct((workdir / "var" / "bootstrap_keys.json").stat().st_mode & 0o777) == "0o600"
    assert issued["admin:acme"]["key"] not in capsys.readouterr().out
    assert cli.main(["init"]) == 1


def test_key_lifecycle_and_chain_verification(workdir, capsys):
    assert cli.main(["keys", "create", "acme", "it", "--label", "ci"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert cli.main(["keys", "list"]) == 0
    assert created["key_id"] in capsys.readouterr().out
    assert cli.main(["keys", "revoke", created["key_id"]]) == 0
    assert cli.main(["keys", "revoke", created["key_id"]]) == 1
    capsys.readouterr()
    assert cli.main(["verify"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "SEALED"


def test_train_router_writes_an_artifact(workdir, capsys):
    data = workdir / "graded.jsonl"
    rows = []
    for index in range(200):
        wrong = index % 2
        features = dict.fromkeys(FEATURES, 0.0) | {"logprob_mean": -2.0 if wrong else -0.2}
        rows.append(json.dumps({"features": features, "error": wrong, "group": f"g{index}"}))
    data.write_text("\n".join(rows))
    assert cli.main(["train-router", str(data)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert Path(summary["saved"]).is_file() and summary["held_out"]["auc"] == 1.0
