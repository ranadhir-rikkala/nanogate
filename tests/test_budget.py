import threading
from pathlib import Path

from nanogate.budget import BudgetLedger, Pricing
from nanogate.db import Database
from nanogate.models import Route

pricing = Pricing.from_file(Path(__file__).parent.parent / "config" / "rates.yaml")


def test_local_routes_are_free_and_remote_is_priced():
    assert pricing.cost(Route.LOCAL_SMALL, 1000, 1000) == 0.0
    assert pricing.cost(Route.CACHE, 500, 500) == 0.0
    assert pricing.cost(Route.REMOTE, 1000, 1000) == 0.018
    assert pricing.cost(Route.BLOCKED, 1000, 1000) == 0.0
    assert pricing.rates[Route.REMOTE].basis == "assumed"


def test_ceiling_is_the_most_expensive_permitted_route():
    assert pricing.ceiling([Route.CACHE, Route.LOCAL_SMALL], 200, 1000) == 0.0
    assert pricing.ceiling([Route.LOCAL_SMALL, Route.REMOTE], 1000, 1000) == 0.018
    assert pricing.ceiling([], 1000, 1000) == 0.0


def test_reserve_settle_and_release(db, it_user):
    ledger = BudgetLedger(db)
    first = ledger.reserve(it_user, 0.4, daily_limit_usd=1.0)
    second = ledger.reserve(it_user, 0.4, daily_limit_usd=1.0)
    assert first.granted and second.granted
    assert ledger.reserve(it_user, 0.4, daily_limit_usd=1.0).granted is False

    assert ledger.settle(first, 0.1) is True
    assert ledger.settle(first, 0.1) is False
    assert ledger.release(second) is True
    assert round(ledger.spent_today(it_user), 6) == 0.1
    assert ledger.reserve(it_user, 0.9, daily_limit_usd=1.0).granted is True


def test_exhausted_budget_denies_even_free_requests(db, it_user):
    ledger = BudgetLedger(db)
    ledger.settle(ledger.reserve(it_user, 1.0, daily_limit_usd=1.0), 1.0)
    denied = ledger.reserve(it_user, 0.0, daily_limit_usd=1.0)
    assert denied.granted is False and denied.remaining_usd == 0.0


def test_departments_do_not_share_a_budget(db, it_user, hr_user):
    ledger = BudgetLedger(db)
    ledger.settle(ledger.reserve(it_user, 1.0, daily_limit_usd=1.0), 1.0)
    assert ledger.reserve(hr_user, 0.5, daily_limit_usd=1.0).granted is True
    rows = {(row["tenant"], row["department"]): row for row in ledger.by_department()}
    assert rows[("acme", "it")]["spent_usd"] == 1.0
    assert rows[("acme", "hr")]["held_usd"] == 0.5


def test_budget_resets_on_a_new_utc_day(db, it_user):
    now = [1_800_000_000.0]
    ledger = BudgetLedger(db, clock=lambda: now[0])
    ledger.settle(ledger.reserve(it_user, 1.0, daily_limit_usd=1.0), 1.0)
    assert ledger.reserve(it_user, 0.1, daily_limit_usd=1.0).granted is False
    now[0] += 86400
    assert ledger.reserve(it_user, 0.1, daily_limit_usd=1.0).granted is True


def test_concurrent_reservations_cannot_overspend(tmp_path, it_user):
    path = tmp_path / "race.db"
    Database(path).close()
    granted = []

    def worker():
        ledger = BudgetLedger(Database(path))
        granted.append(ledger.reserve(it_user, 0.1, daily_limit_usd=0.5).granted)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(granted) == 5
