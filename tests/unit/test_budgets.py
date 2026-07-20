"""Spend ceilings (Tier A2): the pure budget math. Window boundaries (UTC), in-window proration of
cost records, evaluation against the ceiling, and account-scoped admission blocking."""

from __future__ import annotations

from datetime import UTC, datetime

from gpu_orchestrator.core import budgets
from gpu_orchestrator.models import Budget, BudgetAction, BudgetWindow, CostRecord


def _utc(y: int, mo: int, d: int, h: int = 0) -> datetime:
    return datetime(y, mo, d, h, 0, tzinfo=UTC)


def _record(rate: float, started: datetime, stopped: datetime | None = None) -> CostRecord:
    return CostRecord(
        deployment_id="dep-x", gpu_hourly_usd=rate, started_at=started, stopped_at=stopped
    )


def test_window_start_daily_and_monthly():
    now = _utc(2026, 7, 6, 14)
    assert budgets.window_start(BudgetWindow.DAILY, now) == _utc(2026, 7, 6, 0)
    assert budgets.window_start(BudgetWindow.MONTHLY, now) == _utc(2026, 7, 1, 0)


def test_window_spend_prorates_across_the_boundary():
    now = _utc(2026, 7, 6, 12)  # noon
    start = budgets.window_start(BudgetWindow.DAILY, now)  # midnight today
    # A $2/hr pod running since yesterday counts only the 12 hours inside today.
    records = [_record(2.0, _utc(2026, 7, 5, 8))]  # still running (stopped_at None)
    assert budgets.window_spend(records, start, now) == 24.0


def test_window_spend_sums_multiple_records():
    now = _utc(2026, 7, 6, 10)
    start = budgets.window_start(BudgetWindow.DAILY, now)
    records = [
        _record(3.0, _utc(2026, 7, 6, 6), _utc(2026, 7, 6, 8)),  # 2h * $3 = $6, fully in-window
        _record(1.0, _utc(2026, 7, 6, 9)),  # 1h so far * $1 = $1, still running
    ]
    assert budgets.window_spend(records, start, now) == 7.0


def test_evaluate_flags_warn_then_exceeded():
    now = _utc(2026, 7, 6, 12)
    budget = Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=30.0, warn_fraction=0.8)
    records = [_record(2.0, _utc(2026, 7, 6, 0))]  # $2/hr * 12h = $24 -> 0.8 of 30
    status = budgets.evaluate(budget, records, now)
    assert status.spent_usd == 24.0
    assert status.over_warn and not status.exceeded  # exactly at the warn threshold

    now_later = _utc(2026, 7, 6, 16)  # $32 > $30
    assert budgets.evaluate(budget, records, now_later).exceeded


def test_admission_blocked_only_for_account_block_new_over_ceiling():
    now = _utc(2026, 7, 6, 12)
    records = [_record(5.0, _utc(2026, 7, 6, 0))]  # $60 so far
    account = Budget(
        id="bud-acct", window=BudgetWindow.DAILY, limit_usd=50.0, on_exceed=BudgetAction.BLOCK_NEW
    )
    assert budgets.admission_blocked([account], records, now) is account

    # A per-deployment budget never blocks a brand-new deploy (nothing to bind to yet).
    scoped = account.model_copy(update={"id": "bud-dep", "deployment_id": "dep-x"})
    assert budgets.admission_blocked([scoped], records, now) is None

    # A warn/stop budget does not block admission even when over.
    warn = account.model_copy(update={"id": "bud-warn", "on_exceed": BudgetAction.WARN})
    assert budgets.admission_blocked([warn], records, now) is None
