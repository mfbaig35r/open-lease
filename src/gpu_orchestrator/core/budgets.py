"""Spend ceilings (capacity plan, Tier A2): evaluate a budget against the cost already accrued in
its current window, so infrastructure spend has a knowable ceiling instead of an open meter.

Pure and clock-free by contract, like ``core/schedule.py``: ``now`` is always an argument. A budget
is a policy layer over the existing ``CostRecord`` data (GPU time), so there is no new accounting
here. Window boundaries are UTC in Phase 1. The daemon calls ``evaluate`` each budget tick and the
orchestrator calls ``admission_blocked`` before a new deploy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..models import Budget, BudgetAction, BudgetWindow, CostRecord


def window_start(window: BudgetWindow, now: datetime) -> datetime:
    """The UTC start of the window ``now`` falls in: midnight today (daily) or the 1st (monthly)."""
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if window is BudgetWindow.MONTHLY:
        return midnight.replace(day=1)
    return midnight


def window_spend(records: list[CostRecord], start: datetime, now: datetime) -> float:
    """Dollars accrued between ``start`` and ``now``. Each record contributes rate times the hours
    of its lifetime that fall inside the window, so a pod that started before the window counts only
    from ``start`` and one still running counts up to ``now``."""
    total = 0.0
    for record in records:
        lo = max(record.started_at, start)
        hi = min(record.stopped_at or now, now)
        if hi > lo:
            total += record.gpu_hourly_usd * (hi - lo).total_seconds() / 3600.0
    return round(total, 4)


@dataclass
class BudgetStatus:
    """The state of one budget at a point in time (a pure snapshot for events and enforcement)."""

    budget: Budget
    spent_usd: float
    fraction: float
    over_warn: bool
    exceeded: bool


def evaluate(budget: Budget, records: list[CostRecord], now: datetime) -> BudgetStatus:
    """Where ``budget`` stands right now, given the cost records in its scope."""
    spent = window_spend(records, window_start(budget.window, now), now)
    fraction = spent / budget.limit_usd if budget.limit_usd > 0 else 0.0
    return BudgetStatus(
        budget=budget,
        spent_usd=spent,
        fraction=round(fraction, 4),
        over_warn=fraction >= budget.warn_fraction,
        exceeded=spent >= budget.limit_usd,
    )


def admission_blocked(
    budgets: list[Budget], records: list[CostRecord], now: datetime
) -> Budget | None:
    """The first account-wide ``block_new`` budget that is over its ceiling now, or None. Only
    account budgets gate a brand-new deploy (a per-deployment budget has no deployment yet to bind
    to). ``records`` is the full cost history; no scope filtering is needed for account budgets."""
    for budget in budgets:
        if (
            budget.deployment_id is None
            and budget.on_exceed is BudgetAction.BLOCK_NEW
            and evaluate(budget, records, now).exceeded
        ):
            return budget
    return None
