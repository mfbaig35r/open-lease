"""Operating schedules (capacity plan, Tier A1): resolve a deployment's desired running state from
the wall-clock, so capacity follows a declared plan (business hours, overnight shutdown) instead of
a variable meter.

Pure and clock-free by contract: ``now`` is always an argument, never read here, so this composes
with the reconciler's decision discipline (``next_step`` never reads the clock; the daemon resolves
the posture and passes the resulting ``desired_state`` in). The daemon calls this once per
reconcile tick.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from ..models import Deployment, DeploymentState, Posture, Schedule

# Posture -> the desired_state the reconciler should drive toward. ON is full capacity (READY); OFF
# is torn down with config retained (STOPPED). WARM_STANDBY is deferred to replicas (Tier B1).
_POSTURE_STATE: dict[Posture, DeploymentState] = {
    Posture.ON: DeploymentState.READY,
    Posture.OFF: DeploymentState.STOPPED,
}


def resolve_desired_state(deployment: Deployment, now: datetime) -> DeploymentState:
    """The desired_state a deployment should hold at ``now``. With no schedule the desired state is
    whatever it already is (manual control); a schedule makes the plan authoritative."""
    if deployment.schedule is None:
        return deployment.desired_state
    return _POSTURE_STATE[resolve_posture(deployment.schedule, now)]


def resolve_posture(schedule: Schedule, now: datetime) -> Posture:
    """The posture in force at ``now``: the first matching rule wins, else ``default_posture``.
    ``now`` is a timezone-aware instant (UTC from the daemon); it is read in the schedule's own
    timezone, so windows are wall-clock and DST-correct."""
    local = now.astimezone(ZoneInfo(schedule.timezone))
    weekday = local.weekday()  # 0=Mon .. 6=Sun
    clock = local.time()
    for rule in schedule.rules:
        if weekday in rule.days and _in_window(clock, rule.start, rule.end):
            return rule.posture
    return schedule.default_posture


def _in_window(clock: time, start: str, end: str) -> bool:
    """Is ``clock`` inside the half-open window [start, end)? A ``start`` later than ``end`` is an
    overnight window that wraps past midnight (22:00-06:00 covers 23:00 and 05:00, not noon).
    ``days`` match the weekday of the evaluated instant, not the window's start day."""
    lo, hi = time.fromisoformat(start), time.fromisoformat(end)
    if lo <= hi:
        return lo <= clock < hi
    return clock >= lo or clock < hi
