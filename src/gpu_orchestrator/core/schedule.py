"""Operating schedules (capacity plan, Tier A1): resolve a deployment's desired running state from
the wall-clock, so capacity follows a declared plan (business hours, overnight shutdown) instead of
a variable meter.

Pure and clock-free by contract: ``now`` is always an argument, never read here, so this composes
with the reconciler's decision discipline (``next_step`` never reads the clock; the daemon resolves
the posture and passes the resulting ``desired_state`` in). The daemon calls this once per
reconcile tick.

``build_schedule`` also lives here: the human window spec (``"mon-fri 06:00-18:00"``) is the same
whether it arrives from the CLI or an MCP tool call, so one parser serves both rather than each
interface growing its own.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from ..models import Deployment, DeploymentState, Posture, Schedule, ScheduleRule

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


# --- the human window spec ------------------------------------------------------------

_DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def parse_days(token: str) -> list[int]:
    """Parse a day token: ``all``, a range (``mon-fri``), or a list (``mon,wed,fri``)."""
    token = token.strip().lower()
    if token == "all":
        return [0, 1, 2, 3, 4, 5, 6]
    if "-" in token:
        lo, hi = token.split("-", 1)
        if lo not in _DAY_INDEX or hi not in _DAY_INDEX:
            raise ValueError(f"unknown day range {token!r} (use mon..sun)")
        return list(range(_DAY_INDEX[lo], _DAY_INDEX[hi] + 1))
    days = []
    for name in token.split(","):
        if name not in _DAY_INDEX:
            raise ValueError(f"unknown day {name!r} (use mon..sun, comma-separated)")
        days.append(_DAY_INDEX[name])
    return days


def parse_window(spec: str, posture: Posture) -> ScheduleRule:
    """Parse one window, ``"<days> HH:MM-HH:MM"`` (e.g. ``"mon-fri 06:00-18:00"``)."""
    parts = spec.split()
    if len(parts) != 2 or "-" not in parts[1]:
        raise ValueError(f'window must be "<days> HH:MM-HH:MM", got {spec!r}')
    start, end = parts[1].split("-", 1)
    return ScheduleRule(days=parse_days(parts[0]), start=start, end=end, posture=posture)


def build_schedule(
    on: list[str], off: list[str], timezone: str = "UTC", default: str = "off"
) -> Schedule:
    """Assemble a Schedule from ON / OFF window specs. ON windows are matched first, then OFF;
    ``default`` is the posture when nothing matches. Raises ValueError with a clean message on any
    malformed input (bad day, time, posture, timezone), so every interface reports the same."""
    if default.lower() not in ("on", "off"):
        raise ValueError("default posture must be on or off")
    try:
        rules = [parse_window(w, Posture.ON) for w in on]
        rules += [parse_window(w, Posture.OFF) for w in off]
        return Schedule(timezone=timezone, default_posture=Posture(default.lower()), rules=rules)
    except ValidationError as exc:  # model validators reject bad HH:MM / timezone
        raise ValueError(str(exc.errors()[0]["msg"])) from exc
