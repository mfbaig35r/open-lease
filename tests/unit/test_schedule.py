"""Operating schedules (Tier A1): the pure posture resolver. Business-hours windows, overnight
wrap, timezone/weekday correctness, and the no-schedule pass-through. All dates below were verified
against America/New_York (EDT, UTC-4 in July): 2026-07-06 is a Monday, 2026-07-04 a Saturday."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gpu_orchestrator.core import schedule
from gpu_orchestrator.models import DeploymentState, Posture, Schedule, ScheduleRule
from tests.fixtures.deployments import make_deployment

# Mon-Fri 06:00-18:00 ON, else OFF: the canonical business-hours plan.
_BUSINESS = Schedule(
    timezone="America/New_York",
    default_posture=Posture.OFF,
    rules=[ScheduleRule(days=[0, 1, 2, 3, 4], start="06:00", end="18:00", posture=Posture.ON)],
)

# Default ON, overnight 22:00-06:00 OFF (every day): exercises the midnight wrap.
_OVERNIGHT = Schedule(
    timezone="America/New_York",
    default_posture=Posture.ON,
    rules=[ScheduleRule(start="22:00", end="06:00", posture=Posture.OFF)],
)


def _utc(y: int, mo: int, d: int, h: int) -> datetime:
    return datetime(y, mo, d, h, 0, tzinfo=UTC)


def test_on_inside_business_window():
    # UTC 14:00 Mon = 10:00 EDT, inside 06:00-18:00.
    assert schedule.resolve_posture(_BUSINESS, _utc(2026, 7, 6, 14)) is Posture.ON


def test_off_after_business_window():
    # UTC 23:00 Mon = 19:00 EDT, past 18:00.
    assert schedule.resolve_posture(_BUSINESS, _utc(2026, 7, 6, 23)) is Posture.OFF


def test_weekend_falls_to_default_off():
    # UTC 16:00 Sat = 12:00 EDT Saturday: no Mon-Fri rule matches -> default OFF, even at midday.
    assert schedule.resolve_posture(_BUSINESS, _utc(2026, 7, 4, 16)) is Posture.OFF


def test_overnight_window_wraps_midnight():
    # UTC 03:00 Mon = 23:00 EDT Sunday, inside the wrapped 22:00-06:00 -> OFF.
    assert schedule.resolve_posture(_OVERNIGHT, _utc(2026, 7, 6, 3)) is Posture.OFF
    # UTC 14:00 Mon = 10:00 EDT, outside the overnight window -> default ON.
    assert schedule.resolve_posture(_OVERNIGHT, _utc(2026, 7, 6, 14)) is Posture.ON


def test_resolve_desired_state_maps_posture():
    dep = make_deployment(DeploymentState.STOPPED)
    dep.schedule = _BUSINESS
    assert schedule.resolve_desired_state(dep, _utc(2026, 7, 6, 14)) is DeploymentState.READY
    assert schedule.resolve_desired_state(dep, _utc(2026, 7, 6, 23)) is DeploymentState.STOPPED


def test_no_schedule_passes_through_current_desired():
    dep = make_deployment(DeploymentState.READY)  # desired_state READY, no schedule
    assert dep.schedule is None
    assert schedule.resolve_desired_state(dep, _utc(2026, 7, 4, 16)) is dep.desired_state


def test_malformed_schedule_is_rejected_at_construction():
    with pytest.raises(ValueError):
        ScheduleRule(start="6am", end="18:00", posture=Posture.ON)  # bad HH:MM
    with pytest.raises(ValueError):
        ScheduleRule(days=[7], start="06:00", end="18:00", posture=Posture.ON)  # day out of range
    with pytest.raises(ValueError):
        Schedule(timezone="Mars/Olympus")  # unknown zone
