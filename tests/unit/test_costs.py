"""Step-6 cost tracking (spec §11): a record opens at instance creation and closes at destruction;
accrued = rate x elapsed. Deliberately simple."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpu_orchestrator.core import costs
from gpu_orchestrator.events import EventLog
from gpu_orchestrator.models import DeploymentState, EventKind
from gpu_orchestrator.store import Store
from tests.fixtures.deployments import make_deployment

_T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


def test_open_record_starts_accrual(tmp_path):
    store = Store(tmp_path / "c.db")
    dep = make_deployment(DeploymentState.READY)
    costs.open_record(dep, 1.89, _T0, store)
    records = store.get_cost_records(dep.id)
    assert len(records) == 1
    assert records[0].stopped_at is None
    assert records[0].gpu_hourly_usd == 1.89


def test_close_sets_stopped_and_accrued(tmp_path):
    store = Store(tmp_path / "c.db")
    dep = make_deployment(DeploymentState.READY)
    costs.open_record(dep, 2.0, _T0, store)
    costs.close_open_records(dep.id, _T0 + timedelta(hours=3), store)
    (record,) = store.get_cost_records(dep.id)
    assert record.stopped_at is not None
    assert record.accrued_usd == 6.0  # 2.0/hr x 3h


def test_open_closes_dangling_record(tmp_path):
    # An instance dying out of band and being replaced must not leave two open records.
    store = Store(tmp_path / "c.db")
    dep = make_deployment(DeploymentState.READY)
    costs.open_record(dep, 1.0, _T0, store)
    costs.open_record(dep, 1.0, _T0 + timedelta(hours=1), store)
    records = store.get_cost_records(dep.id)
    assert len(records) == 2
    assert len([r for r in records if r.stopped_at is None]) == 1


def test_close_is_idempotent(tmp_path):
    store = Store(tmp_path / "c.db")
    dep = make_deployment(DeploymentState.READY)
    costs.open_record(dep, 1.0, _T0, store)
    costs.close_open_records(dep.id, _T0 + timedelta(hours=1), store)
    costs.close_open_records(dep.id, _T0 + timedelta(hours=2), store)  # no-op second time
    (record,) = store.get_cost_records(dep.id)
    assert record.accrued_usd == 1.0  # closed at +1h, not moved by the second call


def test_emit_snapshot(tmp_path):
    store = Store(tmp_path / "c.db")
    events = EventLog(store)
    dep = make_deployment(DeploymentState.READY)
    costs.open_record(dep, 1.0, _T0, store)
    costs.emit_snapshot(dep, store, events, now=_T0 + timedelta(hours=1))
    snapshots = events.query(dep.id, kind=EventKind.COST_SNAPSHOT)
    assert len(snapshots) == 1
    assert snapshots[0].payload["open"] is True
