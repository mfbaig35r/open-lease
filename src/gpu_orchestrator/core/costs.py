"""Cost tracking (spec §11): deliberately simple. ``accrued = gpu_hourly_usd x elapsed_hours``.

A ``CostRecord`` opens when an instance is created and closes when it is destroyed; the accrual is a
computed field on the model, so there is nothing to tick here. The real cost feature is the
reconciler's no-orphans invariant (§7.3, §11); this module just records the money.

Lifecycle is driven from the reconciler's execute path: ``open_record`` on a successful create,
``close_open_records`` on a destroy. ``emit_snapshot`` writes a ``cost_snapshot`` event, called on a
slow cadence by the daemon (step 7). Per-token / bandwidth / storage costs are deferred (§11).
"""

from __future__ import annotations

from datetime import datetime

from ..events import EventLog
from ..models import CostRecord, Deployment, EventKind, _utcnow
from ..store import Store
from .outcomes import emit


def open_record(deployment: Deployment, gpu_hourly_usd: float, now: datetime, store: Store) -> None:
    """Start a cost accrual for a freshly-created instance. Any dangling open record (e.g. from an
    instance that died out of band and is being replaced) is closed first, so a deployment never has
    two open records at once."""
    close_open_records(deployment.id, now, store)
    store.save_cost_record(
        CostRecord(
            deployment_id=deployment.id,
            gpu_hourly_usd=gpu_hourly_usd,
            started_at=now,
        )
    )


def close_open_records(deployment_id: str, now: datetime, store: Store) -> None:
    """Stop accrual on every open record for a deployment (instance destroyed). Idempotent: closing
    an already-closed or absent record is a no-op."""
    for record in store.get_cost_records(deployment_id):
        if record.stopped_at is None:
            store.save_cost_record(record.model_copy(update={"stopped_at": now}))


def emit_snapshot(
    deployment: Deployment, store: Store, events: EventLog, now: datetime | None = None
) -> None:
    """Emit a ``cost_snapshot`` event carrying the deployment's current accrued total. The daemon
    calls this hourly (step 7); it is a pure read of the stored records plus one event."""
    now = now or _utcnow()
    records = store.get_cost_records(deployment.id)
    if not records:
        return
    accrued = round(sum(r.accrued_usd for r in records), 4)
    emit(
        events,
        deployment,
        EventKind.COST_SNAPSHOT,
        {"accrued_usd": accrued, "open": any(r.stopped_at is None for r in records)},
    )
