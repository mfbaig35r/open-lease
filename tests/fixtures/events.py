"""Canonical Event fixtures: one per EventKind, sharing a correlation id.

Used to test the event log round-trip and, later, the timeline/render layers (spec §12).
"""

from __future__ import annotations

from datetime import UTC, datetime

from gpu_orchestrator.models import Event, EventKind

_T0 = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)
CORRELATION_ID = "corr-0001"
DEPLOYMENT_ID = "dep-a1b2c3"


def make_event(kind: EventKind, *, seq: int = 0, payload: dict | None = None) -> Event:
    return Event(
        id=f"evt-{seq:04d}",
        at=_T0,
        correlation_id=CORRELATION_ID,
        deployment_id=DEPLOYMENT_ID,
        kind=kind,
        payload=payload or {},
    )


# One event of every kind, in a plausible lifecycle order.
ALL_EVENTS: list[Event] = [make_event(kind, seq=i) for i, kind in enumerate(EventKind)]

# A representative happy-path timeline.
HAPPY_PATH: list[Event] = [
    make_event(EventKind.DEPLOYMENT_REQUESTED, seq=0),
    make_event(EventKind.INSTANCE_CREATED, seq=1, payload={"provider_instance_id": "pod-xyz123"}),
    make_event(EventKind.IMAGE_PULLED, seq=2),
    make_event(EventKind.MODEL_DOWNLOAD_STARTED, seq=3),
    make_event(EventKind.MODEL_DOWNLOAD_COMPLETED, seq=4),
    make_event(EventKind.SERVER_STARTED, seq=5),
    make_event(EventKind.HEALTH_PASSED, seq=6),
    make_event(EventKind.DEPLOYMENT_READY, seq=7, payload={"endpoint_url": "https://example"}),
]
