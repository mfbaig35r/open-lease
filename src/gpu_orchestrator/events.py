"""Append-only event log (spec §12).

``EventLog.emit(event)`` does exactly two things: persist the event to SQLite and write one
structured log line carrying the event's correlation id. No subscribers, no bus, no callbacks.
Swamp's timeline view reads this table via the API later; nothing pushes.
"""

from __future__ import annotations

from datetime import datetime

from .logging import get_logger
from .models import Event, EventKind
from .store import Store

_log = get_logger("events")


class EventLog:
    def __init__(self, store: Store) -> None:
        self._store = store

    def emit(self, event: Event) -> None:
        """Persist the event, then log it. The write is the source of truth; the log mirrors it."""
        self._store.append_event(event)
        _log.info(
            "event",
            extra={
                "event_id": event.id,
                "event_kind": event.kind.value,
                "deployment_id": event.deployment_id,
                "correlation_id": event.correlation_id,
                "payload": event.payload,
            },
        )

    def query(
        self,
        deployment_id: str | None = None,
        *,
        since: datetime | None = None,
        kind: EventKind | None = None,
    ) -> list[Event]:
        return self._store.query_events(
            deployment_id=deployment_id,
            since=since,
            kind=kind.value if kind is not None else None,
        )
