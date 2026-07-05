"""SQLite state store: deployments, events, cost_records (spec §12).

No ORM. ``sqlite3`` stdlib with thin typed helpers. Each row keeps the Pydantic model as a JSON
document in a ``doc`` column, plus a few extracted columns for indexing/filtering.

Two-layer migrations (spec §12):
  1. DDL migrations (this module's ``_MIGRATIONS``) own table shape, tracked by ``PRAGMA
     user_version``, applied on startup.
  2. Per-document ``schema_version`` upgraders own the JSON payload (``_UPGRADERS``); a document
     whose version has no path to the current version fails loudly with ``SchemaVersionError``.

Phase 1 is single-process (spec §7.4): WAL mode plus one process-wide lock. No distributed locking.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from .errors import DeploymentNotFoundError, SchemaVersionError
from .models import SCHEMA_VERSION, CostRecord, Deployment, Event

# --- DDL migrations (own table/column shape only) ------------------------------------

_MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE deployments (
        id             TEXT PRIMARY KEY,
        doc            TEXT NOT NULL,
        observed_state TEXT NOT NULL,
        updated_at     TEXT NOT NULL
    );
    CREATE TABLE events (
        id            TEXT PRIMARY KEY,
        deployment_id TEXT,
        at            TEXT NOT NULL,
        kind          TEXT NOT NULL,
        doc           TEXT NOT NULL
    );
    CREATE INDEX idx_events_deployment ON events(deployment_id);
    CREATE INDEX idx_events_at ON events(at);
    CREATE TABLE cost_records (
        deployment_id TEXT NOT NULL,
        started_at    TEXT NOT NULL,
        doc           TEXT NOT NULL,
        PRIMARY KEY (deployment_id, started_at)
    );
    """,
    # v2: per-model download lock so two concurrent cold deploys of the same model do not both
    # write weights to the shared cache volume and corrupt it (§14). A lease, not a hard lock.
    """
    CREATE TABLE download_locks (
        model_id    TEXT PRIMARY KEY,
        holder      TEXT NOT NULL,
        acquired_at TEXT NOT NULL
    );
    """,
]

# --- Payload upgraders (own JSON payload only) ---------------------------------------
# Keyed by (model_name, from_version) -> function returning the doc dict at from_version + 1.
# Empty at v1; add an entry here (never a DDL change) when a model's JSON shape changes.
_UPGRADERS: dict[tuple[str, int], Callable[[dict], dict]] = {}


def _migrate_document(model_name: str, data: dict) -> dict:
    """Upgrade a stored JSON payload to the current schema version, or fail loudly."""
    version = data.get("schema_version", 1)
    while version != SCHEMA_VERSION:
        upgrader = _UPGRADERS.get((model_name, version))
        if upgrader is None:
            raise SchemaVersionError(
                f"{model_name} document at schema_version {version} cannot be loaded "
                f"(current is {SCHEMA_VERSION}); no upgrader registered."
            )
        data = upgrader(data)
        version = data.get("schema_version", version)
    return data


class Store:
    """The single owner of the SQLite connection."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    # --- lifecycle ------------------------------------------------------------------

    def _migrate(self) -> None:
        with self._lock:
            current = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for version in range(current, len(_MIGRATIONS)):
                self._conn.executescript(_MIGRATIONS[version])
                self._conn.execute(f"PRAGMA user_version = {version + 1}")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- deployments ----------------------------------------------------------------

    def save_deployment(self, deployment: Deployment) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO deployments (id, doc, observed_state, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    doc = excluded.doc,
                    observed_state = excluded.observed_state,
                    updated_at = excluded.updated_at
                """,
                (
                    deployment.id,
                    deployment.model_dump_json(),
                    deployment.observed_state.value,
                    deployment.updated_at.isoformat(),
                ),
            )
            self._conn.commit()

    def get_deployment(self, deployment_id: str) -> Deployment:
        with self._lock:
            row = self._conn.execute(
                "SELECT doc FROM deployments WHERE id = ?", (deployment_id,)
            ).fetchone()
        if row is None:
            raise DeploymentNotFoundError(f"No deployment with id {deployment_id!r}")
        data = _migrate_document("Deployment", json.loads(row["doc"]))
        return Deployment.model_validate(data)

    def list_deployments(self, *, include_stopped: bool = False) -> list[Deployment]:
        query = "SELECT doc FROM deployments"
        params: tuple = ()
        if not include_stopped:
            query += " WHERE observed_state != ?"
            params = ("stopped",)
        query += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            Deployment.model_validate(_migrate_document("Deployment", json.loads(r["doc"])))
            for r in rows
        ]

    def delete_deployment(self, deployment_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM deployments WHERE id = ?", (deployment_id,))
            self._conn.commit()

    # --- events ---------------------------------------------------------------------

    def append_event(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO events (id, deployment_id, at, kind, doc) VALUES (?, ?, ?, ?, ?)",
                (
                    event.id,
                    event.deployment_id,
                    event.at.isoformat(),
                    event.kind.value,
                    event.model_dump_json(),
                ),
            )
            self._conn.commit()

    def query_events(
        self,
        *,
        deployment_id: str | None = None,
        since: datetime | None = None,
        kind: str | None = None,
    ) -> list[Event]:
        clauses: list[str] = []
        params: list[object] = []
        if deployment_id is not None:
            clauses.append("deployment_id = ?")
            params.append(deployment_id)
        if since is not None:
            clauses.append("at >= ?")
            params.append(since.isoformat())
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        query = "SELECT doc FROM events"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY at ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            Event.model_validate(_migrate_document("Event", json.loads(r["doc"]))) for r in rows
        ]

    # --- cost records ---------------------------------------------------------------

    def save_cost_record(self, record: CostRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO cost_records (deployment_id, started_at, doc)
                VALUES (?, ?, ?)
                ON CONFLICT(deployment_id, started_at) DO UPDATE SET doc = excluded.doc
                """,
                (record.deployment_id, record.started_at.isoformat(), record.model_dump_json()),
            )
            self._conn.commit()

    def get_cost_records(self, deployment_id: str | None = None) -> list[CostRecord]:
        query = "SELECT doc FROM cost_records"
        params: tuple = ()
        if deployment_id is not None:
            query += " WHERE deployment_id = ?"
            params = (deployment_id,)
        query += " ORDER BY started_at ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            CostRecord.model_validate(_migrate_document("CostRecord", json.loads(r["doc"])))
            for r in rows
        ]

    # --- download locks (cache-write coordination, §14) -----------------------------

    def acquire_download_lock(
        self, model_id: str, holder: str, now: datetime, ttl_seconds: int
    ) -> bool:
        """Claim the per-model download lease. Returns True if this ``holder`` now holds it (free,
        already ours, or the previous holder's lease is stale past ``ttl_seconds``); False if a
        different holder holds a fresh lease. The stale-steal keeps a crashed holder from blocking
        a model forever."""
        with self._lock:
            row = self._conn.execute(
                "SELECT holder, acquired_at FROM download_locks WHERE model_id = ?", (model_id,)
            ).fetchone()
            if row is not None and row["holder"] != holder:
                held_for = (now - datetime.fromisoformat(row["acquired_at"])).total_seconds()
                if held_for < ttl_seconds:
                    return False
            self._conn.execute(
                """
                INSERT INTO download_locks (model_id, holder, acquired_at) VALUES (?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET holder = excluded.holder,
                    acquired_at = excluded.acquired_at
                """,
                (model_id, holder, now.isoformat()),
            )
            self._conn.commit()
            return True

    def release_download_lock(self, model_id: str, holder: str) -> None:
        """Release the lease if this ``holder`` holds it (idempotent, no-op otherwise)."""
        with self._lock:
            self._conn.execute(
                "DELETE FROM download_locks WHERE model_id = ? AND holder = ?", (model_id, holder)
            )
            self._conn.commit()

    # --- retention ------------------------------------------------------------------

    def prune_events(self, before: datetime) -> int:
        """Delete events older than ``before``; returns how many were removed. The event log is
        append-only and otherwise grows unbounded under a long-running daemon."""
        with self._lock:
            cursor = self._conn.execute("DELETE FROM events WHERE at < ?", (before.isoformat(),))
            self._conn.commit()
            return cursor.rowcount
