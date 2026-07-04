"""The daemon: the owner that outlives a single CLI command (loop ownership = daemon, CLAUDE.md).

It runs three periodic loops over the shared store, each built on a callable core so the loops are
thin and the cores stay unit-testable:

- reconcile (every ``reconcile_interval``): ``reconcile_once`` on each active deployment.
- health (every ``health_poll_interval``): ``HealthMonitor.check_once`` on serving deployments.
- orphan sweep (every ``orphan_sweep_interval``, and once at startup): destroy any pod in this
  install's namespace that no active deployment owns, upgrading the cost-safety invariant from
  per-deployment to global-within-namespace (spec §7.5). A grace period avoids racing an in-flight
  create.

``tick_*`` run one pass and are what the tests drive; ``run`` wires them into sleeping loops.
Phase 1 is single-process (spec §7.4): no distributed locking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from uuid import uuid4

from ..config import Config
from ..errors import OrchestratorError
from ..events import EventLog
from ..logging import get_logger
from ..models import DeploymentState, Event, EventKind, _utcnow
from ..providers.base import Provider
from ..store import Store
from . import costs
from .catalog import Catalog, load_catalog
from .health import HealthMonitor
from .orchestrator import build_provider, build_runtime
from .reconciler import reconcile_once

_log = get_logger("daemon")
_SERVING = (DeploymentState.READY, DeploymentState.DEGRADED)


class Daemon:
    def __init__(
        self,
        config: Config | None = None,
        *,
        store: Store | None = None,
        events: EventLog | None = None,
        catalog: Catalog | None = None,
        provider: Provider | None = None,
        runtime: object | None = None,
    ) -> None:
        # provider/runtime injection is the test seam (mock provider + fake runtime), mirroring
        # Orchestrator; production leaves them None and builds by name from config.
        self._config = config or Config()
        self._store = store or Store(self._config.state_db)
        self._events = events or EventLog(self._store)
        self._catalog = catalog or load_catalog()
        self._injected_provider = provider
        self._injected_runtime = runtime
        self._monitor = HealthMonitor(self._config)
        self._orphan_seen: dict[str, datetime] = {}

    # --- one pass of each loop (the testable cores) ---------------------------------

    async def tick_reconcile(self, now: datetime | None = None) -> None:
        for deployment in self._store.list_deployments(include_stopped=False):
            await reconcile_once(
                deployment,
                provider=self._provider(deployment.provider),
                runtime=self._runtime(),
                catalog=self._catalog,
                config=self._config,
                store=self._store,
                events=self._events,
                now=now,
            )

    async def tick_health(self, now: datetime | None = None) -> None:
        for deployment in self._store.list_deployments(include_stopped=False):
            if deployment.observed_state in _SERVING:
                await self._monitor.check_once(
                    deployment,
                    provider=self._provider(deployment.provider),
                    runtime=self._runtime(),
                    store=self._store,
                    events=self._events,
                    now=now,
                )

    async def tick_sweep(self, now: datetime | None = None) -> list[str]:
        now = now or _utcnow()
        names = {d.provider for d in self._store.list_deployments(include_stopped=True)} or {
            "runpod"
        }
        destroyed: list[str] = []
        for name in names:
            try:
                destroyed += await sweep_orphans(
                    self._store,
                    self._provider(name),
                    self._config,
                    now,
                    self._orphan_seen,
                    self._events,
                )
            except OrchestratorError as exc:
                _log.warning("orphan sweep failed", extra={"provider": name, "error": str(exc)})
        return destroyed

    async def tick_costs(self) -> None:
        for deployment in self._store.list_deployments(include_stopped=False):
            costs.emit_snapshot(deployment, self._store, self._events)

    # --- the long-running loop ------------------------------------------------------

    async def run(self) -> None:
        """Run all loops until cancelled. The sweep runs once immediately (startup sweep, §7.5)."""
        await self.tick_sweep()
        await asyncio.gather(
            self._loop(self.tick_reconcile, self._config.reconcile_interval),
            self._loop(self.tick_health, self._config.health_poll_interval),
            self._loop(self.tick_sweep, self._config.orphan_sweep_interval),
            self._loop(self.tick_costs, 3600),  # cost_snapshot is hourly (§11)
        )

    async def _loop(self, tick, interval: int) -> None:
        while True:
            try:
                await tick()
            except OrchestratorError as exc:
                _log.warning("daemon tick failed", extra={"tick": tick.__name__, "error": str(exc)})
            await asyncio.sleep(interval)

    def _provider(self, name: str) -> Provider:
        return self._injected_provider or build_provider(self._config, name)

    def _runtime(self):
        return self._injected_runtime or build_runtime()


async def sweep_orphans(
    store: Store,
    provider: Provider,
    config: Config,
    now: datetime,
    seen: dict[str, datetime],
    events: EventLog,
) -> list[str]:
    """Destroy pods in this namespace that no active deployment owns, after a grace period (§7.5).

    ``seen`` tracks first-sighting time per orphan across calls so the grace period spans sweeps and
    a just-created pod (not yet saved to its deployment record) is not reaped mid-creation."""
    active = [d for d in store.list_deployments(include_stopped=False) if d.instance is not None]
    known = {d.instance.provider_instance_id for d in active}  # type: ignore[union-attr]

    destroyed: list[str] = []
    present: set[str] = set()
    for instance in await provider.list_instances():
        pid = instance.provider_instance_id
        present.add(pid)
        if pid in known:
            seen.pop(pid, None)
            continue
        first_seen = seen.setdefault(pid, now)
        if first_seen is now:
            _emit_orphan(events, EventKind.ORPHAN_DETECTED, pid)
        if (now - first_seen).total_seconds() >= config.orphan_grace_period:
            await provider.destroy_instance(pid)
            seen.pop(pid, None)
            destroyed.append(pid)
            _emit_orphan(events, EventKind.ORPHAN_DESTROYED, pid)

    for pid in [p for p in seen if p not in present]:  # forget vanished candidates
        seen.pop(pid, None)
    return destroyed


def _emit_orphan(events: EventLog, kind: EventKind, provider_instance_id: str) -> None:
    events.emit(
        Event(
            id=f"evt-{uuid4().hex[:12]}",
            correlation_id=provider_instance_id,
            deployment_id=None,
            kind=kind,
            payload={"instance": provider_instance_id},
        )
    )
