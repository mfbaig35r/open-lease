"""The daemon: the owner that outlives a single CLI command (loop ownership = daemon, CLAUDE.md).

It runs five periodic loops over the shared store, each built on a callable core so the loops are
thin and the cores stay unit-testable:

- reconcile (every ``reconcile_interval``): ``reconcile_once`` on each active deployment.
- health (every ``health_poll_interval``): ``HealthMonitor.check_once`` on serving deployments.
- orphan sweep (every ``orphan_sweep_interval``, and once at startup): destroy any pod in this
  install's namespace that no active deployment owns, upgrading the cost-safety invariant from
  per-deployment to global-within-namespace (spec §7.5). A grace period avoids racing an in-flight
  create.
- costs (hourly): ``cost_snapshot`` per active deployment (spec §11).
- retention (hourly): prune events older than ``event_retention_days`` so the log stays bounded.

``tick_*`` run one pass and are what the tests drive; ``run`` wires them into sleeping loops.
Phase 1 is single-process (spec §7.4): no distributed locking.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from uuid import uuid4

from ..config import Config
from ..errors import OrchestratorError
from ..events import EventLog
from ..logging import get_logger
from ..models import BudgetAction, Deployment, DeploymentState, Event, EventKind, _utcnow
from ..providers.base import Provider
from ..store import Store
from . import autoscale, budgets, costs, schedule
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
        # Per-budget (window_start, phase) so a warning/exceeded event fires once per escalation,
        # not every tick. Reset when the window rolls over (Tier A2).
        self._budget_phase: dict[str, tuple[str, str]] = {}
        # Per-model (target, streak) for autoscale hysteresis: a new replica target must hold for
        # ``autoscale_hysteresis_ticks`` before it is applied, so a noisy rate does not flap (B2).
        self._autoscale_target: dict[str, tuple[int, int]] = {}

    # --- one pass of each loop (the testable cores) ---------------------------------

    async def tick_reconcile(self, now: datetime | None = None) -> None:
        now = now or _utcnow()
        for deployment in self._reconcilable(now):
            self._apply_policy(deployment, now)
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

    def _reconcilable(self, now: datetime) -> list[Deployment]:
        """Deployments to reconcile this tick: every non-stopped one, plus any stopped deployment
        whose schedule says it should be running now (so a window boundary wakes it). A stopped
        deployment with no such schedule is at rest and is not ticked, which is what keeps a resting
        OFF deployment from churning the store every interval."""
        out: dict[str, Deployment] = {}
        for d in self._store.list_deployments(include_stopped=True):
            if d.observed_state != DeploymentState.STOPPED:
                out[d.id] = d
            elif (
                d.schedule is not None
                and not d.budget_hold  # a budget hold outranks the schedule: do not wake it
                and schedule.resolve_desired_state(d, now) != DeploymentState.STOPPED
            ):
                out[d.id] = d
        return list(out.values())

    def _apply_policy(self, deployment: Deployment, now: datetime) -> None:
        """Resolve the desired_state a deployment should hold now, by precedence: an exceeded
        stop-budget (STOPPED) outranks the schedule, which outranks manual control. A no-op when
        nothing changes, so a settled deployment is left untouched."""
        if deployment.budget_hold:
            desired = DeploymentState.STOPPED
        elif deployment.schedule is not None:
            desired = schedule.resolve_desired_state(deployment, now)
        else:
            return
        if desired == deployment.desired_state:
            return
        deployment.desired_state = desired
        self._store.save_deployment(deployment)
        self._events.emit(
            Event(
                id=f"evt-{uuid4().hex[:12]}",
                correlation_id=deployment.id,
                deployment_id=deployment.id,
                kind=EventKind.RECONCILE_ACTION,
                payload={"action": "policy_desired", "desired_state": desired.value},
            )
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

    async def tick_retention(self, now: datetime | None = None) -> int:
        """Prune events older than the retention window, so the append-only log stays bounded."""
        now = now or _utcnow()
        cutoff = now - timedelta(days=self._config.event_retention_days)
        removed = self._store.prune_events(cutoff) + self._store.prune_usage(cutoff)
        if removed:
            _log.info("pruned old records", extra={"removed": removed})
        return removed

    async def tick_budget(self, now: datetime | None = None) -> None:
        """Evaluate every spend ceiling against the cost accrued this window (Tier A2): emit a
        warning/exceeded event on escalation, then reconcile the stop-holds so an exceeded
        stop-budget forces its scope down (and a fresh window releases it)."""
        now = now or _utcnow()
        statuses = [
            budgets.evaluate(b, self._store.get_cost_records(b.deployment_id), now)
            for b in self._store.list_budgets()
        ]
        for status in statuses:
            self._emit_budget_transition(status, now)
        self._reconcile_budget_holds(statuses)

    def _emit_budget_transition(self, status: budgets.BudgetStatus, now: datetime) -> None:
        phase = "exceeded" if status.exceeded else "warn" if status.over_warn else "ok"
        window = budgets.window_start(status.budget.window, now).isoformat()
        prev_window, prev_phase = self._budget_phase.get(status.budget.id, ("", "ok"))
        if window != prev_window:
            prev_phase = "ok"  # a new window starts fresh
        rank = {"ok": 0, "warn": 1, "exceeded": 2}
        if rank[phase] > rank[prev_phase]:
            kind = EventKind.BUDGET_EXCEEDED if status.exceeded else EventKind.BUDGET_WARNING
            self._emit_budget(kind, status)
        self._budget_phase[status.budget.id] = (window, phase)

    def _reconcile_budget_holds(self, statuses: list[budgets.BudgetStatus]) -> None:
        """Set budget_hold on every deployment an exceeded stop-budget covers, clear it otherwise.
        Derived fresh each tick, so a window rollover (spend back under the ceiling) releases the
        hold. An account budget (deployment_id None) covers every deployment."""
        deployments = self._store.list_deployments(include_stopped=True)
        held: set[str] = set()
        for status in statuses:
            if not (status.exceeded and status.budget.on_exceed is BudgetAction.STOP):
                continue
            if status.budget.deployment_id is None:
                held |= {d.id for d in deployments}
            else:
                held.add(status.budget.deployment_id)
        for deployment in deployments:
            should_hold = deployment.id in held
            if should_hold and not deployment.budget_hold:
                if deployment.desired_state is DeploymentState.STOPPED:
                    continue  # already down for its own reasons: nothing to force, nothing to undo
                deployment.budget_hold = True
                deployment.desired_state = DeploymentState.STOPPED
                self._store.save_deployment(deployment)
            elif not should_hold and deployment.budget_hold:
                deployment.budget_hold = False
                # Give the capacity back. The hold is the only reason it is down (a deployment
                # already stopped when the ceiling hit is never marked held, just above), and a
                # daily ceiling that stopped a deployment forever is not a window, it is a delete.
                deployment.desired_state = DeploymentState.READY
                self._store.save_deployment(deployment)
                self._emit_dep_event(deployment, EventKind.BUDGET_RELEASED, {})

    def _emit_budget(self, kind: EventKind, status: budgets.BudgetStatus) -> None:
        self._events.emit(
            Event(
                id=f"evt-{uuid4().hex[:12]}",
                correlation_id=status.budget.id,
                deployment_id=status.budget.deployment_id,
                kind=kind,
                payload={
                    "budget": status.budget.id,
                    "spent_usd": status.spent_usd,
                    "limit_usd": status.budget.limit_usd,
                    "fraction": status.fraction,
                },
            )
        )

    def _emit_dep_event(self, deployment: Deployment, kind: EventKind, payload: dict) -> None:
        self._events.emit(
            Event(
                id=f"evt-{uuid4().hex[:12]}",
                correlation_id=deployment.id,
                deployment_id=deployment.id,
                kind=kind,
                payload=payload,
            )
        )

    async def tick_autoscale(self, now: datetime | None = None) -> None:
        """Adjust each model's replica count to its served request rate (Tier B2). Declarative: it
        creates replica records or marks surplus STOPPED, and the reconcile loop drives them. A
        model whose deployments carry a schedule is skipped, since a schedule and an autoscaler
        would fight over desired_state; use one or the other per model in this phase."""
        now = now or _utcnow()
        since = now - timedelta(seconds=self._config.autoscale_window_seconds)
        window_minutes = self._config.autoscale_window_seconds / 60.0
        # A budget stop holds a deployment by forcing desired_state STOPPED, which drops it out of
        # the member count below. Without this the autoscaler reads the pool as short of its floor
        # and clones a replacement carrying no hold, so a per-deployment ceiling would be spent
        # straight through. A budget stop outranks the autoscaler, as it outranks a schedule.
        held_models = {
            d.model_id for d in self._store.list_deployments(include_stopped=True) if d.budget_hold
        }
        for policy in self._store.list_autoscale_policies():
            if policy.model_id in held_models:
                continue
            members = sorted(
                (
                    d
                    for d in self._store.list_deployments(include_stopped=False)
                    if d.model_id == policy.model_id and d.desired_state != DeploymentState.STOPPED
                ),
                key=lambda d: d.created_at,
            )
            if not members or any(m.schedule is not None for m in members):
                continue  # nothing to clone from, or a schedule already owns this model's lifecycle
            rpm = self._store.count_recent_requests([m.id for m in members], since) / window_minutes
            target = autoscale.desired_replicas(rpm, policy)
            if not self._autoscale_hold_elapsed(policy.model_id, target):
                continue
            if target > len(members):
                self._autoscale_up(members, target - len(members))
            elif target < len(members):
                self._autoscale_down(members, len(members) - target)

    def _autoscale_hold_elapsed(self, model_id: str, target: int) -> bool:
        """Hysteresis: True once ``target`` has been the computed answer for enough consecutive
        ticks to act on it, so a jittery rate does not add and remove replicas every minute."""
        prev, streak = self._autoscale_target.get(model_id, (target, 0))
        streak = streak + 1 if target == prev else 1
        self._autoscale_target[model_id] = (target, streak)
        return streak >= self._config.autoscale_hysteresis_ticks

    def _autoscale_up(self, members: list[Deployment], count: int) -> None:
        template = members[-1]  # newest member is the clone template
        for _ in range(count):
            clone = autoscale.replica_from(template, f"dep-{uuid4().hex[:6]}")
            self._store.save_deployment(clone)
            self._emit_dep_event(
                clone, EventKind.AUTOSCALED, {"action": "scale_up", "model_id": clone.model_id}
            )

    def _autoscale_down(self, members: list[Deployment], count: int) -> None:
        for surplus in members[-count:]:  # stop the newest surplus, keep the established replicas
            surplus.desired_state = DeploymentState.STOPPED
            self._store.save_deployment(surplus)
            self._emit_dep_event(
                surplus,
                EventKind.AUTOSCALED,
                {"action": "scale_down", "model_id": surplus.model_id},
            )

    # --- the long-running loop ------------------------------------------------------

    async def run(self) -> None:
        """Run all loops until cancelled. The sweep runs once immediately (startup sweep, §7.5)."""
        await self.tick_sweep()
        await asyncio.gather(
            self._loop(self.tick_reconcile, self._config.reconcile_interval),
            self._loop(self.tick_health, self._config.health_poll_interval),
            self._loop(self.tick_sweep, self._config.orphan_sweep_interval),
            self._loop(self.tick_costs, 3600),  # cost_snapshot is hourly (§11)
            self._loop(self.tick_retention, 3600),  # prune old events hourly
            self._loop(self.tick_budget, self._config.budget_poll_interval),  # spend ceilings (A2)
            self._loop(self.tick_autoscale, self._config.autoscale_poll_interval),  # replicas (B2)
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
