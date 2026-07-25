"""Step-7 daemon: the three loops driven one tick at a time (the testable cores), plus the orphan
sweep with its grace period (spec §7.5)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.catalog import Catalog
from gpu_orchestrator.core.daemon import Daemon
from gpu_orchestrator.events import EventLog
from gpu_orchestrator.models import (
    AutoscalePolicy,
    Budget,
    BudgetWindow,
    CostRecord,
    Deployment,
    DeploymentState,
    EventKind,
    InstanceRequest,
    Posture,
    Schedule,
    ScheduleRule,
)
from gpu_orchestrator.providers.mock import MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime
from gpu_orchestrator.store import Store
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC

S = DeploymentState
_PROFILE = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "MOCK-GPU"})
_CATALOG = Catalog({"qwen3-0.6b": QWEN3_06B_SPEC}, {"qwen3-0.6b": _PROFILE})
_T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


def _runtime(*, model_ok: bool = True) -> VLLMRuntime:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-0.6b"}] if model_ok else []})
        return httpx.Response(404)

    return VLLMRuntime(transport=httpx.MockTransport(handler))


def _daemon(tmp_path, provider, *, runtime=None, **cfg) -> tuple[Daemon, Store, EventLog]:
    store = Store(tmp_path / "d.db")
    events = EventLog(store)
    config = Config(namespace="test", state_db=tmp_path / "d.db", reconcile_interval=0, **cfg)
    daemon = Daemon(
        config,
        store=store,
        events=events,
        catalog=_CATALOG,
        provider=provider,
        runtime=runtime or _runtime(),
    )
    return daemon, store, events


def _seed(
    store: Store,
    *,
    state: S = S.REQUESTED,
    instance=None,
    endpoint=None,
    desired: S = S.READY,
    schedule=None,
) -> Deployment:
    dep = Deployment(
        id="dep-d1",
        model_id="qwen3-0.6b",
        provider="mock",
        desired_state=desired,
        observed_state=state,
        profile=_PROFILE,
        instance=instance,
        endpoint_url=endpoint,
        schedule=schedule,
    )
    store.save_deployment(dep)
    return dep


async def test_reconcile_loop_drives_to_ready(tmp_path):
    daemon, store, _ = _daemon(tmp_path, MockProvider(namespace="test"))
    _seed(store)
    for _ in range(10):
        await daemon.tick_reconcile()
        if store.get_deployment("dep-d1").observed_state is S.READY:
            break
    assert store.get_deployment("dep-d1").observed_state is S.READY


async def test_health_loop_degrades_after_threshold(tmp_path):
    provider = MockProvider(namespace="test")
    instance = await provider.create_instance(
        InstanceRequest(
            name="gpu-orch-test-dep-d1", gpu_type="MOCK-GPU", image="i", disk_gb=10, ports=[8000]
        )
    )
    await provider.get_instance(instance.provider_instance_id)  # -> RUNNING
    daemon, store, _ = _daemon(tmp_path, provider, runtime=_runtime(model_ok=False))
    _seed(store, state=S.READY, instance=instance, endpoint="https://pod-8000.mock.local")

    for _ in range(3):  # health_failure_threshold default
        await daemon.tick_health()

    assert store.get_deployment("dep-d1").observed_state is S.DEGRADED


async def test_orphan_sweep_destroys_after_grace(tmp_path):
    provider = MockProvider(namespace="test")
    # A pod in our namespace that no deployment owns (leaked by an interrupted run).
    await provider.create_instance(
        InstanceRequest(
            name="gpu-orch-test-dep-ghost", gpu_type="MOCK-GPU", image="i", disk_gb=10, ports=[8000]
        )
    )
    daemon, _, events = _daemon(tmp_path, provider, orphan_grace_period=120)

    # Within grace: detected but not destroyed.
    assert await daemon.tick_sweep(now=_T0) == []
    assert len(await provider.list_instances()) == 1

    # Past grace: destroyed.
    destroyed = await daemon.tick_sweep(now=_T0 + timedelta(seconds=121))
    assert len(destroyed) == 1
    assert await provider.list_instances() == []
    assert len(events.query(kind=EventKind.ORPHAN_DESTROYED)) == 1


async def test_retention_prunes_old_events(tmp_path):
    from gpu_orchestrator.models import EventKind
    from tests.fixtures.events import make_event

    daemon, store, _ = _daemon(tmp_path, MockProvider(namespace="test"), event_retention_days=30)
    store.append_event(make_event(EventKind.DEPLOYMENT_REQUESTED))  # stamped 2026-07-03

    removed = await daemon.tick_retention(now=_T0 + timedelta(days=60))

    assert removed == 1
    assert store.query_events() == []


_BUSINESS = Schedule(
    timezone="America/New_York",
    default_posture=Posture.OFF,
    rules=[ScheduleRule(days=[0, 1, 2, 3, 4], start="06:00", end="18:00", posture=Posture.ON)],
)
_IN_WINDOW = datetime(2026, 7, 6, 14, 0, tzinfo=UTC)  # Mon 10:00 EDT -> ON
_OUT_OF_WINDOW = datetime(2026, 7, 6, 23, 0, tzinfo=UTC)  # Mon 19:00 EDT -> OFF


async def test_schedule_wakes_a_stopped_deployment_in_window(tmp_path):
    daemon, store, events = _daemon(tmp_path, MockProvider(namespace="test"))
    # At rest: stopped and desired stopped, but a business schedule says it should be running now.
    _seed(store, state=S.STOPPED, desired=S.STOPPED, schedule=_BUSINESS)

    for _ in range(10):
        await daemon.tick_reconcile(now=_IN_WINDOW)
        if store.get_deployment("dep-d1").observed_state is S.READY:
            break

    dep = store.get_deployment("dep-d1")
    assert dep.desired_state is S.READY  # schedule flipped desired, authoritatively
    assert dep.observed_state is S.READY
    assert any(e.payload.get("action") == "policy_desired" for e in events.query())


async def test_schedule_stops_a_running_deployment_out_of_window(tmp_path):
    provider = MockProvider(namespace="test")
    instance = await provider.create_instance(
        InstanceRequest(
            name="gpu-orch-test-dep-d1", gpu_type="MOCK-GPU", image="i", disk_gb=10, ports=[8000]
        )
    )
    await provider.get_instance(instance.provider_instance_id)  # -> RUNNING
    daemon, store, _ = _daemon(tmp_path, provider)
    _seed(store, state=S.READY, instance=instance, endpoint="https://x", schedule=_BUSINESS)

    for _ in range(6):
        await daemon.tick_reconcile(now=_OUT_OF_WINDOW)
        if store.get_deployment("dep-d1").observed_state is S.STOPPED:
            break

    dep = store.get_deployment("dep-d1")
    assert dep.desired_state is S.STOPPED
    assert dep.observed_state is S.STOPPED
    assert await provider.list_instances() == []  # pod torn down (cost-safety)


async def test_resting_off_deployment_is_not_ticked(tmp_path):
    daemon, store, events = _daemon(tmp_path, MockProvider(namespace="test"))
    _seed(store, state=S.STOPPED, desired=S.STOPPED, schedule=_BUSINESS)

    # Out of window (OFF): the deployment is already where the schedule wants it, so it must not be
    # pulled into the reconcile set and must not churn the store or the event log.
    assert daemon._reconcilable(_OUT_OF_WINDOW) == []
    await daemon.tick_reconcile(now=_OUT_OF_WINDOW)

    assert store.get_deployment("dep-d1").observed_state is S.STOPPED
    assert events.query() == []


async def test_orphan_sweep_spares_owned_instances(tmp_path):
    provider = MockProvider(namespace="test")
    instance = await provider.create_instance(
        InstanceRequest(
            name="gpu-orch-test-dep-d1", gpu_type="MOCK-GPU", image="i", disk_gb=10, ports=[8000]
        )
    )
    daemon, store, _ = _daemon(tmp_path, provider, orphan_grace_period=0)
    _seed(store, state=S.READY, instance=instance, endpoint="https://x")

    destroyed = await daemon.tick_sweep(now=_T0)

    assert destroyed == []  # owned by an active deployment
    assert len(await provider.list_instances()) == 1


def _open_cost(store: Store, rate: float = 1.0) -> None:
    # A pod running since midnight UTC on 2026-07-06; at _IN_WINDOW (14:00) it has accrued rate*14.
    store.save_cost_record(
        CostRecord(
            deployment_id="dep-d1",
            gpu_hourly_usd=rate,
            started_at=datetime(2026, 7, 6, 0, 0, tzinfo=UTC),
            stopped_at=None,
        )
    )


async def test_budget_stop_holds_deployment_against_schedule(tmp_path):
    provider = MockProvider(namespace="test")
    instance = await provider.create_instance(
        InstanceRequest(
            name="gpu-orch-test-dep-d1", gpu_type="MOCK-GPU", image="i", disk_gb=10, ports=[8000]
        )
    )
    await provider.get_instance(instance.provider_instance_id)  # -> RUNNING
    daemon, store, events = _daemon(tmp_path, provider)
    _seed(store, state=S.READY, instance=instance, endpoint="https://x", schedule=_BUSINESS)
    _open_cost(store)  # ~$14 accrued this window
    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=5.0, on_exceed="stop")
    )

    await daemon.tick_budget(now=_IN_WINDOW)  # over the $5 ceiling -> hold

    held = store.get_deployment("dep-d1")
    assert held.budget_hold is True and held.desired_state is S.STOPPED
    assert any(e.kind is EventKind.BUDGET_EXCEEDED for e in events.query())

    # The schedule window is ON, but the budget hold outranks it: it must tear down and stay down.
    for _ in range(6):
        await daemon.tick_reconcile(now=_IN_WINDOW)
        if store.get_deployment("dep-d1").observed_state is S.STOPPED:
            break
    final = store.get_deployment("dep-d1")
    assert final.observed_state is S.STOPPED and final.budget_hold is True
    assert await provider.list_instances() == []


async def test_budget_warns_once_per_escalation(tmp_path):
    daemon, store, events = _daemon(tmp_path, MockProvider(namespace="test"))
    _open_cost(store)  # ~$14
    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=20.0, warn_fraction=0.5)
    )  # $14 of $20 -> over the 0.5 warn line, not exceeded

    await daemon.tick_budget(now=_IN_WINDOW)
    await daemon.tick_budget(now=_IN_WINDOW)  # same phase, same window -> no duplicate

    warns = [e for e in events.query() if e.kind is EventKind.BUDGET_WARNING]
    assert len(warns) == 1


async def test_budget_hold_releases_when_no_longer_over(tmp_path):
    daemon, store, events = _daemon(tmp_path, MockProvider(namespace="test"))
    _seed(store, state=S.STOPPED, desired=S.STOPPED)
    dep = store.get_deployment("dep-d1")
    dep.budget_hold = True  # as if a prior window's overspend held it
    store.save_deployment(dep)
    # A stop budget with no cost records this window: spend 0, not exceeded -> hold released.
    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=5.0, on_exceed="stop")
    )

    await daemon.tick_budget(now=_IN_WINDOW)

    assert store.get_deployment("dep-d1").budget_hold is False
    assert any(e.kind is EventKind.BUDGET_RELEASED for e in events.query())


async def test_budget_hold_release_gives_the_capacity_back(tmp_path):
    """A daily ceiling is a window, not a delete: once spend is back under it, the deployment the
    hold forced down comes back up. Releasing the flag while leaving desired_state STOPPED left
    capacity down permanently, which is not what a window means (or what the docs promise)."""
    daemon, store, events = _daemon(tmp_path, MockProvider(namespace="test"))
    _seed(store, state=S.READY, endpoint="https://x")
    _open_cost(store)  # ~$14 accrued this window
    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=5.0, on_exceed="stop")
    )
    await daemon.tick_budget(now=_IN_WINDOW)
    held = store.get_deployment("dep-d1")
    assert held.budget_hold is True and held.desired_state is S.STOPPED

    # Raising the ceiling puts spend back under it, which is the same shape as a window rollover.
    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=500.0, on_exceed="stop")
    )
    await daemon.tick_budget(now=_IN_WINDOW)

    released = store.get_deployment("dep-d1")
    assert released.budget_hold is False
    assert released.desired_state is S.READY, "the reconciler must be free to bring it back up"
    assert any(e.kind is EventKind.BUDGET_RELEASED for e in events.query())


async def test_budget_hold_does_not_resurrect_a_deployment_the_user_stopped(tmp_path):
    """The flip side of giving capacity back: a deployment already stopped when the ceiling hit is
    never marked held, so releasing cannot start it. Otherwise a budget window would silently undo a
    `gpu stop`."""
    daemon, store, _ = _daemon(tmp_path, MockProvider(namespace="test"))
    _seed(store, state=S.STOPPED, desired=S.STOPPED)
    _open_cost(store)
    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=5.0, on_exceed="stop")
    )

    await daemon.tick_budget(now=_IN_WINDOW)  # over the ceiling
    assert store.get_deployment("dep-d1").budget_hold is False  # nothing to force

    store.save_budget(
        Budget(id="bud-1", window=BudgetWindow.DAILY, limit_usd=500.0, on_exceed="stop")
    )
    await daemon.tick_budget(now=_IN_WINDOW)  # back under

    still_stopped = store.get_deployment("dep-d1")
    assert still_stopped.desired_state is S.STOPPED


def _autoscale_daemon(tmp_path, **cfg):
    return _daemon(
        tmp_path,
        MockProvider(namespace="test"),
        autoscale_window_seconds=60,
        **cfg,
    )


def _policy(minimum=1, maximum=3, target=2.0):
    return AutoscalePolicy(
        model_id="qwen3-0.6b",
        min_replicas=minimum,
        max_replicas=maximum,
        target_rpm_per_replica=target,
    )


async def test_autoscale_adds_replicas_under_load(tmp_path):
    daemon, store, events = _autoscale_daemon(tmp_path, autoscale_hysteresis_ticks=1)
    _seed(store, state=S.READY, endpoint="https://x")  # one member: dep-d1
    store.save_autoscale_policy(_policy())
    for _ in range(5):  # 5 requests in the last minute -> rpm 5 -> ceil(5/2)=3 (the cap)
        store.save_usage_record("dep-d1", 10, 5, _IN_WINDOW)

    await daemon.tick_autoscale(now=_IN_WINDOW)

    members = [
        d
        for d in store.list_deployments()
        if d.model_id == "qwen3-0.6b" and d.desired_state is not S.STOPPED
    ]
    assert len(members) == 3
    assert any(e.kind is EventKind.AUTOSCALED for e in events.query())


async def test_autoscale_removes_idle_replicas(tmp_path):
    daemon, store, _ = _autoscale_daemon(tmp_path, autoscale_hysteresis_ticks=1)
    for i in range(3):
        store.save_deployment(
            Deployment(
                id=f"dep-r{i}",
                model_id="qwen3-0.6b",
                provider="mock",
                desired_state=S.READY,
                observed_state=S.READY,
                profile=_PROFILE,
                endpoint_url="https://x",
                created_at=datetime(2026, 7, 6, 10 + i, 0, tzinfo=UTC),
            )
        )
    store.save_autoscale_policy(_policy())

    await daemon.tick_autoscale(now=_IN_WINDOW)  # no recent usage -> rpm 0 -> target = min 1

    active = [
        d
        for d in store.list_deployments()
        if d.model_id == "qwen3-0.6b" and d.desired_state is not S.STOPPED
    ]
    assert len(active) == 1  # scaled down to the floor, newest surplus stopped


async def test_autoscale_waits_for_hysteresis(tmp_path):
    daemon, store, _ = _autoscale_daemon(tmp_path, autoscale_hysteresis_ticks=2)
    _seed(store, state=S.READY, endpoint="https://x")
    store.save_autoscale_policy(_policy())
    for _ in range(5):
        store.save_usage_record("dep-d1", 10, 5, _IN_WINDOW)

    def member_count():
        return len(
            [
                d
                for d in store.list_deployments()
                if d.model_id == "qwen3-0.6b" and d.desired_state is not S.STOPPED
            ]
        )

    await daemon.tick_autoscale(now=_IN_WINDOW)  # first tick at target=3, hysteresis needs 2
    assert member_count() == 1
    await daemon.tick_autoscale(now=_IN_WINDOW)  # second consecutive tick: acts
    assert member_count() == 3


async def test_autoscale_skips_scheduled_models(tmp_path):
    daemon, store, _ = _autoscale_daemon(tmp_path, autoscale_hysteresis_ticks=1)
    _seed(store, state=S.READY, endpoint="https://x", schedule=_BUSINESS)  # scheduled
    store.save_autoscale_policy(_policy())
    for _ in range(5):
        store.save_usage_record("dep-d1", 10, 5, _IN_WINDOW)

    await daemon.tick_autoscale(now=_IN_WINDOW)

    members = [
        d
        for d in store.list_deployments()
        if d.model_id == "qwen3-0.6b" and d.desired_state is not S.STOPPED
    ]
    assert len(members) == 1  # a schedule owns this model's lifecycle; autoscale leaves it alone


async def test_autoscale_does_not_spend_through_a_budget_hold(tmp_path):
    """A budget stop holds a deployment by forcing desired_state STOPPED, which drops it out of the
    autoscaler's member count. The autoscaler then read the pool as short of its floor and cloned a
    replacement with no hold, so a per-deployment ceiling was spent straight through. A budget stop
    outranks the autoscaler, exactly as it outranks a schedule."""
    daemon, store, _ = _autoscale_daemon(tmp_path, autoscale_hysteresis_ticks=1)
    for i in range(2):  # a two-replica pool, matching the policy floor below
        store.save_deployment(
            Deployment(
                id=f"dep-r{i}",
                model_id="qwen3-0.6b",
                provider="mock",
                desired_state=S.READY,
                observed_state=S.READY,
                profile=_PROFILE,
                endpoint_url="https://x",
                created_at=datetime(2026, 7, 6, 10 + i, 0, tzinfo=UTC),
            )
        )
    store.save_autoscale_policy(_policy(minimum=2, maximum=4))
    # A ceiling scoped to one replica of the pool, already over.
    store.save_cost_record(
        CostRecord(
            deployment_id="dep-r0",
            gpu_hourly_usd=1.0,
            started_at=datetime(2026, 7, 6, 0, 0, tzinfo=UTC),
            stopped_at=None,
        )
    )
    store.save_budget(
        Budget(
            id="bud-1",
            deployment_id="dep-r0",
            window=BudgetWindow.DAILY,
            limit_usd=5.0,
            on_exceed="stop",
        )
    )

    await daemon.tick_budget(now=_IN_WINDOW)
    assert store.get_deployment("dep-r0").budget_hold is True

    await daemon.tick_autoscale(now=_IN_WINDOW)

    rows = store.list_deployments(include_stopped=True)
    everything = [d for d in rows if d.model_id == "qwen3-0.6b"]
    assert len(everything) == 2, "no replacement replica while a ceiling holds one down"
    wants_to_run = [d for d in everything if d.desired_state is not S.STOPPED]
    assert len(wants_to_run) == 1  # only the unheld replica
