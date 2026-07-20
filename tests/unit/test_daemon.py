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
    assert any(e.payload.get("action") == "schedule_posture" for e in events.query())


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
