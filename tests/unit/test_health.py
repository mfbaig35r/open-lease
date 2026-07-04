"""Step-6 health engine (spec §10): the four checks (run_checks) and the flap-absorbing monitor.
A single failed probe must not flip a READY deployment to DEGRADED; only N consecutive ones do."""

from __future__ import annotations

import httpx

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.health import HealthMonitor, run_checks
from gpu_orchestrator.events import EventLog
from gpu_orchestrator.models import (
    Deployment,
    DeploymentState,
    EventKind,
    HealthState,
    InstanceRequest,
)
from gpu_orchestrator.providers.mock import MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime
from gpu_orchestrator.store import Store
from tests.fixtures.catalog import QWEN3_06B_PROFILE

S = DeploymentState
_PROFILE = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "MOCK-GPU"})


def _runtime(*, http_ok: bool = True, model_ok: bool = True) -> VLLMRuntime:
    def handler(request: httpx.Request) -> httpx.Response:
        code = 200 if http_ok else 503
        if request.url.path == "/health":
            return httpx.Response(code)
        if request.url.path == "/v1/models":
            data = [{"id": "qwen3-0.6b"}] if model_ok else []
            return httpx.Response(code, json={"data": data})
        return httpx.Response(404)

    return VLLMRuntime(transport=httpx.MockTransport(handler))


async def _running_deployment(
    provider: MockProvider, *, dep_id: str = "dep-h1", state: S = S.READY
) -> Deployment:
    instance = await provider.create_instance(
        InstanceRequest(
            name=f"gpu-orch-test-{dep_id}", gpu_type="MOCK-GPU", image="i", disk_gb=10, ports=[8000]
        )
    )
    await provider.get_instance(instance.provider_instance_id)  # advance mock pod to RUNNING
    return Deployment(
        id=dep_id,
        model_id="qwen3-0.6b",
        provider="mock",
        desired_state=S.READY,
        observed_state=state,
        profile=_PROFILE,
        instance=instance,
        endpoint_url="https://pod-8000.mock.local",
    )


# --- run_checks -----------------------------------------------------------------------


async def test_run_checks_healthy():
    provider = MockProvider(namespace="test")
    dep = await _running_deployment(provider)
    status = await run_checks(dep, provider, _runtime())
    assert status.status is HealthState.HEALTHY
    assert set(status.checks) == {"instance_alive", "http_alive", "model_loaded", "latency"}
    assert all(c.ok for c in status.checks.values())


async def test_run_checks_degraded_when_model_not_loaded():
    provider = MockProvider(namespace="test")
    dep = await _running_deployment(provider)
    status = await run_checks(dep, provider, _runtime(model_ok=False))
    assert status.status is HealthState.DEGRADED
    assert status.checks["instance_alive"].ok is True
    assert status.checks["model_loaded"].ok is False


async def test_run_checks_failed_when_instance_gone():
    provider = MockProvider(namespace="test")
    dep = await _running_deployment(provider)
    provider.kill(dep.instance.provider_instance_id)
    status = await run_checks(dep, provider, _runtime())
    assert status.status is HealthState.FAILED
    assert status.checks["instance_alive"].ok is False


# --- HealthMonitor: flap absorption ---------------------------------------------------


async def test_monitor_declares_degraded_only_after_threshold(tmp_path):
    provider = MockProvider(namespace="test")
    store = Store(tmp_path / "h.db")
    events = EventLog(store)
    config = Config(namespace="test", state_db=tmp_path / "h.db")  # threshold defaults to 3
    monitor = HealthMonitor(config)
    dep = await _running_deployment(provider)
    sick = _runtime(model_ok=False)

    for _ in range(config.health_failure_threshold - 1):
        await monitor.check_once(dep, provider=provider, runtime=sick, store=store, events=events)
        assert dep.observed_state is S.READY  # not yet: absorbing flaps

    await monitor.check_once(dep, provider=provider, runtime=sick, store=store, events=events)
    assert dep.observed_state is S.DEGRADED
    degraded = events.query(dep.id, kind=EventKind.HEALTH_DEGRADED)
    assert len(degraded) == 1


async def test_monitor_recovers_from_degraded(tmp_path):
    provider = MockProvider(namespace="test")
    store = Store(tmp_path / "h.db")
    events = EventLog(store)
    monitor = HealthMonitor(Config(namespace="test", state_db=tmp_path / "h.db"))
    dep = await _running_deployment(provider, state=S.DEGRADED)

    await monitor.check_once(dep, provider=provider, runtime=_runtime(), store=store, events=events)

    assert dep.observed_state is S.READY
    assert len(events.query(dep.id, kind=EventKind.HEALTH_PASSED)) == 1


async def test_monitor_ignores_non_serving_deployment(tmp_path):
    provider = MockProvider(namespace="test")
    store = Store(tmp_path / "h.db")
    events = EventLog(store)
    monitor = HealthMonitor(Config(namespace="test", state_db=tmp_path / "h.db"))
    dep = await _running_deployment(provider, state=S.PROVISIONING)

    status = await monitor.check_once(
        dep, provider=provider, runtime=_runtime(model_ok=False), store=store, events=events
    )

    assert status.status is HealthState.BOOTING  # not monitored
    assert dep.observed_state is S.PROVISIONING  # untouched
