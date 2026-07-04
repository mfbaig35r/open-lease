"""Step-5 reconcile_once integration: the observe -> decide -> execute -> record tick against the
mock provider and the real vLLM runtime (driven by an httpx MockTransport). Covers the happy path,
adoption, out-of-band death recovery, retry-then-fail, stop, and the stage-timeout cost guard."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.catalog import Catalog
from gpu_orchestrator.core.reconciler import reconcile_once
from gpu_orchestrator.events import EventLog
from gpu_orchestrator.models import Deployment, DeploymentState, EventKind, InstanceRequest
from gpu_orchestrator.providers.mock import MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime
from gpu_orchestrator.store import Store
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC

S = DeploymentState
_PROFILE = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "MOCK-GPU"})


def _transport(*, ready: bool = True) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            data = [{"id": "qwen3-0.6b"}] if ready else []
            return httpx.Response(200, json={"data": data})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _ctx(tmp_path, provider: MockProvider, *, ready: bool = True) -> dict:
    store = Store(tmp_path / "state.db")
    return {
        "provider": provider,
        "runtime": VLLMRuntime(transport=_transport(ready=ready)),
        "catalog": Catalog({"qwen3-0.6b": QWEN3_06B_SPEC}, {"qwen3-0.6b": _PROFILE}),
        "config": Config(namespace="test", state_db=tmp_path / "state.db"),
        "store": store,
        "events": EventLog(store),
    }


def _new_deployment(deployment_id: str = "dep-test01") -> Deployment:
    return Deployment(
        id=deployment_id,
        model_id="qwen3-0.6b",
        provider="mock",
        desired_state=S.READY,
        observed_state=S.REQUESTED,
        profile=_PROFILE,
    )


async def _drive(dep: Deployment, ctx: dict, *, until: set[S], ticks: int = 15) -> Deployment:
    for _ in range(ticks):
        if dep.observed_state in until:
            return dep
        dep = await reconcile_once(dep, **ctx)
    return dep


async def test_reconcile_reaches_ready(tmp_path):
    ctx = _ctx(tmp_path, MockProvider(namespace="test"))
    dep = _new_deployment()
    ctx["store"].save_deployment(dep)

    dep = await _drive(dep, ctx, until={S.READY, S.FAILED})

    assert dep.observed_state == S.READY
    assert dep.instance is not None
    assert dep.endpoint_url is not None
    assert dep.failure is None
    kinds = [e.kind for e in ctx["events"].query("dep-test01")]
    assert EventKind.INSTANCE_CREATED in kinds
    assert EventKind.DEPLOYMENT_READY in kinds


async def test_reconcile_adopts_existing_instance(tmp_path):
    provider = MockProvider(namespace="test")
    # A pod already exists under the deterministic name: an interrupted deploy (spec §7.5).
    await provider.create_instance(
        InstanceRequest(
            name="gpu-orch-test-dep-test01",
            gpu_type="MOCK-GPU",
            image="img",
            disk_gb=10,
            ports=[8000],
        )
    )
    ctx = _ctx(tmp_path, provider)
    dep = _new_deployment()  # instance is None on the record
    ctx["store"].save_deployment(dep)

    dep = await reconcile_once(dep, **ctx)

    assert dep.instance is not None
    adopted = ctx["events"].query("dep-test01", kind=EventKind.INSTANCE_ADOPTED)
    assert len(adopted) == 1


async def test_reconcile_recreates_after_out_of_band_death(tmp_path):
    provider = MockProvider(namespace="test")
    ctx = _ctx(tmp_path, provider)
    dep = _new_deployment()
    ctx["store"].save_deployment(dep)
    dep = await _drive(dep, ctx, until={S.READY, S.FAILED})
    assert dep.observed_state == S.READY

    provider.kill(dep.instance.provider_instance_id)  # someone nukes the pod from a console
    dep = await reconcile_once(dep, **ctx)

    # The reconciler trusts reality: pod gone -> it provisions a fresh one.
    assert dep.instance is not None
    assert dep.observed_state == S.PROVISIONING


async def test_reconcile_retries_then_fails_on_persistent_create_error(tmp_path):
    ctx = _ctx(tmp_path, MockProvider(namespace="test", fail_create=True))
    dep = _new_deployment()
    ctx["store"].save_deployment(dep)

    dep = await _drive(dep, ctx, until={S.FAILED})

    assert dep.observed_state == S.FAILED
    assert dep.failure is not None
    assert dep.failure.attempts == 3  # config.retry_max_attempts
    failed = ctx["events"].query("dep-test01", kind=EventKind.DEPLOYMENT_FAILED)
    assert len(failed) == 1


async def test_reconcile_stop_destroys_and_settles(tmp_path):
    provider = MockProvider(namespace="test")
    ctx = _ctx(tmp_path, provider)
    dep = _new_deployment()
    ctx["store"].save_deployment(dep)
    dep = await _drive(dep, ctx, until={S.READY, S.FAILED})
    assert dep.observed_state == S.READY

    dep.desired_state = S.STOPPED
    ctx["store"].save_deployment(dep)
    dep = await _drive(dep, ctx, until={S.STOPPED, S.FAILED})

    assert dep.observed_state == S.STOPPED
    assert dep.instance is None
    stopped = ctx["events"].query("dep-test01", kind=EventKind.DEPLOYMENT_STOPPED)
    assert len(stopped) >= 1


async def test_reconcile_stage_budget_escalates_stuck_provisioning(tmp_path):
    # A pod that never reaches RUNNING: the provisioning budget is the cost guard (spec §7.3).
    ctx = _ctx(tmp_path, MockProvider(namespace="test", steps_to_running=999))
    dep = _new_deployment()
    ctx["store"].save_deployment(dep)

    t0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)
    dep = await reconcile_once(dep, **ctx, now=t0)
    assert dep.observed_state == S.PROVISIONING

    later = t0 + timedelta(seconds=ctx["config"].timeout_provisioning + 60)
    dep = await reconcile_once(dep, **ctx, now=later)

    assert dep.failure is not None
    assert dep.failure.stage == S.PROVISIONING
    assert dep.failure.retryable is True
