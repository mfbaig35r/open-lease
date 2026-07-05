"""UX #2: persistent model-cache network volumes. A shared per-namespace volume is created and
attached (with the HF cache env) when caching is enabled, and not touched when it is off. Tested
against the mock; the live warm/cold speedup is validated separately against real RunPod."""

from __future__ import annotations

import httpx
import pytest

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.catalog import Catalog
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.core.reconciler import reconcile_once
from gpu_orchestrator.errors import ReconcileError
from gpu_orchestrator.events import EventLog
from gpu_orchestrator.models import Deployment, DeploymentState, VolumeInfo
from gpu_orchestrator.providers.mock import MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime
from gpu_orchestrator.store import Store
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC

S = DeploymentState
_PROFILE = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "MOCK-GPU"})


def _runtime() -> VLLMRuntime:
    return VLLMRuntime(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []}))
    )


def _deploy_once(tmp_path, provider, cfg, profile=_PROFILE) -> Deployment:
    store = Store(cfg.state_db)
    dep = Deployment(
        id="dep-v1",
        model_id="qwen3-0.6b",
        provider="mock",
        desired_state=S.READY,
        observed_state=S.REQUESTED,
        profile=profile,
    )
    store.save_deployment(dep)
    ctx = {
        "provider": provider,
        "runtime": _runtime(),
        "catalog": Catalog({"qwen3-0.6b": QWEN3_06B_SPEC}, {"qwen3-0.6b": profile}),
        "config": cfg,
        "store": store,
        "events": EventLog(store),
    }
    return reconcile_once(dep, **ctx)  # returns a coroutine; awaited by callers


async def test_cache_disabled_attaches_nothing(tmp_path):
    provider = MockProvider(namespace="test")
    cfg = Config(namespace="test", state_db=tmp_path / "v.db")  # cache off by default
    dep = await _deploy_once(tmp_path, provider, cfg)
    assert provider._pods[dep.instance.provider_instance_id].network_volume_id is None
    assert await provider.list_volumes() == []


async def test_cache_enabled_creates_and_attaches(tmp_path):
    provider = MockProvider(namespace="test")
    cfg = Config(
        namespace="test",
        state_db=tmp_path / "v.db",
        cache_volume_enabled=True,
        runpod_data_center_id="DC1",
    )
    dep = await _deploy_once(tmp_path, provider, cfg)

    volumes = await provider.list_volumes()
    assert len(volumes) == 1
    assert volumes[0].name == "gpu-orch-test-cache"
    assert volumes[0].data_center_id == "DC1"
    # the pod attached that exact volume
    assert provider._pods[dep.instance.provider_instance_id].network_volume_id == volumes[0].id


async def test_ensure_cache_volume_is_idempotent():
    provider = MockProvider(namespace="test")
    first = await provider.ensure_cache_volume("cache", 100, "DC1")
    second = await provider.ensure_cache_volume("cache", 100, "DC1")
    assert first == second
    assert len(await provider.list_volumes()) == 1  # not duplicated


async def test_ensure_cache_volume_distinct_per_data_center():
    # A same-name volume in another DC must not be reused (found live: it pins the pod to the wrong
    # region and the create fails).
    provider = MockProvider(namespace="test")
    a = await provider.ensure_cache_volume("cache", 100, "DC1")
    b = await provider.ensure_cache_volume("cache", 100, "DC2")
    assert a != b
    assert len(await provider.list_volumes()) == 2


async def test_orchestrator_list_and_delete_volume(tmp_path):
    provider = MockProvider(namespace="test")
    await provider.ensure_cache_volume("gpu-orch-test-cache", 50, "DC1")
    orch = Orchestrator(
        Config(namespace="test", state_db=tmp_path / "v.db"),
        provider=provider,
        runtime=_runtime(),
    )
    volumes = await orch.list_volumes(provider="mock")
    assert len(volumes) == 1
    await orch.delete_volume(volumes[0].id, provider="mock")
    assert await orch.list_volumes(provider="mock") == []


def test_volume_monthly_cost():
    assert VolumeInfo(id="v", name="n", size_gb=100).estimated_monthly_usd == 7.0


# --- GPU availability + cache DC auto-selection ---------------------------------------


async def test_gpu_availability_filters_by_gpu():
    provider = MockProvider(namespace="test")
    rows = await provider.gpu_availability("RTX-A4000")
    assert {r.data_center_id for r in rows} == {"MOCK-DC-1", "MOCK-DC-2"}
    assert [r for r in rows if r.available][0].data_center_id == "MOCK-DC-1"
    assert await provider.gpu_availability("no-such-gpu") == []


async def test_orchestrator_availability_resolves_model(tmp_path):
    orch = Orchestrator(
        Config(namespace="test", state_db=tmp_path / "v.db"),
        provider=MockProvider(namespace="test"),
        runtime=_runtime(),
    )
    rows = await orch.gpu_availability(model_id="qwen3-0.6b", provider="mock")  # -> RTX-A4000
    assert any(r.available for r in rows)


async def test_cache_auto_selects_available_dc(tmp_path):
    # No explicit DC: pick one that currently has the GPU (RTX-A4000 -> MOCK-DC-1).
    provider = MockProvider(namespace="test")
    cfg = Config(namespace="test", state_db=tmp_path / "v.db", cache_volume_enabled=True)
    await _deploy_once(tmp_path, provider, cfg, profile=QWEN3_06B_PROFILE)  # recommends RTX-A4000
    volumes = await provider.list_volumes()
    assert len(volumes) == 1
    assert volumes[0].data_center_id == "MOCK-DC-1"  # the available one


async def test_cache_auto_select_fails_without_capacity(tmp_path):
    # No explicit DC and the GPU is not available anywhere -> clear error, no volume, no pod.
    provider = MockProvider(namespace="test")
    cfg = Config(namespace="test", state_db=tmp_path / "v.db", cache_volume_enabled=True)
    profile = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "A100-80GB"})
    with pytest.raises(ReconcileError, match="no data center"):
        await _deploy_once(tmp_path, provider, cfg, profile=profile)
    assert await provider.list_volumes() == []
