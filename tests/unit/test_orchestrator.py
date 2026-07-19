"""Step-5 Orchestrator facade tests: the public API (spec §7.1) driven against the mock provider
and the vLLM runtime. Verifies deploy (blocking + non-blocking), stop, delete, restart, and the
read-only surface (models, cost estimate, providers)."""

from __future__ import annotations

import httpx
import pytest

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.catalog import Catalog
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.errors import DeploymentNotFoundError, ModelNotFoundError
from gpu_orchestrator.models import DeploymentState, EventKind, HealthState
from gpu_orchestrator.providers.mock import MockProvider
from gpu_orchestrator.runtimes.vllm import VLLMRuntime
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC

S = DeploymentState
_PROFILE = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "MOCK-GPU"})


def _transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3-0.6b"}]})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _orch(tmp_path) -> Orchestrator:
    return Orchestrator(
        Config(namespace="test", state_db=tmp_path / "orch.db", reconcile_interval=0),
        catalog=Catalog({"qwen3-0.6b": QWEN3_06B_SPEC}, {"qwen3-0.6b": _PROFILE}),
        provider=MockProvider(namespace="test"),
        runtime=VLLMRuntime(transport=_transport()),
    )


async def test_deploy_blocking_reaches_ready(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    assert dep.observed_state == S.READY
    assert dep.endpoint_url is not None
    assert orch.get_deployment(dep.id).observed_state == S.READY


async def test_deploy_non_blocking_returns_immediately(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=False)
    assert dep.observed_state == S.REQUESTED
    assert dep.instance is None
    kinds = [e.kind for e in orch.events(dep.id)]
    assert EventKind.DEPLOYMENT_REQUESTED in kinds


async def test_deploy_unknown_model_raises(tmp_path):
    orch = _orch(tmp_path)
    with pytest.raises(ModelNotFoundError):
        await orch.deploy_model("no-such-model", provider="mock")


async def test_deploy_adhoc_no_catalog_entry_reaches_ready(tmp_path):
    # The engine is model-neutral: deploy any HF repo with no catalog entry. The deployment carries
    # its own hf_repo, so reconcile builds it without a catalog lookup.
    orch = _orch(tmp_path)
    dep = await orch.deploy_adhoc(
        hf_repo="Qwen/Qwen3-14B", gpu="MOCK-GPU", provider="mock", wait=True
    )
    assert dep.observed_state == S.READY
    assert dep.endpoint_url is not None
    assert dep.model_id == "qwen3-14b"  # derived from the repo's last segment
    assert dep.hf_repo == "Qwen/Qwen3-14B"  # self-contained, no catalog needed
    assert "qwen3-14b" not in {m.id for m in orch.list_models()}  # genuinely off-catalog


async def test_deploy_adhoc_gpus_sets_tensor_parallel_and_provisions_multi_gpu(tmp_path):
    # --gpus N on an ad-hoc deploy provisions an N-GPU pod (tensor parallelism).
    orch = _orch(tmp_path)
    dep = await orch.deploy_adhoc(
        hf_repo="Qwen/Qwen3-235B", gpu="MOCK-GPU", provider="mock", gpu_count=4, wait=True
    )
    assert dep.profile.tensor_parallel == 4
    assert dep.instance is not None and dep.instance.gpu_count == 4


async def test_stop_deployment(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    stopped = await orch.stop_deployment(dep.id)
    assert stopped.observed_state == S.STOPPED
    assert stopped.instance is None


async def test_delete_deployment_removes_record(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    await orch.delete_deployment(dep.id)
    with pytest.raises(DeploymentNotFoundError):
        orch.get_deployment(dep.id)


async def test_restart_deployment_returns_to_ready(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    restarted = await orch.restart_deployment(dep.id)
    assert restarted.observed_state == S.READY
    assert restarted.instance is not None


def test_list_models(tmp_path):
    orch = _orch(tmp_path)
    ids = {m.id for m in orch.list_models()}
    assert "qwen3-0.6b" in ids


async def test_estimate_cost(tmp_path):
    orch = _orch(tmp_path)
    estimate = await orch.estimate_cost("qwen3-0.6b", provider="mock", hours=2.0)
    assert estimate.gpu_hourly_usd == 0.50  # mock GPU rate
    assert estimate.estimated_usd == 1.0


async def test_list_providers_includes_mock(tmp_path):
    orch = _orch(tmp_path)
    providers = await orch.list_providers()
    assert any(p.name == "mock" for p in providers)


async def test_deploy_opens_cost_record(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    records = orch.get_costs(dep.id)
    assert len(records) == 1
    assert records[0].stopped_at is None  # accrual open while the instance runs
    assert records[0].gpu_hourly_usd == 0.50


async def test_stop_closes_cost_record(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    await orch.stop_deployment(dep.id)
    (record,) = orch.get_costs(dep.id)
    assert record.stopped_at is not None


async def test_get_health_reports_healthy(tmp_path):
    orch = _orch(tmp_path)
    dep = await orch.deploy_model("qwen3-0.6b", provider="mock", wait=True)
    status = await orch.get_health(dep.id)
    assert status.status is HealthState.HEALTHY
