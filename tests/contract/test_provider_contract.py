"""Provider contract suite (spec §8.3).

One parametrized suite that runs against every provider: mock always, runpod when
``GPU_ORCH_INTEGRATION=1`` (costs real money). This suite IS the provider spec: a new provider
passes it or it is not done.

Integration runs create and destroy real pods, so every test that creates cleans up in a finally.
"""

from __future__ import annotations

import os

import pytest

from gpu_orchestrator.models import InstanceRequest
from gpu_orchestrator.providers import MockProvider, RunPodProvider

INTEGRATION = os.environ.get("GPU_ORCH_INTEGRATION") == "1"

_BUILDERS = [pytest.param(lambda: MockProvider(namespace="test"), id="mock")]
if INTEGRATION:
    _BUILDERS.append(pytest.param(lambda: RunPodProvider(namespace="test"), id="runpod"))


@pytest.fixture(params=_BUILDERS)
def provider(request):
    return request.param()


async def _request(provider, deployment_id: str = "dep-contract") -> InstanceRequest:
    caps = await provider.capabilities()
    gpu = caps.gpu_types[0]
    return InstanceRequest(
        name=provider.instance_name(deployment_id),
        gpu_type=gpu.provider_sku,
        image="vllm/vllm-openai:v0.9.1",
        disk_gb=20,
        ports=[8000],
    )


async def test_capabilities_lists_gpu_types(provider):
    caps = await provider.capabilities()
    assert caps.gpu_types


async def test_create_observe_destroy_roundtrip(provider):
    instance = await provider.create_instance(await _request(provider))
    try:
        observed = await provider.get_instance(instance.provider_instance_id)
        assert observed is not None
        assert observed.provider_instance_id == instance.provider_instance_id
    finally:
        await provider.destroy_instance(instance.provider_instance_id)
    assert await provider.get_instance(instance.provider_instance_id) is None


async def test_destroy_is_idempotent(provider):
    # Destroying something that never existed is a no-op, not an error.
    await provider.destroy_instance("does-not-exist")


async def test_get_unknown_instance_returns_none(provider):
    assert await provider.get_instance("does-not-exist") is None


async def test_find_by_deployment_id(provider):
    instance = await provider.create_instance(await _request(provider, "dep-find"))
    try:
        found = await provider.find_instance_by_deployment_id("dep-find")
        assert found is not None
        assert found.provider_instance_id == instance.provider_instance_id
        assert await provider.find_instance_by_deployment_id("dep-absent") is None
    finally:
        await provider.destroy_instance(instance.provider_instance_id)


async def test_resolve_endpoint_none_before_routable(provider):
    instance = await provider.create_instance(await _request(provider, "dep-endpoint"))
    try:
        # Freshly created: not yet routable.
        assert await provider.resolve_endpoint_url(instance, 8000) is None
    finally:
        await provider.destroy_instance(instance.provider_instance_id)


async def test_get_logs_returns_a_list(provider):
    instance = await provider.create_instance(await _request(provider, "dep-logs"))
    try:
        logs = await provider.get_logs(instance.provider_instance_id)
        assert isinstance(logs, list)
    finally:
        await provider.destroy_instance(instance.provider_instance_id)
