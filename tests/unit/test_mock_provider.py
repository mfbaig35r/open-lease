"""Mock-provider specifics: the determinism and failure seams reconciler tests rely on (§8.2)."""

from __future__ import annotations

import pytest

from gpu_orchestrator.errors import InstanceCreationError, ProviderAPIError
from gpu_orchestrator.models import InstanceRequest
from gpu_orchestrator.providers import MockProvider


def _request(provider: MockProvider, deployment_id: str = "dep-1") -> InstanceRequest:
    return InstanceRequest(
        name=provider.instance_name(deployment_id),
        gpu_type="mock-gpu",
        image="img",
        disk_gb=20,
        ports=[8000],
    )


async def test_state_advances_to_running_and_endpoint_resolves():
    provider = MockProvider(namespace="test", steps_to_running=1)
    instance = await provider.create_instance(_request(provider))
    # Not routable yet.
    assert await provider.resolve_endpoint_url(instance, 8000) is None
    # One observation advances it to RUNNING.
    running = await provider.get_instance(instance.provider_instance_id)
    assert running.state == "RUNNING"
    url = await provider.resolve_endpoint_url(running, 8000)
    assert url is not None and url.startswith("https://")


async def test_kill_makes_instance_gone():
    provider = MockProvider(namespace="test")
    instance = await provider.create_instance(_request(provider))
    provider.kill(instance.provider_instance_id)
    assert await provider.get_instance(instance.provider_instance_id) is None


async def test_list_is_scoped_to_namespace_prefix():
    provider = MockProvider(namespace="alpha")
    instance = await provider.create_instance(_request(provider, "dep-x"))
    listed = await provider.list_instances()
    assert [i.provider_instance_id for i in listed] == [instance.provider_instance_id]


async def test_fail_create_raises():
    provider = MockProvider(namespace="test", fail_create=True)
    with pytest.raises(InstanceCreationError):
        await provider.create_instance(_request(provider))


async def test_fail_api_raises_on_get():
    provider = MockProvider(namespace="test", fail_api=True)
    with pytest.raises(ProviderAPIError):
        await provider.get_instance("mock-1")


async def test_set_logs_and_tail():
    provider = MockProvider(namespace="test")
    instance = await provider.create_instance(_request(provider))
    provider.set_logs(instance.provider_instance_id, ["a", "b", "c"])
    assert await provider.get_logs(instance.provider_instance_id, tail=2) == ["b", "c"]
