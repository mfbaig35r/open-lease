"""Mock provider: in-memory fake for tests and offline development (spec §8.2).

Deterministic, no network. State advances one step per ``get_instance`` call
(PROVISIONING -> RUNNING after ``steps_to_running`` observations), which is enough to exercise the
"None until routable" contract. Failure hooks (``fail_create``, ``fail_api``, ``kill``) are the
seams the reconciler's failure-injection tests drive at build step 5.
"""

from __future__ import annotations

from ..errors import InstanceCreationError, ProviderAPIError
from ..models import GPUType, Instance, InstanceRequest, ProviderCapabilities, VolumeInfo
from .base import Provider

_PROVISIONING = "PROVISIONING"
_RUNNING = "RUNNING"

_MOCK_GPU = GPUType(
    id="MOCK-GPU",
    name="Mock GPU 24GB",
    memory_gb=24,
    hourly_usd=0.50,
    provider_sku="mock-gpu",
)

# Catalog-parity GPUs so the real catalog's profiles (which recommend RunPod GPU ids) resolve
# against the mock, letting the full deploy flow run offline and in tests (spec §8.2). Rates are
# fictional; the mock never bills.
_CATALOG_PARITY = [
    GPUType(
        id="RTX-A4000",
        name="Mock RTX A4000",
        memory_gb=16,
        hourly_usd=0.17,
        provider_sku="RTX-A4000",
    ),
    GPUType(
        id="A100-80GB",
        name="Mock A100 80GB",
        memory_gb=80,
        hourly_usd=1.89,
        provider_sku="A100-80GB",
    ),
]


class _Pod:
    def __init__(
        self,
        instance_id: str,
        name: str,
        gpu_type: str,
        ports: list[int],
        network_volume_id: str | None = None,
    ) -> None:
        self.id = instance_id
        self.name = name
        self.gpu_type = gpu_type
        self.ports = ports
        self.network_volume_id = network_volume_id  # what cache volume (if any) this pod attached
        self.observations = 0
        self.logs: list[str] = []

    def state(self, steps_to_running: int) -> str:
        return _RUNNING if self.observations >= steps_to_running else _PROVISIONING


class MockProvider(Provider):
    name = "mock"

    def __init__(
        self,
        *,
        namespace: str = "test",
        steps_to_running: int = 1,
        fail_create: bool = False,
        fail_api: bool = False,
    ) -> None:
        super().__init__(namespace=namespace)
        self.steps_to_running = steps_to_running
        self.fail_create = fail_create
        self.fail_api = fail_api
        self._pods: dict[str, _Pod] = {}
        self._counter = 0
        self._volumes: dict[str, VolumeInfo] = {}
        self._volume_counter = 0

    # --- test seams -----------------------------------------------------------------

    def kill(self, provider_instance_id: str) -> None:
        """Simulate an out-of-band death (e.g. someone killing the pod from a console)."""
        self._pods.pop(provider_instance_id, None)

    def set_logs(self, provider_instance_id: str, lines: list[str]) -> None:
        pod = self._pods.get(provider_instance_id)
        if pod is not None:
            pod.logs = list(lines)

    # --- Provider interface ---------------------------------------------------------

    async def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            gpu_types=[_MOCK_GPU, *_CATALOG_PARITY], regions=["mock-region"]
        )

    async def create_instance(self, request: InstanceRequest) -> Instance:
        if self.fail_api:
            raise ProviderAPIError("mock: injected API error on create")
        if self.fail_create:
            raise InstanceCreationError("mock: injected creation failure")
        self._counter += 1
        instance_id = f"mock-{self._counter}"
        self._pods[instance_id] = _Pod(
            instance_id, request.name, request.gpu_type, request.ports, request.network_volume_id
        )
        return self._instance(self._pods[instance_id])

    async def get_instance(self, provider_instance_id: str) -> Instance | None:
        if self.fail_api:
            raise ProviderAPIError("mock: injected API error on get")
        pod = self._pods.get(provider_instance_id)
        if pod is None:
            return None
        pod.observations += 1
        return self._instance(pod)

    async def destroy_instance(self, provider_instance_id: str) -> None:
        # Idempotent: no error if it is already gone.
        self._pods.pop(provider_instance_id, None)

    async def list_instances(self) -> list[Instance]:
        if self.fail_api:
            raise ProviderAPIError("mock: injected API error on list")
        prefix = self.instance_prefix()
        return [self._instance(p) for p in self._pods.values() if p.name.startswith(prefix)]

    async def find_instance_by_deployment_id(self, deployment_id: str) -> Instance | None:
        target = self.instance_name(deployment_id)
        for pod in self._pods.values():
            if pod.name == target:
                return self._instance(pod)
        return None

    async def resolve_endpoint_url(self, instance: Instance, port: int) -> str | None:
        pod = self._pods.get(instance.provider_instance_id)
        if pod is None or pod.state(self.steps_to_running) != _RUNNING:
            return None
        return f"https://{instance.provider_instance_id}-{port}.mock.local"

    async def get_logs(self, provider_instance_id: str, tail: int = 100) -> list[str]:
        pod = self._pods.get(provider_instance_id)
        return pod.logs[-tail:] if pod else []

    # --- volumes (in-memory) --------------------------------------------------------

    async def ensure_cache_volume(self, name: str, size_gb: int, region: str | None) -> str:
        for volume in self._volumes.values():
            if volume.name == name:
                return volume.id  # find-or-create: idempotent by name
        self._volume_counter += 1
        volume_id = f"vol-{self._volume_counter}"
        self._volumes[volume_id] = VolumeInfo(
            id=volume_id, name=name, size_gb=size_gb, data_center_id=region
        )
        return volume_id

    async def list_volumes(self) -> list[VolumeInfo]:
        return list(self._volumes.values())

    async def delete_volume(self, volume_id: str) -> None:
        self._volumes.pop(volume_id, None)  # idempotent

    def _instance(self, pod: _Pod) -> Instance:
        return Instance(
            provider_instance_id=pod.id,
            provider=self.name,
            gpu_type=pod.gpu_type,
            state=pod.state(self.steps_to_running),
            public_url=None,
            ports=pod.ports,
        )
