"""RunPod provider: extracted and adapted from runpod-ephemeral (spec §8.1).

The API mechanics (auth, create body, proxy-URL shape, idempotent delete) come from live-validated
runpod-ephemeral code. Adapted for long-running serving: namespaced names, list/find, capabilities,
``get_instance`` returning ``None`` when gone, and endpoint gating on the running state.

Provider-native pod states are stored VERBATIM in ``Instance.state`` (spec §8.1). This module never
translates them to ``DeploymentState``; that happens in exactly one core function
(``map_to_observed_state``, arriving at build step 5).

Known limitation: RunPod's REST API has no pod-log retrieval (a runpod-ephemeral finding), so
``get_logs`` returns ``[]``. vLLM download-progress parsing (§9) therefore has nothing to read on
RunPod, and the reconciler falls back to the download-stage timeout budget (§7.3, §9 allow this).
"""

from __future__ import annotations

import os

import httpx

from ..errors import InstanceCreationError, NotSupportedError, ProviderAPIError
from ..logging import get_logger
from ..models import CloudType, GPUType, Instance, InstanceRequest, ProviderCapabilities
from .base import Provider

_BASE = "https://rest.runpod.io/v1"
_RUNNING = "RUNNING"
_log = get_logger("providers.runpod")

# Curated Phase 1 GPU menu. Rates are approximate and refined by catalog validation metadata (§14).
_GPU_TYPES = [
    GPUType(
        id="RTX-A4000",
        name="NVIDIA RTX A4000",
        memory_gb=16,
        hourly_usd=0.17,
        provider_sku="NVIDIA RTX A4000",
    ),
    GPUType(
        id="A100-80GB",
        name="NVIDIA A100 80GB PCIe",
        memory_gb=80,
        hourly_usd=1.89,
        provider_sku="NVIDIA A100 80GB PCIe",
    ),
]


class RunPodProvider(Provider):
    name = "runpod"

    def __init__(
        self, *, namespace: str, api_key: str | None = None, base_url: str = _BASE
    ) -> None:
        super().__init__(namespace=namespace)
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        if not self.api_key:
            raise ProviderAPIError("RUNPOD_API_KEY not set")
        self.base_url = base_url

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30,
        )

    async def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            gpu_types=list(_GPU_TYPES),
            supports_volumes=False,  # deferred in Phase 1 (spec §8)
            supports_snapshots=False,
            regions=[],
        )

    async def create_instance(self, request: InstanceRequest) -> Instance:
        if request.cloud_type is CloudType.SPOT:
            raise NotSupportedError("Spot instances are deferred in Phase 1 (spec §8.1)")
        if request.volume is not None:
            raise NotSupportedError("Volumes are deferred in Phase 1 (spec §8, §14)")

        body: dict[str, object] = {
            "name": request.name,  # gpu-orch-{namespace}-{deployment_id} (spec §7.5)
            "imageName": request.image,
            # gpuTypeIds is an ordered preference list of literal RunPod ids (runpod-ephemeral).
            "gpuTypeIds": [request.gpu_type],
            "gpuCount": 1,
            "cloudType": "SECURE",
            "containerDiskInGb": request.disk_gb,
            "env": request.env or None,
            # dockerEntrypoint overrides the image ENTRYPOINT so the pod runs our command on start
            # (runpod-ephemeral). Empty list => use the image default.
            "dockerEntrypoint": request.command or None,
        }
        # RunPod REST v1 wants ports as a JSON array of "<port>/<proto>" strings, not a joined
        # string (verified against the live 400: "/pods/properties/ports/type: got string, want
        # array"). runpod-ephemeral's comma-joined form was for an older API.
        if request.ports:
            body["ports"] = [f"{p}/http" for p in request.ports]
        payload = {k: v for k, v in body.items() if v is not None}
        try:
            async with self._client() as client:
                resp = await client.post("/pods", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise InstanceCreationError(f"RunPod rejected create: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"RunPod create failed: {exc}") from exc

        return Instance(
            provider_instance_id=data["id"],
            provider=self.name,
            gpu_type=request.gpu_type,
            state=str(data.get("desiredStatus", "")),
            public_url=None,
            ports=request.ports,
        )

    async def get_instance(self, provider_instance_id: str) -> Instance | None:
        try:
            async with self._client() as client:
                resp = await client.get(f"/pods/{provider_instance_id}")
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"RunPod get failed: {exc}") from exc
        return self._to_instance(data)

    async def destroy_instance(self, provider_instance_id: str) -> None:
        try:
            async with self._client() as client:
                resp = await client.delete(f"/pods/{provider_instance_id}")
            if resp.status_code not in (200, 204, 404):  # 404 = already gone (idempotent)
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"RunPod destroy failed: {exc}") from exc

    async def list_instances(self) -> list[Instance]:
        try:
            async with self._client() as client:
                resp = await client.get("/pods")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"RunPod list failed: {exc}") from exc
        pods = data if isinstance(data, list) else data.get("pods", [])
        prefix = self.instance_prefix()
        return [self._to_instance(p) for p in pods if str(p.get("name", "")).startswith(prefix)]

    async def find_instance_by_deployment_id(self, deployment_id: str) -> Instance | None:
        # Instance carries no name, so match against the raw pod list by the exact tag (spec §7.5).
        target = self.instance_name(deployment_id)
        try:
            async with self._client() as client:
                resp = await client.get("/pods")
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as exc:
            raise ProviderAPIError(f"RunPod find failed: {exc}") from exc
        pods = data if isinstance(data, list) else data.get("pods", [])
        for pod in pods:
            if str(pod.get("name", "")) == target:
                return self._to_instance(pod)
        return None

    async def resolve_endpoint_url(self, instance: Instance, port: int) -> str | None:
        # RunPod proxies each pod at a deterministic URL, routable once the pod is RUNNING.
        if instance.state != _RUNNING:
            return None
        return f"https://{instance.provider_instance_id}-{port}.proxy.runpod.net"

    async def get_logs(self, provider_instance_id: str, tail: int = 100) -> list[str]:
        # RunPod REST exposes no pod logs; the pod is expected to ship its own (runpod-ephemeral).
        _log.debug("runpod get_logs is a no-op: no REST log endpoint", extra={"tail": tail})
        return []

    def _to_instance(self, data: dict) -> Instance:
        return Instance(
            provider_instance_id=str(data["id"]),
            provider=self.name,
            gpu_type=str(data.get("machine", {}).get("gpuTypeId", "") or data.get("gpuTypeId", "")),
            state=str(data.get("desiredStatus", "")),
            public_url=None,
            ports=[],
        )
