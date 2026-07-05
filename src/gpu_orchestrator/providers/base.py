"""Provider seam: provision compute, know nothing about LLMs (spec §8).

Async end-to-end (resolved 2026-07-03, CLAUDE.md): every method is a coroutine. A provider never
imports from ``runtimes/`` or ``catalog.py``. Endpoint-URL shape is provider knowledge and lives
behind ``resolve_endpoint_url``, which returns ``None`` until the endpoint is actually routable so
the reconciler can treat that as WAIT_FOR_PROVIDER.

The instance-naming convention (spec §7.5) is centralized here on the base: every provider builds
and filters ``gpu-orch-{namespace}-{deployment_id}`` names the same way, so ``find_instance_by_
deployment_id`` and ``list_instances`` stay scoped to one install's pods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .. import naming
from ..errors import NotSupportedError
from ..models import Instance, InstanceRequest, ProviderCapabilities, VolumeInfo


class Provider(ABC):
    name: ClassVar[str]

    def __init__(self, *, namespace: str) -> None:
        self.namespace = namespace

    # --- naming (shared, not overridden) --------------------------------------------

    def instance_name(self, deployment_id: str) -> str:
        return naming.instance_name(self.namespace, deployment_id)

    def instance_prefix(self) -> str:
        return naming.instance_prefix(self.namespace)

    # --- capabilities ---------------------------------------------------------------

    @abstractmethod
    async def capabilities(self) -> ProviderCapabilities: ...

    # --- lifecycle ------------------------------------------------------------------

    @abstractmethod
    async def create_instance(self, request: InstanceRequest) -> Instance: ...

    @abstractmethod
    async def get_instance(self, provider_instance_id: str) -> Instance | None:
        """Return the instance, or ``None`` if it is gone. ``None`` is the canonical gone signal."""

    @abstractmethod
    async def destroy_instance(self, provider_instance_id: str) -> None:
        """Idempotent: destroying a nonexistent instance is a no-op, not an error."""

    @abstractmethod
    async def list_instances(self) -> list[Instance]:
        """All instances owned by this install (filtered to ``instance_prefix()``)."""

    @abstractmethod
    async def find_instance_by_deployment_id(self, deployment_id: str) -> Instance | None:
        """Look up an instance by its ``gpu-orch-{namespace}-{deployment_id}`` name (spec §7.5)."""

    # --- endpoint + logs ------------------------------------------------------------

    @abstractmethod
    async def resolve_endpoint_url(self, instance: Instance, port: int) -> str | None:
        """A routable public URL for ``port``, or ``None`` until routable (spec §8)."""

    @abstractmethod
    async def get_logs(self, provider_instance_id: str, tail: int = 100) -> list[str]: ...

    # --- persistent volumes (optional capability; default unsupported) --------------

    async def ensure_cache_volume(self, name: str, size_gb: int, region: str | None) -> str:
        """Find-or-create a persistent network volume by name, returning its id (spec §14)."""
        raise NotSupportedError(f"{type(self).__name__} does not support cache volumes")

    async def list_volumes(self) -> list[VolumeInfo]:
        return []

    async def delete_volume(self, volume_id: str) -> None:
        raise NotSupportedError(f"{type(self).__name__} does not support cache volumes")


# Populated at import time from the concrete providers below. A dict, not a plugin loader (E1).
from .mock import MockProvider  # noqa: E402
from .runpod import RunPodProvider  # noqa: E402

PROVIDERS: dict[str, type[Provider]] = {
    "runpod": RunPodProvider,
    "mock": MockProvider,
}
