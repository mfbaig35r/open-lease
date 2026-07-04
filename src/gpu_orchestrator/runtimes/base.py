"""Runtime seam: serve models, know nothing about providers (spec §9).

A runtime turns a model + profile + GPU into an ``InstanceRequest`` (compute shape only) and answers
health questions about a running endpoint. It never talks to a provider and never learns a provider
instance id; it only knows URLs and ports.

Method sync/async split follows where the I/O is (async decision, CLAUDE.md): ``build_instance_
request`` and ``download_progress`` are PURE (no I/O) and stay sync and exhaustively testable;
``health_check`` and ``model_ready`` do HTTP and are async.

Deviations from the spec's §9 signatures, driven by concrete need:
- ``build_instance_request`` also takes the ``ModelSpec`` (for ``hf_repo`` -> vLLM ``--model`` and
  ``context_window`` -> ``--max-model-len``) and the instance ``name`` (naming is the orchestrator's
  job, not the runtime's; the name is required on ``InstanceRequest`` per spec §7.5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..models import CheckResult, GPUType, InstanceRequest, ModelSpec, RuntimeProfile


class Runtime(ABC):
    name: ClassVar[str]
    # The single port the model server listens on. The provider (not the runtime) turns
    # (instance, serving_port) into a public URL, so this is all a runtime declares about routing
    # (spec §8). vLLM serves on 8000.
    serving_port: ClassVar[int]

    @abstractmethod
    def build_instance_request(
        self,
        spec: ModelSpec,
        profile: RuntimeProfile,
        gpu: GPUType,
        *,
        name: str,
    ) -> InstanceRequest:
        """Compose the compute request (pure; no I/O)."""

    @abstractmethod
    async def health_check(self, endpoint_url: str) -> CheckResult:
        """HTTP-level liveness (e.g. GET /health)."""

    @abstractmethod
    async def model_ready(self, endpoint_url: str, model_id: str) -> CheckResult:
        """Whether the model is loaded and serving (e.g. GET /v1/models)."""

    @abstractmethod
    def download_progress(self, logs: list[str]) -> float | None:
        """Parse download progress in 0..1 from logs, or None when unparseable (pure)."""


# Populated from the concrete runtimes below. A dict, not a plugin loader (E1).
from .vllm import VLLMRuntime  # noqa: E402

RUNTIMES: dict[str, type[Runtime]] = {"vllm": VLLMRuntime}
