"""llama.cpp runtime: serve an OpenAI-compatible endpoint from llama-server (spec §9).

Runtime #2. It earns its place twice over: it serves GGUF (so small-GPU and CPU-assisted
deployments become possible at all), and it is the only engine that can run the one tiered-execution
configuration that survived measurement in docs/adr-adaptive-execution-planning.md.

``cpu_moe_offload`` maps to ``--n-cpu-moe``, which keeps a MoE model's routed experts in host RAM
and runs those matmuls on the CPU, so only activations cross PCIe. Read the caveat on
``RuntimeProfile.cpu_moe_offload`` before reaching for it: measured throughput is 15% of resident
at batch 1 and 8.5% at concurrency 16. It buys a cheaper machine, not a faster one.

Secrets are NOT injected here, same as vLLM: the runtime sets only ``profile.env`` and the
orchestrator adds credentials before create.
"""

from __future__ import annotations

import re
import time

import httpx

from ..models import CheckResult, GPUType, InstanceRequest, ModelSpec, RuntimeProfile
from .base import Runtime

_LLAMA_PORT = 8080  # llama-server's default

# The server image ships the binary at /app/llama-server. Not yet proven against a live pod: no
# catalog entry can ship until ValidationMetadata says someone launched it (spec §14), which is
# exactly the gate that keeps an unvalidated path from being presented as a supported one.
_ENTRYPOINT = ["/app/llama-server"]

# llama.cpp download lines carry a percentage much like the HF hub's. Same shape, same parse.
_PROGRESS_RE = re.compile(r"(\d{1,3})%")

# -ngl: how many layers to put on the GPU. 99 means "all of them", the normal case; llama.cpp
# silently caps it at the model's real layer count.
_ALL_LAYERS_ON_GPU = "99"


class LlamaCppRuntime(Runtime):
    name = "llamacpp"
    serving_port = _LLAMA_PORT

    def __init__(self, *, transport: httpx.BaseTransport | None = None) -> None:
        # transport is an injection seam for tests (httpx.MockTransport); None = real network.
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=10)

    def build_instance_request(
        self,
        spec: ModelSpec,
        profile: RuntimeProfile,
        gpu: GPUType,
        *,
        name: str,
    ) -> InstanceRequest:
        args: dict[str, str] = {
            # -hf pulls the GGUF straight from the hub, so no separate fetch step is needed.
            "-hf": spec.hf_repo,
            "--host": "0.0.0.0",
            "--port": str(_LLAMA_PORT),
            "-ngl": _ALL_LAYERS_ON_GPU,
        }
        if spec.context_window:  # 0 on an ad-hoc deploy: let llama.cpp read it from the GGUF
            args["-c"] = str(spec.context_window)
        if profile.cpu_moe_offload:
            # Routed experts for the last N layers live in host RAM and are computed on the CPU.
            args["--n-cpu-moe"] = str(profile.cpu_moe_offload)
        args.update(profile.launch_args)  # profile overrides defaults

        command = list(_ENTRYPOINT)
        for flag, value in args.items():
            command += [flag, value]

        return InstanceRequest(
            name=name,
            gpu_type=gpu.provider_sku,
            # Unlike vLLM there is no tensor-parallel flag to match: llama.cpp splits layers across
            # whatever GPUs it finds. tensor_parallel therefore sizes the pod and nothing else.
            gpu_count=profile.tensor_parallel,
            image=profile.image,
            env=dict(profile.env),
            disk_gb=profile.min_disk_gb,
            ports=[_LLAMA_PORT],
            command=command,
        )

    async def health_check(self, endpoint_url: str) -> CheckResult:
        start = time.perf_counter()
        try:
            async with self._client() as client:
                resp = await client.get(f"{endpoint_url}/health")
            latency = (time.perf_counter() - start) * 1000
            # llama-server answers 503 while the model is still loading, which is "not ready yet"
            # rather than "broken". The reconciler already treats not-ok as WAIT, so reporting the
            # status verbatim keeps that distinction visible in the detail line.
            return CheckResult(
                ok=resp.status_code == 200, latency_ms=latency, detail=f"HTTP {resp.status_code}"
            )
        except httpx.HTTPError as exc:
            return CheckResult(ok=False, detail=f"unreachable: {exc}")

    async def model_ready(self, endpoint_url: str, model_id: str) -> CheckResult:
        start = time.perf_counter()
        try:
            async with self._client() as client:
                resp = await client.get(f"{endpoint_url}/v1/models")
            latency = (time.perf_counter() - start) * 1000
            if resp.status_code != 200:
                return CheckResult(ok=False, latency_ms=latency, detail=f"HTTP {resp.status_code}")
            served = {m.get("id") for m in resp.json().get("data", []) if m.get("id")}
            # Same reasoning as vLLM: llama-server reports the GGUF's own name, not our catalog id,
            # and we launch one model per pod, so any served model means ready.
            ok = model_id in served or bool(served)
            detail = f"serving {sorted(served)}" if ok else "no model loaded yet"
            return CheckResult(ok=ok, latency_ms=latency, detail=detail)
        except httpx.HTTPError as exc:
            return CheckResult(ok=False, detail=f"unreachable: {exc}")

    def download_progress(self, logs: list[str]) -> float | None:
        last: float | None = None
        for line in logs:
            matches = _PROGRESS_RE.findall(line)
            if matches:
                last = min(100, int(matches[-1])) / 100.0
        return last
