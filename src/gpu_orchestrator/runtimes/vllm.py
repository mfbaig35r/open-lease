"""vLLM runtime: serve an OpenAI-compatible endpoint from the vllm-openai image (spec §9).

Launch args come from the profile (with sensible defaults from the model spec); the composed command
runs the vLLM OpenAI server. Health is two checks: HTTP alive (/health) and model loaded
(/v1/models).

Secrets are NOT injected here. The runtime sets only ``profile.env``; the orchestrator adds the HF
token to the request env before create (keeps credential handling in one place, off the runtime).
"""

from __future__ import annotations

import re
import time

import httpx

from ..models import CheckResult, GPUType, InstanceRequest, ModelSpec, RuntimeProfile
from .base import Runtime

_VLLM_PORT = 8000
_ENTRYPOINT = ["python3", "-m", "vllm.entrypoints.openai.api_server"]

# HF hub / vLLM download lines look like "... 45%|#### | 1.2G/2.6G". Grab the last percentage.
_PROGRESS_RE = re.compile(r"(\d{1,3})%")


class VLLMRuntime(Runtime):
    name = "vllm"
    serving_port = _VLLM_PORT

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
            "--model": spec.hf_repo,
            "--tensor-parallel-size": str(profile.tensor_parallel),
            "--gpu-memory-utilization": str(profile.gpu_memory_utilization),
        }
        if spec.context_window:  # 0 for an ad-hoc deploy with no --context: let vLLM auto-detect
            args["--max-model-len"] = str(spec.context_window)
        args.update(profile.launch_args)  # profile overrides defaults

        command = list(_ENTRYPOINT)
        for flag, value in args.items():
            command += [flag, value]

        return InstanceRequest(
            name=name,
            gpu_type=gpu.provider_sku,
            # One knob drives both: the pod gets tensor_parallel GPUs and vLLM shards across exactly
            # that many. They must match.
            gpu_count=profile.tensor_parallel,
            image=profile.image,
            env=dict(profile.env),
            disk_gb=profile.min_disk_gb,
            ports=[_VLLM_PORT],
            command=command,
        )

    async def health_check(self, endpoint_url: str) -> CheckResult:
        start = time.perf_counter()
        try:
            async with self._client() as client:
                resp = await client.get(f"{endpoint_url}/health")
            latency = (time.perf_counter() - start) * 1000
            ok = resp.status_code == 200
            return CheckResult(ok=ok, latency_ms=latency, detail=f"HTTP {resp.status_code}")
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
            # vLLM serves the model under its HF repo id (e.g. "Qwen/Qwen3-0.6B"), which differs
            # from our catalog id ("qwen3-0.6b"). We launch exactly one model per pod, so any served
            # model means the server is up and ready; matching the catalog id would never pass.
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
                pct = min(100, int(matches[-1]))
                last = pct / 100.0
        return last
