"""Step-4 vLLM runtime tests: request composition (pure) and health checks (via MockTransport)."""

from __future__ import annotations

import httpx

from gpu_orchestrator.models import GPUType
from gpu_orchestrator.runtimes import VLLMRuntime
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC

_GPU = GPUType(
    id="RTX-A4000",
    name="NVIDIA RTX A4000",
    memory_gb=16,
    hourly_usd=0.17,
    provider_sku="NVIDIA RTX A4000",
)


def _pairs(command: list[str]) -> dict[str, str]:
    # command = [python3, -m, module, --flag, value, --flag, value, ...]
    flags = command[3:]
    return {flags[i]: flags[i + 1] for i in range(0, len(flags), 2)}


def test_build_instance_request_composes_command():
    rt = VLLMRuntime()
    req = rt.build_instance_request(
        QWEN3_06B_SPEC, QWEN3_06B_PROFILE, _GPU, name="gpu-orch-test-dep-1"
    )
    assert req.name == "gpu-orch-test-dep-1"
    assert req.gpu_type == "NVIDIA RTX A4000"
    assert req.image == QWEN3_06B_PROFILE.image
    assert req.ports == [8000]
    args = _pairs(req.command)
    assert args["--model"] == "Qwen/Qwen3-0.6B"
    assert args["--tensor-parallel-size"] == "1"
    assert args["--max-model-len"] == str(QWEN3_06B_SPEC.context_window)


def test_profile_launch_args_override_defaults():
    rt = VLLMRuntime()
    profile = QWEN3_06B_PROFILE.model_copy(update={"launch_args": {"--max-model-len": "8192"}})
    req = rt.build_instance_request(QWEN3_06B_SPEC, profile, _GPU, name="n")
    assert _pairs(req.command)["--max-model-len"] == "8192"


def test_download_progress_parses_and_returns_none():
    rt = VLLMRuntime()
    assert rt.download_progress(["Downloading model.safetensors: 45%|### | 1.2G/2.6G"]) == 0.45
    assert rt.download_progress(["nothing to see here"]) is None
    # last match wins across lines
    assert rt.download_progress(["10%", "90%"]) == 0.90


async def test_health_check_ok_and_fail():
    ok_rt = VLLMRuntime(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    res = await ok_rt.health_check("http://pod:8000")
    assert res.ok and res.latency_ms is not None

    down_rt = VLLMRuntime(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    assert (await down_rt.health_check("http://pod:8000")).ok is False


async def test_model_ready_checks_served_id():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-0.6B"}]})

    rt = VLLMRuntime(transport=httpx.MockTransport(handler))
    assert (await rt.model_ready("http://pod:8000", "Qwen/Qwen3-0.6B")).ok is True
    assert (await rt.model_ready("http://pod:8000", "other/model")).ok is False


async def test_model_ready_unreachable_is_not_ok():
    def boom(request):
        raise httpx.ConnectError("refused")

    rt = VLLMRuntime(transport=httpx.MockTransport(boom))
    res = await rt.model_ready("http://pod:8000", "x")
    assert res.ok is False and "unreachable" in res.detail
