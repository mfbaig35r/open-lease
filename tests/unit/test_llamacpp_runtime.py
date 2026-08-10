"""llama.cpp runtime specifics, and the profile.runtime wiring that makes runtime #2 reachable.

The shared invariants live in tests/contract/test_runtime_contract.py. This file covers only what is
particular to llama.cpp: GGUF pull via -hf, and the --n-cpu-moe offload switch.
"""

from __future__ import annotations

from gpu_orchestrator.core.orchestrator import build_runtime
from gpu_orchestrator.models import GPUType, ModelSpec, RuntimeProfile, ValidationMetadata
from gpu_orchestrator.runtimes.llamacpp import LlamaCppRuntime
from gpu_orchestrator.runtimes.vllm import VLLMRuntime

_GPU = GPUType(id="A40-48GB", name="A40", memory_gb=48, hourly_usd=0.44, provider_sku="NVIDIA A40")

_SPEC = ModelSpec(
    id="qwen3-30b-a3b",
    hf_repo="unsloth/Qwen3-30B-A3B-GGUF",
    family="qwen3",
    parameter_count="30B",
    min_gpu_memory_gb=24,
    context_window=8192,
    license="apache-2.0",
)


def _profile(**overrides) -> RuntimeProfile:
    base = dict(
        model_id="qwen3-30b-a3b",
        runtime="llamacpp",
        image="ghcr.io/ggml-org/llama.cpp:server-cuda",
        recommended_gpu="A40-48GB",
        min_disk_gb=60,
        validation=ValidationMetadata(
            validated_at="",
            validated_provider="mock",
            validated_gpu="A40-48GB",
            validated_image="ghcr.io/ggml-org/llama.cpp:server-cuda",
            startup_timeout_seconds=1200,
        ),
    )
    base.update(overrides)
    return RuntimeProfile(**base)


def _pairs(command: list[str]) -> dict[str, str]:
    return {command[i]: command[i + 1] for i in range(1, len(command) - 1, 2)}


def test_pulls_the_gguf_from_the_hub_and_binds_all_interfaces():
    req = LlamaCppRuntime().build_instance_request(_SPEC, _profile(), _GPU, name="n")
    args = _pairs(req.command)
    assert args["-hf"] == "unsloth/Qwen3-30B-A3B-GGUF"
    assert args["--host"] == "0.0.0.0"  # the provider reaches it from outside the container
    assert args["-c"] == "8192"


def test_zero_context_window_lets_llamacpp_read_it_from_the_gguf():
    req = LlamaCppRuntime().build_instance_request(
        _SPEC.model_copy(update={"context_window": 0}), _profile(), _GPU, name="n"
    )
    assert "-c" not in req.command


def test_cpu_moe_offload_emits_n_cpu_moe():
    req = LlamaCppRuntime().build_instance_request(
        _SPEC, _profile(cpu_moe_offload=48), _GPU, name="n"
    )
    assert _pairs(req.command)["--n-cpu-moe"] == "48"


def test_no_offload_by_default_so_the_normal_case_stays_fully_resident():
    """Offload must be opt-in. It costs 3.2x to 22x more per token, so defaulting it on would make
    every llama.cpp deploy quietly slow and expensive (docs/adr-adaptive-execution-planning.md)."""
    assert _profile().cpu_moe_offload == 0
    req = LlamaCppRuntime().build_instance_request(_SPEC, _profile(), _GPU, name="n")
    assert "--n-cpu-moe" not in req.command


def test_profile_launch_args_can_override_a_default():
    req = LlamaCppRuntime().build_instance_request(
        _SPEC, _profile(launch_args={"-ngl": "20"}), _GPU, name="n"
    )
    assert _pairs(req.command)["-ngl"] == "20"


# --- the wiring that was inert while vLLM was the only runtime --------------------------


def test_build_runtime_resolves_both_engines_by_name():
    assert isinstance(build_runtime("llamacpp"), LlamaCppRuntime)
    assert isinstance(build_runtime("vllm"), VLLMRuntime)
