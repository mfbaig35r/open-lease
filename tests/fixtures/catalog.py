"""Canonical catalog fixtures: valid ModelSpec + RuntimeProfile pairs.

These double as documentation of a well-formed catalog entry (spec §14). Every RuntimeProfile
here carries ValidationMetadata, because a profile without it does not ship.
"""

from __future__ import annotations

from gpu_orchestrator.models import ModelSpec, RuntimeProfile, ValidationMetadata

# A large model: the headline "zero to chat" target (slow cold start, big GPU).
QWEN3_32B_SPEC = ModelSpec(
    id="qwen3-32b",
    hf_repo="Qwen/Qwen3-32B",
    family="qwen3",
    parameter_count="32B",
    min_gpu_memory_gb=80,
    context_window=32768,
    license="apache-2.0",
    chat=True,
    supports_tools=True,
    supports_reasoning=True,
)

QWEN3_32B_PROFILE = RuntimeProfile(
    model_id="qwen3-32b",
    runtime="vllm",
    image="vllm/vllm-openai:v0.9.1",
    launch_args={"--max-model-len": "32768"},
    tensor_parallel=1,
    gpu_memory_utilization=0.90,
    recommended_gpu="A100-80GB",
    min_disk_gb=120,
    validation=ValidationMetadata(
        validated_at="2026-07-03",
        validated_provider="runpod",
        validated_gpu="A100-80GB",
        validated_image="vllm/vllm-openai:v0.9.1",
        startup_timeout_seconds=2400,
        notes="First launch downloads ~65GB; expect slow cold start.",
    ),
)

# A small model: the cheapest thing to run in the integration gauntlet.
QWEN3_06B_SPEC = ModelSpec(
    id="qwen3-0.6b",
    hf_repo="Qwen/Qwen3-0.6B",
    family="qwen3",
    parameter_count="0.6B",
    min_gpu_memory_gb=8,
    context_window=32768,
    license="apache-2.0",
    chat=True,
)

QWEN3_06B_PROFILE = RuntimeProfile(
    model_id="qwen3-0.6b",
    runtime="vllm",
    image="vllm/vllm-openai:v0.9.1",
    tensor_parallel=1,
    gpu_memory_utilization=0.90,
    recommended_gpu="RTX-A4000",
    min_disk_gb=20,
    validation=ValidationMetadata(
        validated_at="2026-07-03",
        validated_provider="runpod",
        validated_gpu="RTX-A4000",
        validated_image="vllm/vllm-openai:v0.9.1",
        startup_timeout_seconds=600,
        notes="Smallest catalog model; used for the integration gauntlet.",
    ),
)

SPECS = {s.id: s for s in (QWEN3_32B_SPEC, QWEN3_06B_SPEC)}
PROFILES = {p.model_id: p for p in (QWEN3_32B_PROFILE, QWEN3_06B_PROFILE)}
