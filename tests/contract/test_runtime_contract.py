"""Runtime contract suite (spec §9), the mirror of the provider one.

This suite IS the Runtime spec: a new runtime passes it or it is not done. It exists because the ABC
had exactly one implementer for its whole life, which means nothing had ever tested whether the seam
was an abstraction or just vLLM's shape with a base class bolted on (issue #23).

Everything here is offline. ``build_instance_request`` and ``download_progress`` are pure by the
ABC's own contract, and the two HTTP methods are exercised through an injected transport.
"""

from __future__ import annotations

import httpx
import pytest

from gpu_orchestrator.models import GPUType, ModelSpec, RuntimeProfile, ValidationMetadata
from gpu_orchestrator.runtimes import RUNTIMES
from gpu_orchestrator.runtimes.llamacpp import LlamaCppRuntime
from gpu_orchestrator.runtimes.vllm import VLLMRuntime

_GPU = GPUType(
    id="A100-80GB",
    name="NVIDIA A100 80GB PCIe",
    memory_gb=80,
    hourly_usd=1.89,
    provider_sku="NVIDIA A100 80GB PCIe",
    host_ram_gb=117,
    vcpu_count=8,
)

_SPEC = ModelSpec(
    id="qwen3-32b",
    hf_repo="Qwen/Qwen3-32B",
    family="qwen3",
    parameter_count="32B",
    min_gpu_memory_gb=80,
    context_window=32768,
    license="apache-2.0",
)


def _profile(runtime: str, **overrides) -> RuntimeProfile:
    base = dict(
        model_id="qwen3-32b",
        runtime=runtime,
        image=f"{runtime}/image:pinned",
        recommended_gpu="A100-80GB",
        min_disk_gb=60,
        validation=ValidationMetadata(
            validated_at="2026-08-09",
            validated_provider="mock",
            validated_gpu="A100-80GB",
            validated_image=f"{runtime}/image:pinned",
            startup_timeout_seconds=1200,
        ),
    )
    base.update(overrides)
    return RuntimeProfile(**base)


def _unreachable() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    return httpx.MockTransport(handler)


_BUILDERS = [
    pytest.param((lambda transport=None: VLLMRuntime(transport=transport), "vllm"), id="vllm"),
    pytest.param(
        (lambda transport=None: LlamaCppRuntime(transport=transport), "llamacpp"), id="llamacpp"
    ),
]


@pytest.fixture(params=_BUILDERS)
def pair(request):
    build, name = request.param
    return build, name


# --- registration ----------------------------------------------------------------------


def test_runtime_is_registered_under_its_own_name(pair):
    _, name = pair
    assert name in RUNTIMES
    assert RUNTIMES[name].name == name


def test_serving_port_is_declared_and_plausible(pair):
    build, _ = pair
    runtime = build()
    assert isinstance(runtime.serving_port, int)
    assert 0 < runtime.serving_port < 65536


# --- build_instance_request: pure, and wires the pod up to the server -------------------


def test_build_instance_request_is_pure(pair):
    """Called twice with the same inputs it returns the same request, and touches no network.

    This is the ABC's central claim (``build_instance_request`` is PURE), and it is what lets the
    reconciler stay exhaustively testable.
    """
    build, name = pair
    runtime = build()
    once = runtime.build_instance_request(_SPEC, _profile(name), _GPU, name="gpu-orch-t-dep-1")
    twice = runtime.build_instance_request(_SPEC, _profile(name), _GPU, name="gpu-orch-t-dep-1")
    assert once == twice


def test_request_carries_the_name_it_was_given(pair):
    """The ``gpu-orch-{namespace}-{deployment_id}`` tag is the hook every idempotency, adoption, and
    orphan-sweep guarantee hangs on, so a runtime may never rewrite it."""
    build, name = pair
    req = build().build_instance_request(_SPEC, _profile(name), _GPU, name="gpu-orch-t-dep-1")
    assert req.name == "gpu-orch-t-dep-1"


def test_request_uses_the_provider_sku_not_the_catalog_id(pair):
    build, name = pair
    req = build().build_instance_request(_SPEC, _profile(name), _GPU, name="n")
    assert req.gpu_type == _GPU.provider_sku


def test_request_exposes_the_serving_port(pair):
    """The provider turns (instance, serving_port) into a URL, so the port must be opened."""
    build, name = pair
    runtime = build()
    req = runtime.build_instance_request(_SPEC, _profile(name), _GPU, name="n")
    assert runtime.serving_port in req.ports


def test_request_takes_image_and_disk_from_the_profile(pair):
    build, name = pair
    profile = _profile(name)
    req = build().build_instance_request(_SPEC, profile, _GPU, name="n")
    assert req.image == profile.image
    assert req.disk_gb == profile.min_disk_gb


def test_multi_gpu_profile_requests_multiple_gpus(pair):
    build, name = pair
    req = build().build_instance_request(_SPEC, _profile(name, tensor_parallel=4), _GPU, name="n")
    assert req.gpu_count == 4


def test_profile_launch_args_reach_the_command(pair):
    build, name = pair
    profile = _profile(name, launch_args={"--contract-canary": "1"})
    req = build().build_instance_request(_SPEC, profile, _GPU, name="n")
    assert "--contract-canary" in req.command


def test_profile_env_is_carried_but_no_secrets_are_invented(pair):
    """Runtimes set only ``profile.env``; the orchestrator adds credentials before create, so that
    credential handling lives in exactly one place."""
    build, name = pair
    req = build().build_instance_request(_SPEC, _profile(name, env={"FOO": "bar"}), _GPU, name="n")
    assert req.env == {"FOO": "bar"}


# --- download_progress: pure, bounded ---------------------------------------------------


def test_download_progress_is_none_for_empty_logs(pair):
    build, _ = pair
    assert build().download_progress([]) is None


def test_download_progress_is_none_or_a_fraction(pair):
    build, _ = pair
    got = build().download_progress(["something 45% something", "no percent here"])
    assert got is None or 0.0 <= got <= 1.0


def test_download_progress_never_exceeds_one(pair):
    """Logs lie: a "150%" in a progress bar must not produce a fraction above 1."""
    build, _ = pair
    got = build().download_progress(["downloading 150%"])
    assert got is None or got <= 1.0


# --- health: never raise, always answer -------------------------------------------------


async def test_health_check_reports_failure_rather_than_raising(pair):
    """The reconciler treats an unhealthy runtime as a state, not an exception. A runtime that
    raises here would crash a reconcile tick instead of producing a WAIT/DEGRADED decision."""
    build, _ = pair
    result = await build(_unreachable()).health_check("http://nowhere:1")
    assert result.ok is False
    assert result.detail


async def test_model_ready_reports_failure_rather_than_raising(pair):
    build, _ = pair
    result = await build(_unreachable()).model_ready("http://nowhere:1", "qwen3-32b")
    assert result.ok is False
