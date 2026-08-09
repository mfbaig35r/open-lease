"""UX #1: the `gpu status` progress hint. A real percent when the runtime exposed download progress,
else elapsed-in-stage against the model's startup budget, so a cold start never looks stuck."""

from __future__ import annotations

from gpu_orchestrator.cli import render
from gpu_orchestrator.models import (
    DeploymentState,
    GPUType,
    ProviderCapabilities,
    ProviderInfo,
)
from tests.fixtures.deployments import make_deployment


def test_hint_shows_percent_when_progress_known():
    dep = make_deployment(DeploymentState.STARTING)
    dep.download_progress = 0.37
    assert render.progress_hint(dep) == "37%"


def test_hint_falls_back_to_elapsed_over_budget():
    # No download progress (e.g. RunPod): show elapsed/budget so the user sees it is not stuck.
    # make_deployment stamps the STARTING transition well in the past, so elapsed is large.
    dep = make_deployment(DeploymentState.STARTING)
    dep.download_progress = None
    hint = render.progress_hint(dep)
    assert hint is not None and "/" in hint  # elapsed / startup budget


def test_hint_none_once_ready():
    assert render.progress_hint(make_deployment(DeploymentState.READY)) is None


def test_hint_none_when_stopped():
    assert render.progress_hint(make_deployment(DeploymentState.STOPPED)) is None


def test_gpus_table_distinguishes_unreported_host_resources_from_zero(capsys):
    """The whole point of issue #26 is that "we did not ask" must not read as "none allocated"."""
    info = ProviderInfo(
        name="p",
        capabilities=ProviderCapabilities(
            gpu_types=[
                GPUType(
                    id="known",
                    name="Known",
                    memory_gb=24,
                    hourly_usd=0.22,
                    provider_sku="known",
                    host_ram_gb=125,
                    vcpu_count=16,
                ),
                GPUType(id="quiet", name="Quiet", memory_gb=24, hourly_usd=0.16, provider_sku="q"),
            ]
        ),
    )
    render.gpus_table([info])
    out = capsys.readouterr().out
    assert "125 GB" in out and "16" in out
    assert "not reported" in out
    assert " 0 GB" not in out, "an unknown host must never render as zero"


def test_gpu_type_host_fields_default_to_none_not_zero():
    gpu = GPUType(id="g", name="G", memory_gb=24, hourly_usd=1.0, provider_sku="g")
    assert gpu.host_ram_gb is None
    assert gpu.vcpu_count is None
