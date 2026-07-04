"""UX #1: the `gpu status` progress hint. A real percent when the runtime exposed download progress,
else elapsed-in-stage against the model's startup budget, so a cold start never looks stuck."""

from __future__ import annotations

from gpu_orchestrator.cli import render
from gpu_orchestrator.models import DeploymentState
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
