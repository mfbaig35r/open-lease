"""Demand-driven autoscaling (Tier B2): the pure decision (replicas from served rate, clamped to
min/max) and the replica clone builder."""

from __future__ import annotations

from gpu_orchestrator.core import autoscale
from gpu_orchestrator.models import AutoscalePolicy, DeploymentState
from tests.fixtures.deployments import make_deployment


def _policy(minimum: int = 1, maximum: int = 5, target: float = 60.0) -> AutoscalePolicy:
    return AutoscalePolicy(
        model_id="qwen3-0.6b",
        min_replicas=minimum,
        max_replicas=maximum,
        target_rpm_per_replica=target,
    )


def test_desired_replicas_tracks_rate_within_bounds():
    policy = _policy(minimum=1, maximum=5, target=60.0)
    assert autoscale.desired_replicas(0, policy) == 1  # idle -> the min floor
    assert autoscale.desired_replicas(60, policy) == 1  # exactly one replica's worth
    assert autoscale.desired_replicas(61, policy) == 2  # ceil past the target
    assert autoscale.desired_replicas(300, policy) == 5  # capped at max
    assert autoscale.desired_replicas(9999, policy) == 5  # still capped


def test_desired_replicas_respects_the_min_floor():
    policy = _policy(minimum=2, maximum=5, target=60.0)
    assert autoscale.desired_replicas(0, policy) == 2  # keep the warm floor even when idle
    assert autoscale.desired_replicas(30, policy) == 2  # ceil(0.5)=1, floored back up to min


def test_replica_from_clones_the_template_config():
    template = make_deployment(DeploymentState.READY)
    template.max_concurrency = 8
    template.hf_repo = "Qwen/Qwen3-0.6B"
    clone = autoscale.replica_from(template, "dep-new")
    assert clone.id == "dep-new"
    assert clone.model_id == template.model_id and clone.max_concurrency == 8
    assert clone.hf_repo == "Qwen/Qwen3-0.6B"
    assert clone.desired_state is DeploymentState.READY
    assert clone.observed_state is DeploymentState.REQUESTED
    assert clone.instance is None  # a fresh pod, not sharing the template's
