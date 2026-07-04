"""Canonical Deployment fixtures: one in every DeploymentState.

The reconciler's ``next_step`` is tested against the full desired/observed matrix, so having a
ready-made deployment in each state keeps those tests declarative. ``BY_STATE`` is the map the
unit suite iterates (spec §18).
"""

from __future__ import annotations

from datetime import UTC, datetime

from gpu_orchestrator.models import (
    Deployment,
    DeploymentState,
    FailureInfo,
    Instance,
    StateTransition,
)

from .catalog import QWEN3_06B_PROFILE

_T0 = datetime(2026, 7, 3, 12, 0, 0, tzinfo=UTC)

# A running instance, used by any state at/after BOOTING.
RUNNING_INSTANCE = Instance(
    provider_instance_id="pod-xyz123",
    provider="runpod",
    gpu_type="RTX-A4000",
    state="RUNNING",
    public_url="https://pod-xyz123-8000.proxy.runpod.net",
    ports=[8000],
)


def make_deployment(
    state: DeploymentState,
    *,
    deployment_id: str = "dep-a1b2c3",
    with_instance: bool | None = None,
    failure: FailureInfo | None = None,
) -> Deployment:
    """Build a Deployment in ``state``.

    ``with_instance`` defaults to True for any state that implies a live pod (BOOTING onward,
    excluding STOPPED/FAILED which have had their instance destroyed by the cost-safety rule).
    """
    live_states = {
        DeploymentState.BOOTING,
        DeploymentState.DOWNLOADING,
        DeploymentState.STARTING,
        DeploymentState.READY,
        DeploymentState.DEGRADED,
        DeploymentState.STOPPING,
    }
    if with_instance is None:
        with_instance = state in live_states

    endpoint = (
        RUNNING_INSTANCE.public_url
        if state in {DeploymentState.READY, DeploymentState.DEGRADED}
        else None
    )
    return Deployment(
        id=deployment_id,
        model_id="qwen3-0.6b",
        provider="runpod",
        desired_state=DeploymentState.STOPPED
        if state in {DeploymentState.STOPPING, DeploymentState.STOPPED}
        else DeploymentState.READY,
        observed_state=state,
        profile=QWEN3_06B_PROFILE,
        instance=RUNNING_INSTANCE if with_instance else None,
        endpoint_url=endpoint,
        state_history=[
            StateTransition(
                from_state=DeploymentState.REQUESTED,
                to_state=state,
                at=_T0,
                reason="fixture",
            )
        ],
        failure=failure,
        created_at=_T0,
        updated_at=_T0,
    )


FAILED_DEPLOYMENT = make_deployment(
    DeploymentState.FAILED,
    failure=FailureInfo(
        stage=DeploymentState.DOWNLOADING,
        message="OOM on correctly-sized GPU",
        retryable=False,
        attempts=1,
    ),
)

# One canonical deployment per state, for matrix-style tests.
BY_STATE: dict[DeploymentState, Deployment] = {
    state: (FAILED_DEPLOYMENT if state is DeploymentState.FAILED else make_deployment(state))
    for state in DeploymentState
}
