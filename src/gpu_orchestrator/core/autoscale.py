"""Demand-driven replica scaling (capacity plan, Tier B2): decide how many deployments should serve
a model, given the recent served request rate, and build a replica record when the daemon scales up.

Pure, like ``core/schedule.py`` and ``core/budgets.py``: no clock, no I/O. The daemon reads the
recent request count and the wall-clock, computes the rate, and calls ``desired_replicas``; it
builds new replica records with ``replica_from``. The signal is served rate, so it tracks sustained
load and cannot see demand rejected at saturation (set the target below a replica's ceiling).
"""

from __future__ import annotations

import math

from ..models import AutoscalePolicy, Deployment, DeploymentState


def desired_replicas(recent_rpm: float, policy: AutoscalePolicy) -> int:
    """How many replicas the served rate implies, clamped to the policy's min/max. Each replica
    should carry about ``target_rpm_per_replica`` requests per minute, so N = ceil(rate / target),
    floored at ``min_replicas`` (always keep the warm floor) and capped at ``max_replicas``."""
    needed = math.ceil(recent_rpm / policy.target_rpm_per_replica) if recent_rpm > 0 else 0
    return max(policy.min_replicas, min(policy.max_replicas, needed))


def replica_from(template: Deployment, new_id: str) -> Deployment:
    """Build one more replica from a template deployment: a fresh id and a clean desired-READY
    record, carrying the template's model, provider, profile, concurrency limits, and schedule.
    Pure; the caller persists, emits, and (for an inline scale) drives it."""
    return Deployment(
        id=new_id,
        model_id=template.model_id,
        provider=template.provider,
        hf_repo=template.hf_repo,
        context_window=template.context_window,
        desired_state=DeploymentState.READY,
        observed_state=DeploymentState.REQUESTED,
        profile=template.profile,
        max_concurrency=template.max_concurrency,
        max_queue=template.max_queue,
        queue_timeout_s=template.queue_timeout_s,
        schedule=template.schedule,
    )
