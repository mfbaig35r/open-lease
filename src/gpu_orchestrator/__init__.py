"""gpu_orchestrator public API.

Phase 1 build in progress. The ``Orchestrator`` facade is added at build step 5; for now this
exposes the contract (domain models, errors, config) so downstream steps import from one place.
"""

from __future__ import annotations

from . import errors
from .config import Config
from .models import (
    CheckResult,
    CloudType,
    CostEstimate,
    CostRecord,
    Deployment,
    DeploymentState,
    Event,
    EventKind,
    FailureInfo,
    GPUType,
    HealthState,
    HealthStatus,
    Instance,
    InstanceRequest,
    ModelSpec,
    ProviderCapabilities,
    ProviderInfo,
    ReconcileAction,
    RuntimeOverrides,
    RuntimeProfile,
    StateTransition,
    ValidationMetadata,
    VolumeSpec,
)

__all__ = [
    "Config",
    "errors",
    # enums
    "DeploymentState",
    "ReconcileAction",
    "HealthState",
    "CloudType",
    "EventKind",
    # catalog
    "ModelSpec",
    "RuntimeProfile",
    "ValidationMetadata",
    "RuntimeOverrides",
    # provider / compute
    "GPUType",
    "VolumeSpec",
    "InstanceRequest",
    "Instance",
    "ProviderCapabilities",
    "ProviderInfo",
    # deployment
    "StateTransition",
    "FailureInfo",
    "Deployment",
    # health
    "CheckResult",
    "HealthStatus",
    # cost / event
    "CostRecord",
    "CostEstimate",
    "Event",
]
