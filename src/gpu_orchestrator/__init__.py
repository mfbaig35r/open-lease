"""gpu_orchestrator public API.

Exposes the contract (domain models, errors, config) from one place. The ``Orchestrator`` facade
and the Provider/Runtime seams live in ``gpu_orchestrator.core`` / ``.providers`` / ``.runtimes``;
import them from there so a models-only consumer does not pull in the whole engine.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

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

try:
    __version__ = version("open-lease")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+source"

__all__ = [
    "__version__",
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
