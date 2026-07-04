"""Exception hierarchy.

Every error raised by the orchestrator is an ``OrchestratorError`` subclass. Interfaces
(CLI/API/MCP) catch this base to render one clear human sentence; internal code catches specific
subclasses. Never let a bare third-party or builtin exception escape a provider/runtime boundary:
convert it to the right subclass here (reliability constraint, CLAUDE.md).
"""

from __future__ import annotations


class OrchestratorError(Exception):
    """Base for every error this package raises."""


# --- Configuration -------------------------------------------------------------------


class ConfigError(OrchestratorError):
    """Invalid or missing configuration (bad TOML, missing credential, bad precedence value)."""


# --- Persistence / schema ------------------------------------------------------------


class StoreError(OrchestratorError):
    """State store failure."""


class SchemaVersionError(StoreError):
    """A persisted document carries a schema_version this build cannot load.

    Fail loudly, never silently: a document with an unknown version is a bug or a downgrade, not
    something to guess at (spec §6).
    """


# --- Catalog -------------------------------------------------------------------------


class CatalogError(OrchestratorError):
    """Model catalog could not be loaded or validated."""


class ModelNotFoundError(CatalogError):
    """No catalog entry for the requested model id."""


class InvalidProfileError(CatalogError):
    """A catalog entry is malformed or missing required validation metadata (spec §14)."""


# --- Deployments ---------------------------------------------------------------------


class DeploymentNotFoundError(OrchestratorError):
    """No deployment with the given id."""


class ReconcileError(OrchestratorError):
    """The reconciler could not make progress for a reason that is not a provider/runtime fault."""


class TimeoutBudgetExceeded(OrchestratorError):
    """A deployment stage ran past its configured budget (spec §7.3)."""


# --- Provider seam -------------------------------------------------------------------


class ProviderError(OrchestratorError):
    """Base for provider (compute) faults."""


class ProviderAPIError(ProviderError):
    """The provider API returned an error or was unreachable."""


class InstanceCreationError(ProviderError):
    """The provider accepted the request but the instance never came up."""


class NotSupportedError(ProviderError):
    """The provider does not implement an optional capability in this phase (spec §8).

    Example: RunPod volume/snapshot methods may raise this in Phase 1.
    """


# --- Runtime seam --------------------------------------------------------------------


class RuntimeError_(OrchestratorError):
    """Base for runtime (model-serving) faults.

    Named with a trailing underscore to avoid shadowing the builtin ``RuntimeError``.
    """


class RuntimeLaunchError(RuntimeError_):
    """The runtime server failed to start or load the model."""
