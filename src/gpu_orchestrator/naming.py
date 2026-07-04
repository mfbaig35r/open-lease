"""The instance-naming convention: one source of truth (spec §7.5).

Every provider instance is named ``gpu-orch-{namespace}-{deployment_id}``. The namespace scopes an
install so the orphan sweep never touches another install's pods (the A2 hardening). Both the config
layer and the provider layer build/filter names, so the convention lives here and nowhere else (E3).
"""

from __future__ import annotations

_PREFIX = "gpu-orch"


def instance_name(namespace: str, deployment_id: str) -> str:
    return f"{_PREFIX}-{namespace}-{deployment_id}"


def instance_prefix(namespace: str) -> str:
    """The prefix the orphan sweep and list filters match on: only this install's pods."""
    return f"{_PREFIX}-{namespace}-"
