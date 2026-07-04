"""Configuration: settings, precedence, and credential loading (spec §12).

Precedence, highest wins:

    CLI flags (init kwargs)
      -> environment variables (GPU_ORCH_*, plus RUNPOD_API_KEY, HF_TOKEN)
        -> ./gpu-orchestrator.toml
          -> ~/.gpu-orchestrator/config.toml
            -> defaults

Secrets are ``SecretStr`` so they never appear in logs, ``repr``, or ``state.db``. Use
``Config.effective()`` to render config for ``gpu config`` with secrets masked.
"""

from __future__ import annotations

import re
import socket
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from . import naming

PROJECT_TOML = Path("./gpu-orchestrator.toml")
USER_TOML = Path.home() / ".gpu-orchestrator" / "config.toml"
DEFAULT_STATE_DB = Path.home() / ".gpu-orchestrator" / "state.db"

_SECRET_FIELDS = {"runpod_api_key", "hf_token"}


def _default_namespace() -> str:
    """A stable per-install namespace so the orphan sweep never touches another install's pods.

    Defaults to the sanitized hostname (spec §7.5). Override with GPU_ORCH_NAMESPACE.
    """
    host = socket.gethostname().split(".")[0].lower()
    cleaned = re.sub(r"[^a-z0-9-]", "-", host).strip("-")
    return cleaned or "default"


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GPU_ORCH_",
        extra="ignore",
        # Read ./.env so the dotenv source in settings_customise_sources is actually populated;
        # without this, that source is a no-op and the documented .env convenience does nothing.
        env_file=".env",
        env_file_encoding="utf-8",
        # SecretStr keeps credentials out of repr/logs by construction.
    )

    # --- identity / isolation -------------------------------------------------------
    namespace: str = Field(default_factory=_default_namespace)

    # --- credentials (unprefixed env names, per spec §12) ---------------------------
    runpod_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("RUNPOD_API_KEY")
    )
    hf_token: SecretStr | None = Field(default=None, validation_alias=AliasChoices("HF_TOKEN"))

    # --- storage --------------------------------------------------------------------
    state_db: Path = DEFAULT_STATE_DB

    # --- reconcile / health loops (seconds) -----------------------------------------
    reconcile_interval: int = 10
    orphan_sweep_interval: int = 300
    orphan_grace_period: int = 120
    health_poll_interval: int = 30
    health_failure_threshold: int = 3

    # --- retry / backoff ------------------------------------------------------------
    retry_max_attempts: int = 3
    retry_backoff_min: int = 10
    retry_backoff_max: int = 60

    # --- per-stage timeout budgets (seconds), spec §7.3 -----------------------------
    timeout_provisioning: int = 300
    timeout_booting: int = 300
    timeout_download: int = 1800
    timeout_starting: int = 300

    # --- proxy ----------------------------------------------------------------------
    proxy_host: str = "localhost"
    proxy_port: int = 8080

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Order is priority order (first wins). This IS the documented precedence.
        return (
            init_settings,  # CLI flags
            env_settings,  # GPU_ORCH_*, secrets
            dotenv_settings,  # .env convenience
            TomlConfigSettingsSource(settings_cls, PROJECT_TOML),  # ./gpu-orchestrator.toml
            TomlConfigSettingsSource(settings_cls, USER_TOML),  # ~/.gpu-orchestrator/...
            file_secret_settings,
        )

    def instance_name(self, deployment_id: str) -> str:
        """The mandatory ``gpu-orch-{namespace}-{deployment_id}`` tag (spec §7.5)."""
        return naming.instance_name(self.namespace, deployment_id)

    def instance_prefix(self) -> str:
        """The prefix the orphan sweep filters on: only this install's pods (spec §7.5)."""
        return naming.instance_prefix(self.namespace)

    def effective(self) -> dict[str, object]:
        """Config for display (``gpu config``) with secrets masked, never revealed."""
        out: dict[str, object] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if name in _SECRET_FIELDS:
                out[name] = "***set***" if value is not None else None
            elif isinstance(value, Path):
                out[name] = str(value)
            else:
                out[name] = value
        return out
