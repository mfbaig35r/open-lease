"""Model catalog: curated data loaded and validated on startup (spec §14).

Models are data, not code. Each TOML entry becomes a ``ModelSpec`` plus a ``RuntimeProfile`` (which
carries required ``ValidationMetadata`` - a profile without it does not load). Adding a model is
adding a TOML block; no code change.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import ValidationError

from ..errors import InvalidProfileError, ModelNotFoundError
from ..models import ModelSpec, RuntimeProfile


def default_catalog_path() -> Path:
    """The repo's ``catalog/models.toml`` (spec §5).

    Resolved relative to this file for the editable dev install. Packaging the catalog into the
    wheel is a follow-up (it currently lives at the repo root, outside ``src/``).
    """
    return Path(__file__).resolve().parents[3] / "catalog" / "models.toml"


class Catalog:
    def __init__(self, specs: dict[str, ModelSpec], profiles: dict[str, RuntimeProfile]) -> None:
        self._specs = specs
        self._profiles = profiles

    def list_models(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def get_spec(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise ModelNotFoundError(f"No catalog entry for model {model_id!r}") from None

    def get_profile(self, model_id: str) -> RuntimeProfile:
        try:
            return self._profiles[model_id]
        except KeyError:
            raise ModelNotFoundError(f"No profile for model {model_id!r}") from None


def load_catalog(path: Path | None = None) -> Catalog:
    path = path or default_catalog_path()
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise InvalidProfileError(f"Cannot read catalog {path}: {exc}") from exc

    specs: dict[str, ModelSpec] = {}
    profiles: dict[str, RuntimeProfile] = {}
    for key, entry in data.get("models", {}).items():
        if "profile" not in entry:
            raise InvalidProfileError(f"Catalog entry {key!r} has no [models.{key}.profile]")
        profile_data = dict(entry["profile"])
        spec_data = {k: v for k, v in entry.items() if k != "profile"}
        try:
            spec = ModelSpec.model_validate(spec_data)
            profile_data["model_id"] = spec.id  # derive, never hand-written in TOML
            profile = RuntimeProfile.model_validate(profile_data)
        except ValidationError as exc:
            raise InvalidProfileError(f"Catalog entry {key!r} is invalid: {exc}") from exc
        specs[spec.id] = spec
        profiles[spec.id] = profile
    return Catalog(specs, profiles)
