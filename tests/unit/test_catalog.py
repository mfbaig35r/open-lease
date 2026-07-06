"""Step-4 catalog tests: the real catalog loads and validates; bad entries fail loudly (§14)."""

from __future__ import annotations

import pytest

from gpu_orchestrator.core.catalog import load_catalog
from gpu_orchestrator.errors import InvalidProfileError, ModelNotFoundError

_GOOD_ENTRY = """
[models.tiny]
id = "tiny"
hf_repo = "org/tiny"
family = "test"
parameter_count = "0.1B"
min_gpu_memory_gb = 4
context_window = 4096
license = "apache-2.0"

[models.tiny.profile]
image = "vllm/vllm-openai:v0.9.1"
recommended_gpu = "RTX-A4000"
min_disk_gb = 10

[models.tiny.profile.validation]
validated_at = "2026-07-03"
validated_provider = "runpod"
validated_gpu = "RTX-A4000"
validated_image = "vllm/vllm-openai:v0.9.1"
startup_timeout_seconds = 300
"""


def test_real_catalog_loads_models():
    catalog = load_catalog()
    ids = {m.id for m in catalog.list_models()}
    assert ids == {"qwen3-0.6b", "llama-3.1-8b-instruct", "qwen3-8b", "qwen3-32b"}


def test_every_profile_carries_validation_metadata():
    catalog = load_catalog()
    for spec in catalog.list_models():
        profile = catalog.get_profile(spec.id)
        assert profile.model_id == spec.id
        assert profile.validation.startup_timeout_seconds > 0


def test_get_spec_and_profile_and_missing():
    catalog = load_catalog()
    assert catalog.get_spec("qwen3-32b").hf_repo == "Qwen/Qwen3-32B"
    with pytest.raises(ModelNotFoundError):
        catalog.get_spec("nope")
    with pytest.raises(ModelNotFoundError):
        catalog.get_profile("nope")


def test_good_temp_catalog_loads(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(_GOOD_ENTRY)
    catalog = load_catalog(path)
    assert catalog.get_spec("tiny").hf_repo == "org/tiny"


def test_profile_without_validation_is_rejected(tmp_path):
    bad = _GOOD_ENTRY.replace(
        """
[models.tiny.profile.validation]
validated_at = "2026-07-03"
validated_provider = "runpod"
validated_gpu = "RTX-A4000"
validated_image = "vllm/vllm-openai:v0.9.1"
startup_timeout_seconds = 300
""",
        "",
    )
    path = tmp_path / "bad.toml"
    path.write_text(bad)
    with pytest.raises(InvalidProfileError):
        load_catalog(path)


def test_entry_without_profile_is_rejected(tmp_path):
    path = tmp_path / "noprofile.toml"
    path.write_text(
        '[models.x]\nid = "x"\nhf_repo = "o/x"\nfamily = "t"\nparameter_count = "1B"\n'
        'min_gpu_memory_gb = 4\ncontext_window = 4096\nlicense = "apache-2.0"\n'
    )
    with pytest.raises(InvalidProfileError):
        load_catalog(path)
