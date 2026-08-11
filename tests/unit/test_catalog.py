"""Step-4 catalog tests: the real catalog loads and validates; bad entries fail loudly (§14)."""

from __future__ import annotations

import pytest

from gpu_orchestrator.core.catalog import load_catalog
from gpu_orchestrator.errors import InvalidProfileError, ModelNotFoundError
from gpu_orchestrator.runtimes import RUNTIMES

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
    """Asserts invariants, not an exact id list. The build order calls for growing this to 10-15
    entries, and a hardcoded set turns every addition into a test failure that teaches nothing."""
    catalog = load_catalog()
    ids = {m.id for m in catalog.list_models()}
    assert {"qwen3-0.6b", "qwen3-32b"} <= ids  # the two the live gauntlet ran against
    assert len(ids) == len(catalog.list_models())  # ids are unique


def test_every_profile_names_a_registered_runtime():
    """A profile can ask for any engine; an unregistered one would fail at deploy time on a real
    pod instead of at load time here."""
    catalog = load_catalog()
    for spec in catalog.list_models():
        assert catalog.get_profile(spec.id).runtime in RUNTIMES


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


def test_every_profile_can_serve_one_max_length_request():
    """The check that would have caught qwen3-32b before a pod was ever rented.

    A profile is only deployable if weights AND one max-length request's KV cache fit inside
    ``gpu_memory_utilization`` of the recommended card. qwen3-32b shipped pairing a 32768 context
    with 0.90 utilisation on an 80GB card: 61 GiB of bf16 weights left 4.5 GiB of KV where 8.0 GiB
    was needed, so vLLM exited during engine init on every start. RunPod restarts the container in
    place, so from outside the pod stayed RUNNING with a dead port and the deployment sat in
    starting_server for the full 2400s budget before being torn down and retried. Three attempts and
    roughly $2.90 went into diagnosing arithmetic that fits on one line.

    Deliberately conservative: real engines also reserve memory for activations and CUDA graphs, so
    a profile that only just passes here can still fail on a pod. Passing is necessary, not
    sufficient.
    """
    catalog = load_catalog()
    gpu_vram_gb = {"RTX-A4000": 16, "A40-48GB": 48, "A100-80GB": 80, "H100-80GB": 80}
    for spec in catalog.list_models():
        if spec.kv_bytes_per_token is None:
            continue  # e.g. a GGUF entry with no HF config to read the architecture from
        profile = catalog.get_profile(spec.id)
        vram = gpu_vram_gb.get(profile.recommended_gpu)
        if vram is None:
            continue
        params = float(spec.parameter_count.rstrip("Bb"))
        bytes_per_param = 0.55 if spec.quantization else 2.0
        weights_gb = params * bytes_per_param
        kv_gb = spec.kv_bytes_per_token * spec.context_window / 1e9
        budget_gb = vram * profile.gpu_memory_utilization * profile.tensor_parallel
        assert weights_gb + kv_gb <= budget_gb, (
            f"{spec.id} cannot serve one {spec.context_window}-token request on "
            f"{profile.tensor_parallel}x {profile.recommended_gpu}: needs "
            f"{weights_gb:.1f}GB weights + {kv_gb:.1f}GB KV = {weights_gb + kv_gb:.1f}GB, "
            f"budget is {budget_gb:.1f}GB"
        )
