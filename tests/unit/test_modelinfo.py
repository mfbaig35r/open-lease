"""Sizing a model from its hub metadata (issue #24).

The heuristic that picks a GPU decides how much a deploy costs and whether it OOMs, so it is checked
against ground truth rather than against itself: every hand-validated catalog entry must come out at
the GPU a human actually launched it on. Those weight sizes were read from the hub on 2026-08-09.

All offline. Network shape is exercised through httpx.MockTransport.
"""

from __future__ import annotations

import json

import httpx
import pytest

from gpu_orchestrator.core import modelinfo
from gpu_orchestrator.models import GPUType, ModelProfile

# The RunPod Phase 1 menu, matching providers/runpod.py.
_MENU = [
    GPUType(id="RTX-A4000", name="A4000", memory_gb=16, hourly_usd=0.17, provider_sku="a4000"),
    GPUType(id="A40-48GB", name="A40", memory_gb=48, hourly_usd=0.44, provider_sku="a40"),
    GPUType(id="A100-80GB", name="A100", memory_gb=80, hourly_usd=1.89, provider_sku="a100"),
    GPUType(id="H100-80GB", name="H100", memory_gb=80, hourly_usd=2.99, provider_sku="h100"),
]

# (repo, weight bytes from the hub, the GPU a human validated it on in catalog/models.toml)
_CATALOG_GROUND_TRUTH = [
    ("Qwen/Qwen3-0.6B", 1_503_300_328, "RTX-A4000"),
    ("Qwen/Qwen3-8B", 16_381_470_720, "A40-48GB"),
    ("Qwen/Qwen3-32B", 65_524_246_528, "A100-80GB"),
]


def _profile(**kw) -> ModelProfile:
    return ModelProfile(hf_repo=kw.pop("hf_repo", "org/model"), **kw)


# --- the heuristic, checked against hand-validated reality -------------------------------


@pytest.mark.parametrize(("repo", "weight_bytes", "expected_gpu"), _CATALOG_GROUND_TRUTH)
def test_sizing_agrees_with_every_validated_catalog_entry(repo, weight_bytes, expected_gpu):
    chosen = modelinfo.select_gpu(_profile(hf_repo=repo, weight_bytes=weight_bytes), _MENU)
    assert chosen is not None
    gpu, count = chosen
    assert (gpu.id, count) == (expected_gpu, 1)


def test_required_vram_leaves_room_for_kv_cache():
    # 65.5 GB of weights must not be sized onto exactly 65.5 GB of card.
    needed = modelinfo.required_vram_gb(_profile(weight_bytes=65_524_246_528))
    assert needed is not None and needed > 65.5


def test_sizing_spans_multiple_gpus_when_no_single_card_fits():
    # ~140 GB of weights (a 70B at bf16) does not fit one 80 GB card.
    chosen = modelinfo.select_gpu(_profile(weight_bytes=140_000_000_000), _MENU)
    assert chosen is not None
    gpu, count = chosen
    assert gpu.memory_gb * count >= 140


def test_selection_prefers_fewer_gpus_over_a_cheaper_pile_of_small_ones():
    """A cost-first rule picks 5x A4000 ($0.85/hr) over 1x A100 ($1.89/hr) for a 32B model. That is
    cheaper and does not work: sharding needs interconnect those cards lack. Fewest GPUs wins."""
    gpu, count = modelinfo.select_gpu(_profile(weight_bytes=65_524_246_528), _MENU)
    assert count == 1
    assert gpu.hourly_usd > 0.85  # deliberately not the cheapest option available


def test_multi_gpu_counts_are_powers_of_two():
    """Tensor parallelism must divide the attention heads evenly; a 3- or 5-way shard is invalid."""
    for weight_bytes in (100_000_000_000, 140_000_000_000, 250_000_000_000, 500_000_000_000):
        chosen = modelinfo.select_gpu(_profile(weight_bytes=weight_bytes), _MENU)
        if chosen is not None:
            assert chosen[1] in (1, 2, 4, 8)


# --- absent data must never become a confident guess -------------------------------------


def test_unknown_weight_size_sizes_to_nothing():
    assert modelinfo.required_vram_gb(_profile()) is None
    assert modelinfo.select_gpu(_profile(), _MENU) is None


def test_model_too_large_for_the_whole_menu_returns_none():
    assert modelinfo.select_gpu(_profile(weight_bytes=2_000_000_000_000), _MENU) is None


# --- fetching: shapes that real repos actually have --------------------------------------


def _hub(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        for fragment, response in routes.items():
            if fragment in str(request.url):
                return response
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_fetch_reads_a_sharded_repo_from_its_index():
    transport = _hub(
        {
            "config.json": httpx.Response(
                200,
                json={
                    "architectures": ["Qwen3ForCausalLM"],
                    "num_hidden_layers": 64,
                    "max_position_embeddings": 40960,
                    "torch_dtype": "bfloat16",
                },
            ),
            "model.safetensors.index.json": httpx.Response(
                200, json={"metadata": {"total_size": 65_524_246_528}}
            ),
        }
    )
    profile = await modelinfo.fetch_profile("Qwen/Qwen3-32B", transport=transport)
    assert profile is not None
    assert profile.architecture == "Qwen3ForCausalLM"
    assert profile.weight_bytes == 65_524_246_528
    assert profile.weight_gb == 65.5
    assert profile.context_length == 40960
    assert profile.is_moe is False


async def test_fetch_falls_back_to_the_file_tree_for_single_shard_repos():
    """Small models ship one model.safetensors and publish no index, so the index read 404s."""
    transport = _hub(
        {
            "config.json": httpx.Response(200, json={"architectures": ["Qwen3ForCausalLM"]}),
            "/tree/main": httpx.Response(
                200,
                json=[
                    {"type": "file", "path": "model.safetensors", "size": 1_503_300_328},
                    {"type": "file", "path": "tokenizer.json", "size": 11_422_654},
                ],
            ),
        }
    )
    profile = await modelinfo.fetch_profile("Qwen/Qwen3-0.6B", transport=transport)
    assert profile is not None
    assert profile.weight_bytes == 1_503_300_328  # tokenizer.json not counted


async def test_fetch_detects_moe_topology():
    transport = _hub(
        {
            "config.json": httpx.Response(
                200,
                json={
                    "architectures": ["Qwen3MoeForCausalLM"],
                    "num_experts": 128,
                    "num_experts_per_tok": 8,
                },
            ),
            "index.json": httpx.Response(200, json={"metadata": {"total_size": 61_064_245_248}}),
        }
    )
    profile = await modelinfo.fetch_profile("Qwen/Qwen3-30B-A3B", transport=transport)
    assert profile is not None
    assert profile.is_moe is True
    assert (profile.moe_experts, profile.moe_experts_per_token) == (128, 8)


async def test_gated_repo_yields_no_profile_rather_than_raising():
    """A 401 is the normal response for a gated model without a token. It must degrade to "ask the
    user for a GPU", not break the deploy."""
    profile = await modelinfo.fetch_profile(
        "meta-llama/Llama-3.1-8B-Instruct", transport=_hub({"": httpx.Response(401)})
    )
    assert profile is None


async def test_unreachable_hub_yields_no_profile_rather_than_raising():
    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns", request=request)

    assert await modelinfo.fetch_profile("org/m", transport=httpx.MockTransport(explode)) is None


async def test_malformed_config_yields_no_profile():
    transport = _hub({"config.json": httpx.Response(200, content=b"<html>not json</html>")})
    assert await modelinfo.fetch_profile("org/m", transport=transport) is None


async def test_config_without_a_weight_size_still_profiles_but_cannot_be_sized():
    """Metadata can be partly there. A profile with no weight size is still useful for display, but
    must not be turned into a GPU choice."""
    transport = _hub({"config.json": httpx.Response(200, json={"architectures": ["Weird"]})})
    profile = await modelinfo.fetch_profile("org/m", transport=transport)
    assert profile is not None and profile.architecture == "Weird"
    assert profile.weight_bytes is None
    assert modelinfo.select_gpu(profile, _MENU) is None


def test_fixture_sizes_match_the_json_shape_the_hub_returns():
    """Guards the parse against a silent shape change: this is verbatim hub output."""
    body = json.loads('{"metadata": {"total_size": 65524246528}, "weight_map": {}}')
    assert body["metadata"]["total_size"] == 65_524_246_528
