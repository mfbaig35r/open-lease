"""`gpu plan`: rank the ways to run a model from data already on hand.

The ranking steers what people rent, so it is pure and tested offline. The load-bearing test here is
``test_plan_recommendation_agrees_with_the_deploy_time_chooser``: `gpu plan` recommends and
`gpu deploy` chooses, and a tool that advises one shape then provisions another is worse than one
that stays quiet.
"""

from __future__ import annotations

from gpu_orchestrator.core import modelinfo, plan
from gpu_orchestrator.models import GPUType, ModelProfile, ValidationMetadata

_A4000 = GPUType(
    id="RTX-A4000", name="A4000", memory_gb=16, hourly_usd=0.17, provider_sku="sku-a4000"
)
_A40 = GPUType(id="A40-48GB", name="A40", memory_gb=48, hourly_usd=0.44, provider_sku="sku-a40")
_A100 = GPUType(id="A100-80GB", name="A100", memory_gb=80, hourly_usd=1.89, provider_sku="sku-a100")
_H100 = GPUType(id="H100-80GB", name="H100", memory_gb=80, hourly_usd=2.99, provider_sku="sku-h100")
_MENU = [_A4000, _A40, _A100, _H100]
_ALL_SKUS = {g.provider_sku for g in _MENU}


def _validation(gpu: str, concurrent: float) -> ValidationMetadata:
    return ValidationMetadata(
        validated_at="2026-08-09",
        validated_provider="runpod",
        validated_gpu=gpu,
        validated_image="img",
        startup_timeout_seconds=600,
        throughput_measured_at="2026-08-09",
        throughput_gpu=gpu,
        tokens_per_sec=31.9,
        tokens_per_sec_concurrent=concurrent,
        measured_concurrency=16,
    )


def _by_id(options, gpu_id):
    return next(o for o in options if o.gpu_type == gpu_id)


# --- the consistency requirement ---------------------------------------------------------


def test_plan_recommendation_agrees_with_the_deploy_time_chooser():
    """Whatever `gpu plan` puts a marker next to must be what `modelinfo.select_gpu` would pick.

    These are two independent implementations of "which GPU should this run on" and they will drift
    apart the moment someone tunes one of them. Contradictory advice from one tool is worse than no
    advice.
    """
    for weight_bytes in (1_500_000_000, 16_000_000_000, 60_000_000_000):
        profile = ModelProfile(hf_repo="org/m", weight_bytes=weight_bytes)
        required = modelinfo.required_vram_gb(profile)
        chosen = modelinfo.select_gpu(profile, _MENU)
        options = plan.build_options(
            required_vram_gb=required, gpu_types=_MENU, available_skus=_ALL_SKUS
        )
        recommended = [o for o in options if o.recommended]
        assert chosen is not None and len(recommended) == 1
        assert (recommended[0].gpu_type, recommended[0].gpu_count) == (chosen[0].id, chosen[1])


# --- fit ---------------------------------------------------------------------------------


def test_a_gpu_too_small_is_listed_but_marked_unfit():
    """Listing it is useful (it explains why it is not the answer); hiding it is not.

    "Too small" means too small even at eight of them: 8x 16GB A4000 is 128GB, so a 70GB model does
    fit them, and only something past 128GB genuinely does not.
    """
    options = plan.build_options(required_vram_gb=200.0, gpu_types=_MENU, available_skus=_ALL_SKUS)
    assert _by_id(options, "RTX-A4000").fits is False  # 8x16 = 128GB, still short
    assert _by_id(options, "RTX-A4000").note == "too small"
    assert _by_id(options, "A100-80GB").fits is True  # 4x80 = 320GB


def test_unknown_size_does_not_rule_anything_out():
    """A gated repo cannot be sized. Showing an empty table would be worse than showing a price
    list with an honest gap in it."""
    options = plan.build_options(required_vram_gb=None, gpu_types=_MENU, available_skus=_ALL_SKUS)
    assert all(o.fits for o in options)


# --- stock -------------------------------------------------------------------------------


def test_out_of_stock_options_sink_but_stay_visible():
    options = plan.build_options(
        required_vram_gb=10.0, gpu_types=_MENU, available_skus={_A100.provider_sku}
    )
    assert _by_id(options, "A100-80GB").in_stock is True
    assert _by_id(options, "RTX-A4000").in_stock is False
    assert options[0].gpu_type == "A100-80GB"  # the only in-stock option ranks first


def test_no_availability_data_is_unknown_not_out_of_stock():
    """Three-valued on purpose. A provider with no availability endpoint must not have every option
    read as unavailable."""
    options = plan.build_options(required_vram_gb=10.0, gpu_types=_MENU, available_skus=set())
    assert all(o.in_stock is None for o in options)
    assert any(o.recommended for o in options)


# --- throughput is never projected -------------------------------------------------------


def test_cost_per_token_appears_only_on_the_gpu_it_was_measured_on():
    options = plan.build_options(
        required_vram_gb=10.0,
        gpu_types=_MENU,
        available_skus=_ALL_SKUS,
        validation=_validation("A40-48GB", 430.8),
    )
    assert _by_id(options, "A40-48GB").cost_per_mtok is not None
    for other in ("RTX-A4000", "A100-80GB", "H100-80GB"):
        assert _by_id(options, other).cost_per_mtok is None
        assert _by_id(options, other).note == "not benchmarked"


def test_cost_per_mtok_arithmetic_matches_the_hourly_rate():
    options = plan.build_options(
        required_vram_gb=10.0,
        gpu_types=[_A40],
        available_skus=_ALL_SKUS,
        validation=_validation("A40-48GB", 430.8),
    )
    expected = round(0.44 / (430.8 * 3600) * 1_000_000, 2)
    assert options[0].cost_per_mtok == expected


# --- what may be recommended -------------------------------------------------------------


def test_an_unvalidated_multi_gpu_shard_is_offered_but_never_recommended():
    """Every catalog entry open-lease has validated is tensor_parallel=1, so no shard of anything
    has been proven here. When only shards are in stock the honest output is a table with no
    recommendation, not a confident wrong one."""
    options = plan.build_options(
        required_vram_gb=70.0,  # needs 8x A4000, or one A100/H100
        gpu_types=_MENU,
        available_skus={_A4000.provider_sku},
    )
    shard = _by_id(options, "RTX-A4000")
    assert shard.fits and shard.gpu_count > 1
    assert "unvalidated" in shard.note
    assert not any(o.recommended for o in options)


def test_a_single_gpu_that_fits_and_is_in_stock_is_recommended():
    options = plan.build_options(
        required_vram_gb=70.0, gpu_types=_MENU, available_skus={_A100.provider_sku}
    )
    assert _by_id(options, "A100-80GB").recommended is True
