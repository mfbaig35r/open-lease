"""Rank the ways to run a model right now: what fits, what is in stock, what it costs per token.

open-lease knew all four of these facts and had no place that put them together. Fit came from
``modelinfo``, stock from ``Provider.gpu_availability``, speed from ``ValidationMetadata``, and
price from ``GPUType``. Answering "where should this run" meant reading three commands and doing
the arithmetic yourself.

This is deliberately NOT the execution planner that docs/adr-adaptive-execution-planning.md
rejected. That one searched over execution strategies, needed real provisioning and benchmarking to
evaluate a candidate, and collapsed to a single viable cell. This ranks a menu that already
exists, from data already on hand, with no pods and no spend. It is a table sort.

The ranking is pure. Everything that needs the network is resolved by the caller and passed in, so
the rule that steers what people rent is exhaustively testable offline.
"""

from __future__ import annotations

from ..models import GPUType, PlanOption, ValidationMetadata

_MAX_GPUS = 8
_COUNTS = (1, 2, 4, 8)  # tensor parallelism must divide the attention heads evenly


def build_options(
    *,
    required_vram_gb: float | None,
    gpu_types: list[GPUType],
    available_skus: set[str],
    validation: ValidationMetadata | None = None,
) -> list[PlanOption]:
    """One row per GPU shape, best first.

    ``available_skus`` empty means the provider did not report availability, which is different from
    reporting that nothing is available. Those rows carry ``in_stock=None`` and are not penalised,
    for the same reason GPU substitution leaves a recommendation alone when it has no stock data.

    Throughput is only ever attached to the GPU it was actually measured on. A model benchmarked on
    an A40 says nothing about how it runs on an H100, so those rows report no cost per token rather
    than a projection. That leaves most rows blank until the catalog is measured, which is honest
    about what is known rather than flattering about it.
    """
    rows: list[PlanOption] = []
    for gpu in gpu_types:
        count = _smallest_count_that_fits(gpu, required_vram_gb)
        fits = count is not None
        count = count or 1
        in_stock = (gpu.provider_sku in available_skus) if available_skus else None
        hourly = round(gpu.hourly_usd * count, 4)
        tps = _throughput_for(gpu, validation)
        rows.append(
            PlanOption(
                gpu_type=gpu.id,
                gpu_count=count,
                fits=fits,
                in_stock=in_stock,
                hourly_usd=hourly,
                tokens_per_sec=tps,
                note=_note(fits, in_stock, tps, count),
            )
        )
    rows.sort(key=_rank)
    for i, row in enumerate(rows):
        if _is_recommendable(row):
            rows[i] = row.model_copy(update={"recommended": True})
            break
    return rows


def _smallest_count_that_fits(gpu: GPUType, required_vram_gb: float | None) -> int | None:
    """Fewest GPUs of this type that hold the model, or None if eight cannot. Mirrors
    ``modelinfo.select_gpu``: power-of-two counts only, and never a shape vLLM would reject."""
    if required_vram_gb is None:
        return 1  # unknown size: cannot rule the GPU out, so do not pretend to
    for count in _COUNTS:
        if count <= _MAX_GPUS and gpu.memory_gb * count >= required_vram_gb:
            return count
    return None


def _throughput_for(gpu: GPUType, validation: ValidationMetadata | None) -> float | None:
    if validation is None or not validation.tokens_per_sec_concurrent:
        return None
    measured_on = validation.throughput_gpu or validation.validated_gpu
    if measured_on not in (gpu.id, gpu.provider_sku):
        return None
    return validation.tokens_per_sec_concurrent


def _note(fits: bool, in_stock: bool | None, tps: float | None, count: int) -> str:
    if not fits:
        return "too small"
    if in_stock is False:
        return "out of stock"
    if count > 1:
        return f"{count}-way shard, unvalidated"
    if tps is None:
        return "not benchmarked"
    return ""


def _is_recommendable(option: PlanOption) -> bool:
    """A row worth putting a marker next to.

    Every catalog entry open-lease has ever validated is tensor_parallel=1, so no multi-GPU shard of
    anything has been proven to work here. Listing one as an option is useful; pointing at it and
    saying "run this" would be asserting something nobody has checked. When only shards are in
    stock, the honest output is a table with no recommendation rather than a confident wrong one.
    """
    return option.fits and option.in_stock is not False and option.gpu_count == 1


def _rank(option: PlanOption) -> tuple:
    """Viable, then fewest GPUs, then cheapest per token where known, then cheapest per hour.

    Fewest GPUs comes before price for the same reason it does in ``modelinfo.select_gpu``: eight
    RTX A4000s are cheaper than one A100 and cannot shard a 32B model across the interconnect they
    do not have. Ranking on price alone recommended exactly that.

    This ordering is not just a good idea, it is a consistency requirement. ``gpu plan`` recommends
    and ``gpu deploy`` chooses, and a tool that advises one shape then provisions another is worse
    than one that stays quiet. A test pins the two rules together.

    Cost per token outranks cost per hour because it answers the question people actually have, but
    it only exists where something has been measured, so hourly price is the tie break.
    """
    return (
        not option.fits,
        option.in_stock is False,
        option.gpu_count,
        option.cost_per_mtok is None,
        option.cost_per_mtok or 0.0,
        option.hourly_usd,
    )
