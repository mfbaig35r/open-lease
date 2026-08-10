"""Derive what a model IS from its Hugging Face metadata, so a deploy can size itself (issue #24).

``deploy_adhoc`` used to demand an explicit ``--gpu`` because nothing in open-lease knew how big a
model was. Everything needed to answer that is published as metadata and costs two small JSON reads,
no weights: ``config.json`` for the architecture and ``model.safetensors.index.json`` (or the file
tree, for single-shard repos) for the total weight size.

The split here follows the house pattern: ``fetch_profile`` does the I/O, and the sizing/selection
functions are PURE, so the arithmetic that decides how much you spend is exhaustively testable
offline.

Sizing is a heuristic and is deliberately biased to over-provision. Under-provisioning produces an
OOM several minutes into a paid cold start, which costs real money and fails; over-provisioning
costs a little more per hour and works. When the metadata is missing or unreadable (a gated repo, a
custom layout), every function here returns None and the caller falls back to asking for a GPU
rather than guessing.
"""

from __future__ import annotations

import httpx

from ..logging import get_logger
from ..models import GPUType, ModelProfile

_log = get_logger("modelinfo")

_HF = "https://huggingface.co"
_TIMEOUT = 15.0

# vLLM fits weights AND KV cache inside gpu_memory_utilization of the card, so that fraction is the
# divisor rather than an extra margin. The KV floor is what is left for context; 2 GB is enough for
# a few thousand tokens on a small model and is dwarfed by the weights on a large one.
_KV_FLOOR_GB = 2.0
_BYTES_PER_GB = 1_000_000_000


async def fetch_profile(
    hf_repo: str,
    *,
    token: str | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ModelProfile | None:
    """Read a model's public metadata. ``None`` when it cannot be read (gated, missing, malformed).

    Never raises: an unreachable hub must degrade to "ask the user for a GPU", not break a deploy.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(
            transport=transport, timeout=_TIMEOUT, follow_redirects=True
        ) as client:
            config = await _json(client, f"{_HF}/{hf_repo}/resolve/main/config.json", headers)
            if config is None:
                _log.info("modelinfo: no readable config.json", extra={"hf_repo": hf_repo})
                return None
            weight_bytes = await _weight_bytes(client, hf_repo, headers)
    except httpx.HTTPError as exc:
        _log.info("modelinfo: hub unreachable", extra={"hf_repo": hf_repo, "error": str(exc)})
        return None

    architectures = config.get("architectures") or []
    quant = config.get("quantization_config") or {}
    return ModelProfile(
        hf_repo=hf_repo,
        architecture=architectures[0] if architectures else None,
        weight_bytes=weight_bytes,
        context_length=config.get("max_position_embeddings"),
        # HF is mid-rename from torch_dtype to dtype; accept either.
        precision=config.get("torch_dtype") or config.get("dtype"),
        quantization=quant.get("quant_method") if isinstance(quant, dict) else None,
        num_layers=config.get("num_hidden_layers"),
        moe_experts=config.get("num_experts") or config.get("num_local_experts"),
        moe_experts_per_token=config.get("num_experts_per_tok"),
    )


def required_vram_gb(profile: ModelProfile, *, utilization: float = 0.90) -> float | None:
    """Total VRAM a deploy of this model needs, or ``None`` when the weight size is unknown.

    ``(weights + KV floor) / utilization``. Deliberately simple: the serving engine already fits
    everything inside its utilization fraction, so the only judgement here is how much room to leave
    for KV cache. Validated against every hand-checked catalog entry (see tests).
    """
    if not profile.weight_bytes:
        return None
    weights_gb = profile.weight_bytes / _BYTES_PER_GB
    return round((weights_gb + _KV_FLOOR_GB) / utilization, 1)


def select_gpu(
    profile: ModelProfile,
    gpu_types: list[GPUType],
    *,
    utilization: float = 0.90,
    max_gpus: int = 8,
) -> tuple[GPUType, int] | None:
    """The GPU shape to run this model on, or ``None`` when it cannot be sized.

    Fewest GPUs first, then cheapest among those. NOT globally cheapest per hour, which sounds right
    for a cost tool and is wrong: five RTX A4000s are cheaper than one A100 and cannot serve a 32B
    model. Sharding costs real throughput, needs interconnect the cheap cards lack, and this rule
    reproduces every hand-validated catalog choice while a cost-first rule reproduced none of them.

    Counts are powers of two because tensor parallelism has to divide the attention heads evenly;
    a 3- or 5-way shard is not a configuration vLLM will accept.
    """
    needed = required_vram_gb(profile, utilization=utilization)
    if needed is None:
        return None
    counts = [n for n in (1, 2, 4, 8) if n <= max_gpus]
    options = [
        (count, gpu.hourly_usd * count, gpu)
        for gpu in gpu_types
        for count in counts
        if gpu.memory_gb * count >= needed
    ]
    if not options:
        return None
    count, _, gpu = min(options, key=lambda o: (o[0], o[1]))
    return gpu, count


async def _json(client: httpx.AsyncClient, url: str, headers: dict[str, str]) -> dict | None:
    resp = await client.get(url, headers=headers)
    if resp.status_code != 200:
        return None
    try:
        body = resp.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None


async def _weight_bytes(
    client: httpx.AsyncClient, hf_repo: str, headers: dict[str, str]
) -> int | None:
    """Total safetensors bytes. Sharded repos publish the sum directly; single-shard repos have no
    index, so fall back to the file tree. Both are metadata reads, never weights."""
    index_url = f"{_HF}/{hf_repo}/resolve/main/model.safetensors.index.json"
    index = await _json(client, index_url, headers)
    total = (index or {}).get("metadata", {}).get("total_size")
    if isinstance(total, int) and total > 0:
        return total

    resp = await client.get(f"{_HF}/api/models/{hf_repo}/tree/main", headers=headers)
    if resp.status_code != 200:
        return None
    try:
        entries = resp.json()
    except ValueError:
        return None
    if not isinstance(entries, list):
        return None
    summed = sum(
        int(e.get("size") or 0)
        for e in entries
        if isinstance(e, dict) and str(e.get("path", "")).endswith(".safetensors")
    )
    return summed or None
