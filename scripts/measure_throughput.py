"""Measure a catalog model's throughput on real hardware and emit the TOML to paste back.

This is the treadmill tool. Growing the catalog means launching every entry anyway (a profile
without ValidationMetadata does not ship, spec §14), and this makes that same run also answer "how
fast", which is what lets ``gpu estimate`` report cost per million tokens on a fresh install.

    uv run python scripts/measure_throughput.py qwen3-0.6b
    uv run python scripts/measure_throughput.py qwen3-8b --concurrency 16 --provider runpod

It deploys, waits for READY, drives two load patterns, prints a TOML block, and tears the pod down
in a finally. Two numbers, not one: single-stream decode is what a user feels, aggregate decode
under load is what a batch workload gets, and on real hardware they differ by an order of magnitude
(docs/adr-adaptive-execution-planning.md). Reporting one figure would be a lie either way.

Costs real money. A small model on a cheap GPU is a few cents.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from datetime import date

import httpx

from gpu_orchestrator.config import Config
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.models import DeploymentState

_PROMPT = "Write a short paragraph about the history of the printing press."
_MAX_TOKENS = 128


async def _one(client: httpx.AsyncClient, url: str, model: str) -> int:
    """Run one completion, return the output tokens the server reports (never our own estimate)."""
    resp = await client.post(
        f"{url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": _PROMPT}],
            "max_tokens": _MAX_TOKENS,
            "temperature": 0.0,
        },
        timeout=180,
    )
    resp.raise_for_status()
    return int(resp.json().get("usage", {}).get("completion_tokens", 0))


async def _measure(url: str, model: str, *, concurrency: int, rounds: int) -> float:
    """Aggregate output tokens/sec at a given concurrency, averaged over ``rounds`` waves."""
    async with httpx.AsyncClient() as client:
        await _one(client, url, model)  # warm: first request pays JIT and cache costs
        start = time.perf_counter()
        tokens = 0
        for _ in range(rounds):
            got = await asyncio.gather(*(_one(client, url, model) for _ in range(concurrency)))
            tokens += sum(got)
        return round(tokens / (time.perf_counter() - start), 1)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id", help="Catalog model id, e.g. qwen3-0.6b")
    ap.add_argument("--provider", default="runpod")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    orch = Orchestrator(Config())
    profile = orch._catalog.get_profile(args.model_id)
    gpu = profile.recommended_gpu
    print(f"deploying {args.model_id} on {gpu} ({args.provider})...", flush=True)

    deployment = None
    try:
        deployment = await orch.deploy_model(args.model_id, provider=args.provider, wait=True)
        if deployment.observed_state is not DeploymentState.READY:
            print(f"FAILED to reach READY: {deployment.observed_state.value}")
            return 1
        url = deployment.endpoint_url
        served = deployment.hf_repo or args.model_id
        print(f"READY at {url}; measuring...", flush=True)

        single = await _measure(url, served, concurrency=1, rounds=args.rounds)
        print(f"  single stream        {single:8.1f} tok/s", flush=True)
        concurrent = await _measure(url, served, concurrency=args.concurrency, rounds=args.rounds)
        print(f"  concurrency {args.concurrency:<3d}      {concurrent:8.1f} tok/s", flush=True)

        print("\n--- paste into the entry's [.profile.validation] block ---")
        print(f'throughput_measured_at = "{date.today().isoformat()}"')
        print(f'throughput_gpu = "{gpu}"')  # not necessarily validated_gpu
        print(f"tokens_per_sec = {single}")
        print(f"tokens_per_sec_concurrent = {concurrent}")
        print(f"measured_concurrency = {args.concurrency}")
        return 0
    finally:
        if deployment is not None:
            # stop, not delete: cost safety needs the POD gone, and stop does that while keeping
            # the deployment record. A failed run is exactly when you want the state history and
            # events, and deleting here threw them away the first time this hit a provider error.
            await orch.stop_deployment(deployment.id)
            print(f"\nstopped {deployment.id} (record kept for diagnosis)", flush=True)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
