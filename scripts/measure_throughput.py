"""Measure a catalog model's throughput on real hardware and emit the TOML to paste back.

This is the treadmill tool. Growing the catalog means launching every entry anyway (a profile
without ValidationMetadata does not ship, spec §14), and this makes that same run also answer "how
fast", which is what lets ``gpu estimate`` report cost per million tokens on a fresh install.

    uv run python scripts/measure_throughput.py qwen3-0.6b
    uv run python scripts/measure_throughput.py qwen3-8b --concurrency 16 --provider runpod

Needs a daemon (`gpu daemon --detach`). It deploys NON-blocking and polls, because the daemon owns
the reconcile loop: driving reconcile_once inline here while a daemon ticks the same deployment
would be two reconcilers on one record, and the per-deployment lock is single-process-only in
Phase 1 (CLAUDE.md). Running a daemon is also what makes tick_sweep reap the pod if this script
dies mid-run.

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


async def _await_ready(orch: Orchestrator, deployment_id: str, timeout: int):
    """Poll the store while the daemon reconciles. Prints stage changes so a long cold start is
    visibly progressing rather than looking hung."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        deployment = orch.get_deployment(deployment_id)
        state = deployment.observed_state
        if state is not last:
            pct = deployment.download_progress
            extra = f" ({int(pct * 100)}%)" if pct is not None else ""
            print(f"  {state.value}{extra}", flush=True)
            last = state
        if state in (DeploymentState.READY, DeploymentState.FAILED):
            return deployment
        await asyncio.sleep(10)
    print(f"  gave up waiting after {timeout}s")
    return orch.get_deployment(deployment_id)


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model_id", nargs="?", help="Catalog model id, e.g. qwen3-0.6b")
    ap.add_argument("--hf-repo", help="Measure an uncatalogued repo instead of a catalog id.")
    ap.add_argument("--gpu", help="Ad-hoc: GPU to run on.")
    ap.add_argument("--gpus", type=int, default=1, help="Ad-hoc: GPUs per pod (tensor parallel).")
    ap.add_argument("--context", type=int, default=0, help="Ad-hoc: max model length.")
    ap.add_argument("--provider", default="runpod")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=3000, help="Seconds to wait for READY.")
    args = ap.parse_args()

    orch = Orchestrator(Config())
    if args.hf_repo:
        gpu, label = args.gpu, args.hf_repo
    else:
        gpu, label = orch._catalog.get_profile(args.model_id).recommended_gpu, args.model_id
    shape = f"{args.gpus}x {gpu}" if args.gpus > 1 else gpu
    print(f"deploying {label} on {shape} ({args.provider})...", flush=True)

    deployment = None
    try:
        if args.hf_repo:
            # Measure BEFORE cataloguing: a profile only earns a catalog entry once someone has
            # actually launched it (spec §14), so the ad-hoc path is how a candidate is tried.
            deployment = await orch.deploy_adhoc(
                hf_repo=args.hf_repo,
                gpu=args.gpu,
                gpu_count=args.gpus,
                context_window=args.context,
                provider=args.provider,
                wait=False,
            )
        else:
            deployment = await orch.deploy_model(args.model_id, provider=args.provider, wait=False)
        deployment = await _await_ready(orch, deployment.id, args.timeout)
        if deployment.observed_state is not DeploymentState.READY:
            print(f"FAILED to reach READY: {deployment.observed_state.value}")
            if deployment.failure:
                print(f"  {deployment.failure.message[:200]}")
            return 1
        url = deployment.endpoint_url
        served = deployment.hf_repo or args.model_id
        # The GPU it ACTUALLY got, which is not always the one the profile recommends: an
        # out-of-stock recommendation is substituted at create time. Read it here, while the
        # instance still exists, because recording the recommendation instead would attribute the
        # measurement to hardware it never ran on.
        # instance.gpu_type is the PROVIDER SKU ("NVIDIA H100 80GB HBM3"); the catalog is written
        # in catalog ids ("H100-80GB"). Both match at lookup time, but keep the file consistent.
        raw = deployment.instance.gpu_type if deployment.instance else ""
        caps = await orch._provider(args.provider).capabilities()
        actual_gpu = next(
            (g.id for g in caps.gpu_types if raw in (g.id, g.provider_sku)), raw or gpu
        )
        if actual_gpu != gpu:
            print(f"  note: substituted {gpu} -> {actual_gpu} (recommended GPU was out of stock)")
        print(f"READY at {url} on {actual_gpu}; measuring...", flush=True)

        single = await _measure(url, served, concurrency=1, rounds=args.rounds)
        print(f"  single stream        {single:8.1f} tok/s", flush=True)
        concurrent = await _measure(url, served, concurrency=args.concurrency, rounds=args.rounds)
        print(f"  concurrency {args.concurrency:<3d}      {concurrent:8.1f} tok/s", flush=True)

        print("\n--- paste into the entry's [.profile.validation] block ---")
        print(f'throughput_measured_at = "{date.today().isoformat()}"')
        print(f'throughput_gpu = "{actual_gpu}"')  # what it ran on, not what was recommended
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
