"""Per-deployment concurrency gate (Tier A3): admit up to the cap, queue up to the bound, reject
(429) beyond it, time out a wait that never gets a slot, and reuse a slot once released."""

from __future__ import annotations

import asyncio

from gpu_orchestrator.proxy.limiter import DeploymentLimiter


async def test_rejects_immediately_when_no_queue_and_full():
    limiter = DeploymentLimiter(max_concurrency=1, max_queue=0, timeout_s=1.0)
    assert await limiter.acquire() is True  # slot taken
    assert await limiter.acquire() is False  # full, no queue -> reject at once
    limiter.release()
    assert await limiter.acquire() is True  # slot reusable after release


async def test_queued_waiter_admitted_when_slot_frees():
    limiter = DeploymentLimiter(max_concurrency=1, max_queue=1, timeout_s=1.0)
    assert await limiter.acquire() is True

    waiter = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0.02)  # let the waiter enter the (now full) queue
    assert await limiter.acquire() is False  # queue full -> reject

    limiter.release()  # frees the slot the waiter is queued for
    assert await waiter is True


async def test_wait_times_out_when_no_slot_frees():
    limiter = DeploymentLimiter(max_concurrency=1, max_queue=5, timeout_s=0.05)
    assert await limiter.acquire() is True
    assert await limiter.acquire() is False  # waited out the timeout, no slot -> 429
