"""Per-deployment concurrency gate for the OpenAI proxy (capacity plan, Tier A3).

A ``DeploymentLimiter`` admits at most ``max_concurrency`` requests at once; up to ``max_queue``
more may wait up to ``timeout_s`` for a slot; anything beyond is rejected so the proxy returns 429.
A slot is held for the whole request including the streamed response body, and released exactly once
when the stream ends (or the client disconnects), so slots never leak.

This is live infrastructure, not the pure decision core, so it drives the event loop directly. It
relies on ``asyncio.Semaphore`` being cancellation-safe (CPython 3.11+): an ``acquire`` cancelled by
the wait timeout hands its slot to the next waiter rather than leaking it.
"""

from __future__ import annotations

import asyncio


class DeploymentLimiter:
    def __init__(self, max_concurrency: int, max_queue: int, timeout_s: float) -> None:
        self._sem = asyncio.Semaphore(max_concurrency)
        self._max_queue = max_queue
        self._timeout_s = timeout_s
        self._waiting = 0  # requests currently blocked waiting for a slot

    async def acquire(self) -> bool:
        """Take a slot, waiting in the bounded queue if all slots are busy. True if admitted, False
        if the queue is full or the wait timed out (caller then returns 429). The check-and-count
        below has no ``await`` between the read and the increment, so it is atomic under the
        single-threaded event loop and the queue bound holds without a lock."""
        if self._sem.locked() and self._waiting >= self._max_queue:
            return False  # every slot busy and the queue is already full
        self._waiting += 1
        try:
            await asyncio.wait_for(self._sem.acquire(), self._timeout_s)
            return True
        except TimeoutError:  # asyncio.TimeoutError is an alias for the builtin in 3.11+
            return False
        finally:
            self._waiting -= 1

    def release(self) -> None:
        self._sem.release()


def get_limiter(
    limiters: dict[str, DeploymentLimiter],
    deployment_id: str,
    max_concurrency: int,
    max_queue: int,
    timeout_s: float,
) -> DeploymentLimiter:
    """Get or create the limiter for a deployment. Created once per deployment id from the config in
    force at first use; changing a deployment's limits takes effect after a proxy restart."""
    limiter = limiters.get(deployment_id)
    if limiter is None:
        limiter = DeploymentLimiter(max_concurrency, max_queue, timeout_s)
        limiters[deployment_id] = limiter
    return limiter
