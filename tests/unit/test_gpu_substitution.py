"""Availability-aware GPU selection: a recommendation is a preference, not a pin.

Driven by an observed failure, not a hypothetical. On 2026-08-09 a routine catalog measurement of
qwen3-32b failed with a RunPod 500 because A100 capacity was zero in all four data centers, while an
A40 and an H100 sat available. The model would have run on either.

The decision is pure (``outcomes.substitute_gpu``) so the money-affecting rule is exhaustively
testable without a provider.
"""

from __future__ import annotations

from gpu_orchestrator.core import outcomes
from gpu_orchestrator.models import GPUType, RuntimeProfile, ValidationMetadata

_A40 = GPUType(id="A40-48GB", name="A40", memory_gb=48, hourly_usd=0.44, provider_sku="NVIDIA A40")
_A100 = GPUType(
    id="A100-80GB", name="A100", memory_gb=80, hourly_usd=1.89, provider_sku="NVIDIA A100 80GB PCIe"
)
_H100 = GPUType(
    id="H100-80GB", name="H100", memory_gb=80, hourly_usd=2.99, provider_sku="NVIDIA H100 80GB HBM3"
)
_A4000 = GPUType(
    id="RTX-A4000", name="A4000", memory_gb=16, hourly_usd=0.17, provider_sku="NVIDIA RTX A4000"
)
_MENU = [_A4000, _A40, _A100, _H100]


def test_in_stock_recommendation_is_left_alone():
    assert outcomes.substitute_gpu(_A100, _MENU, {_A100.provider_sku, _H100.provider_sku}) is None


def test_out_of_stock_recommendation_is_swapped_for_an_equal_or_larger_card():
    """The exact case that failed: A100 gone, H100 available, same 80 GB."""
    chosen = outcomes.substitute_gpu(_A100, _MENU, {_H100.provider_sku, _A40.provider_sku})
    assert chosen == _H100


def test_substitution_never_downsizes():
    """A smaller card may be cheaper and in stock and still cannot hold the model. Sizing was done
    against the recommendation, so a substitute must be at least as large to stay safe without
    re-consulting the model."""
    chosen = outcomes.substitute_gpu(_A100, _MENU, {_A40.provider_sku, _A4000.provider_sku})
    assert chosen is None


def test_substitution_picks_the_cheapest_sufficient_card():
    chosen = outcomes.substitute_gpu(_A40, _MENU, {_A100.provider_sku, _H100.provider_sku})
    assert chosen == _A100  # $1.89 beats the H100's $2.99, both are >= 48 GB


def test_no_availability_data_means_unknown_not_empty():
    """A provider that reports nothing must not be read as "everything is out of stock". Absence of
    data is not evidence of absence, and substituting here would move a deploy off its validated
    hardware for no reason."""
    assert outcomes.substitute_gpu(_A100, _MENU, set()) is None


def test_nothing_available_at_all_keeps_the_recommendation():
    """When no candidate qualifies, keep the recommendation and let the deploy fail honestly on the
    hardware it was validated for, rather than failing somewhere unexpected."""
    assert outcomes.substitute_gpu(_A100, _MENU, {"some-unknown-sku"}) is None


def test_a_pinned_gpu_is_a_decision_not_a_recommendation():
    """A human who typed --gpu gets that GPU or an error. Silently running somewhere else is not a
    smaller surprise than failing."""
    profile = RuntimeProfile(
        model_id="m",
        image="img",
        recommended_gpu="A100-80GB",
        min_disk_gb=20,
        gpu_pinned=True,
        validation=ValidationMetadata(
            validated_at="2026-08-09",
            validated_provider="runpod",
            validated_gpu="A100-80GB",
            validated_image="img",
            startup_timeout_seconds=600,
        ),
    )
    assert profile.gpu_pinned is True


def test_recommendations_are_not_pinned_by_default():
    profile = RuntimeProfile(
        model_id="m",
        image="img",
        recommended_gpu="A100-80GB",
        min_disk_gb=20,
        validation=ValidationMetadata(
            validated_at="2026-08-09",
            validated_provider="runpod",
            validated_gpu="A100-80GB",
            validated_image="img",
            startup_timeout_seconds=600,
        ),
    )
    assert profile.gpu_pinned is False


# --- end to end through the reconciler ---------------------------------------------------


import httpx  # noqa: E402
import pytest  # noqa: E402

from gpu_orchestrator.config import Config  # noqa: E402
from gpu_orchestrator.core.catalog import Catalog  # noqa: E402
from gpu_orchestrator.core.reconciler import reconcile_once  # noqa: E402
from gpu_orchestrator.events import EventLog  # noqa: E402
from gpu_orchestrator.models import (  # noqa: E402
    Deployment,
    DeploymentState,
    EventKind,
    GpuAvailability,
)
from gpu_orchestrator.providers.mock import MockProvider  # noqa: E402
from gpu_orchestrator.runtimes.vllm import VLLMRuntime  # noqa: E402
from gpu_orchestrator.store import Store  # noqa: E402
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC  # noqa: E402


class _StockedProvider(MockProvider):
    """A mock that reports real availability, so the reconciler's swap can be observed."""

    def __init__(self, *, namespace: str, available: set[str]) -> None:
        super().__init__(namespace=namespace)
        self._available = available

    async def gpu_availability(self, gpu_type: str | None = None) -> list[GpuAvailability]:
        caps = await self.capabilities()
        return [
            GpuAvailability(
                data_center_id="DC-1",
                gpu_type_id=g.provider_sku,
                available=g.provider_sku in self._available,
            )
            for g in caps.gpu_types
        ]


def _ctx(tmp_path, provider, profile):
    store = Store(tmp_path / "sub.db")
    return {
        "provider": provider,
        "runtime": VLLMRuntime(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
        "catalog": Catalog({"qwen3-0.6b": QWEN3_06B_SPEC}, {"qwen3-0.6b": profile}),
        "config": Config(namespace="test", state_db=tmp_path / "sub.db"),
        "store": store,
        "events": EventLog(store),
    }


def _deployment(profile):
    return Deployment(
        id="dep-sub01",
        model_id="qwen3-0.6b",
        provider="mock",
        desired_state=DeploymentState.READY,
        observed_state=DeploymentState.REQUESTED,
        profile=profile,
    )


@pytest.mark.parametrize(
    ("pinned", "expected_gpu"),
    [(False, "A40-48GB"), (True, "RTX-A4000")],
    ids=["recommendation-is-substituted", "pin-is-honoured"],
)
async def test_reconciler_substitutes_only_an_unpinned_recommendation(
    tmp_path, pinned, expected_gpu
):
    """The 16 GB A4000 is out of stock; a 48 GB A40 and an 80 GB A100 are both available.

    An unpinned deploy moves to the A40: cheapest card that is still at least as large, not the
    largest available. A pinned one stays put and is left to fail on the hardware its operator
    chose, because silently running elsewhere is not a smaller surprise than failing.
    """
    profile = QWEN3_06B_PROFILE.model_copy(
        update={"recommended_gpu": "RTX-A4000", "gpu_pinned": pinned}
    )
    provider = _StockedProvider(namespace="test", available={"A100-80GB", "A40-48GB"})
    ctx = _ctx(tmp_path, provider, profile)
    dep = await reconcile_once(_deployment(profile), **ctx)
    assert dep.instance is not None
    assert dep.instance.gpu_type == expected_gpu


async def test_substitution_is_recorded_as_an_event(tmp_path):
    """Silently running somewhere other than the validated GPU would be invisible in the cost and
    incident record. The swap has to be auditable."""
    profile = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": "RTX-A4000"})
    provider = _StockedProvider(namespace="test", available={"A100-80GB"})
    ctx = _ctx(tmp_path, provider, profile)
    await reconcile_once(_deployment(profile), **ctx)
    kinds = [e.kind for e in ctx["store"].query_events(deployment_id="dep-sub01")]
    assert EventKind.GPU_SUBSTITUTED in kinds
