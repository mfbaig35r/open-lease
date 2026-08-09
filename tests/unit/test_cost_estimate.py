"""Cost estimation in both metrics: hourly rate and cost per million tokens (issue #25).

The rule these tests pin down is that per-token cost is *measured or absent*, never inferred. An
estimate that invents a tokens/sec figure looks authoritative and cannot be checked, which is worse
than saying nothing. See docs/adr-adaptive-execution-planning.md for why both metrics ship together:
the cheaper-per-hour configuration can be many times worse per token.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from gpu_orchestrator.config import Config
from gpu_orchestrator.core import usage
from gpu_orchestrator.core.catalog import Catalog
from gpu_orchestrator.core.orchestrator import Orchestrator
from gpu_orchestrator.errors import ModelNotFoundError
from gpu_orchestrator.models import CostEstimate, CostRecord, DeploymentState
from gpu_orchestrator.providers.mock import MockProvider
from tests.fixtures.catalog import QWEN3_06B_PROFILE, QWEN3_06B_SPEC
from tests.fixtures.deployments import make_deployment

_T0 = datetime(2026, 8, 9, 12, 0, 0, tzinfo=UTC)


def _orch(tmp_path) -> Orchestrator:
    return Orchestrator(
        Config(namespace="test", state_db=tmp_path / "est.db", reconcile_interval=0),
        catalog=Catalog({"qwen3-0.6b": QWEN3_06B_SPEC}, {"qwen3-0.6b": QWEN3_06B_PROFILE}),
        provider=MockProvider(namespace="test"),
    )


def _served(orch, *, deployment_id, model_id, gpu, tokens, seconds, started=_T0):
    """Persist a past deployment that actually served traffic, so throughput is measurable."""
    profile = QWEN3_06B_PROFILE.model_copy(update={"recommended_gpu": gpu, "model_id": model_id})
    dep = make_deployment(DeploymentState.STOPPED, deployment_id=deployment_id, profile=profile)
    dep = dep.model_copy(update={"model_id": model_id})
    orch._store.save_deployment(dep)
    orch._store.save_cost_record(
        CostRecord(
            deployment_id=dep.id,
            gpu_hourly_usd=0.50,
            started_at=started,
            stopped_at=started + timedelta(seconds=seconds),
        )
    )
    orch._store.save_usage_record(dep.id, tokens // 2, tokens - tokens // 2, started)
    return dep


# --- the absent-not-guessed rule -------------------------------------------------------


async def test_no_history_reports_hourly_but_no_cost_per_token(tmp_path):
    orch = _orch(tmp_path)
    est = await orch.estimate_cost("qwen3-0.6b", provider="mock", hours=2.0)
    assert est.estimated_usd == pytest.approx(est.gpu_hourly_usd * 2.0)
    assert est.cost_per_mtok is None
    assert est.observed_tokens_per_sec is None
    assert est.throughput_basis is None


def test_cost_per_mtok_is_none_rather_than_zero_when_throughput_unknown():
    est = CostEstimate(
        model_id="m",
        provider="mock",
        gpu_type="MOCK-GPU",
        gpu_hourly_usd=1.0,
        hours=1.0,
        estimated_usd=1.0,
    )
    assert est.cost_per_mtok is None
    assert "cost_per_mtok" in est.model_dump()  # still present in the contract, just null


# --- the arithmetic --------------------------------------------------------------------


def test_cost_per_mtok_arithmetic():
    # $7.14/hr at 33 tok/s is the ADR's resident DeepSeek-R1 row: $60.10 per million tokens.
    est = CostEstimate(
        model_id="deepseek-r1",
        provider="mock",
        gpu_type="MOCK-GPU",
        gpu_hourly_usd=7.14,
        hours=1.0,
        estimated_usd=7.14,
        observed_tokens_per_sec=33.0,
    )
    assert est.cost_per_mtok == pytest.approx(60.10, abs=0.01)


async def test_measured_history_fills_in_cost_per_token(tmp_path):
    orch = _orch(tmp_path)
    gpu = QWEN3_06B_PROFILE.recommended_gpu
    _served(orch, deployment_id="dep-aaa", model_id="qwen3-0.6b", gpu=gpu, tokens=3600, seconds=360)
    est = await orch.estimate_cost("qwen3-0.6b", provider="mock")
    assert est.observed_tokens_per_sec == pytest.approx(10.0)  # 3600 tokens / 360s
    assert est.cost_per_mtok == pytest.approx(est.gpu_hourly_usd / (10.0 * 3600) * 1e6, rel=1e-3)
    assert est.throughput_basis == "measured over 1 past deployment"


async def test_throughput_aggregates_across_past_deployments(tmp_path):
    orch = _orch(tmp_path)
    gpu = QWEN3_06B_PROFILE.recommended_gpu
    _served(orch, deployment_id="dep-aaa", model_id="qwen3-0.6b", gpu=gpu, tokens=1000, seconds=100)
    _served(orch, deployment_id="dep-bbb", model_id="qwen3-0.6b", gpu=gpu, tokens=3000, seconds=100)
    est = await orch.estimate_cost("qwen3-0.6b", provider="mock")
    assert est.observed_tokens_per_sec == pytest.approx(20.0)  # 4000 tokens over 200s
    assert est.throughput_basis == "measured over 2 past deployments"


# --- throughput is a property of (model, GPU), not of the model ------------------------


async def test_measurement_from_a_different_gpu_is_not_reused(tmp_path):
    orch = _orch(tmp_path)
    _served(
        orch,
        deployment_id="dep-aaa",
        model_id="qwen3-0.6b",
        gpu="A40-48GB",  # measured on a different GPU than the profile recommends
        tokens=3600,
        seconds=360,
    )
    est = await orch.estimate_cost("qwen3-0.6b", provider="mock")
    assert est.cost_per_mtok is None, "throughput on one GPU says nothing about another"


def test_rented_but_never_served_is_not_counted_as_zero_throughput(tmp_path):
    orch = _orch(tmp_path)
    dep = make_deployment(DeploymentState.STOPPED, deployment_id="dep-idle")
    orch._store.save_deployment(dep)
    orch._store.save_cost_record(
        CostRecord(
            deployment_id=dep.id,
            gpu_hourly_usd=0.50,
            started_at=_T0,
            stopped_at=_T0 + timedelta(hours=1),
        )
    )
    got = usage.observed_throughput(orch._store, dep.model_id)
    assert got is None, "an idle pod is missing data, not evidence of slowness"


# --- ad-hoc models: the gap that made estimate_cost unusable for half of deploys --------


async def test_adhoc_model_priced_from_an_explicit_gpu(tmp_path):
    orch = _orch(tmp_path)
    est = await orch.estimate_cost("not-in-catalog", provider="mock", gpu="A40-48GB")
    assert est.gpu_type == "A40-48GB"
    assert est.estimated_usd > 0


async def test_adhoc_model_priced_from_its_own_deploy_history(tmp_path):
    orch = _orch(tmp_path)
    _served(
        orch,
        deployment_id="dep-adh",
        model_id="qwen3-14b",  # never in the catalog
        gpu="A40-48GB",
        tokens=1800,
        seconds=180,
    )
    est = await orch.estimate_cost("qwen3-14b", provider="mock")
    assert est.gpu_type == "A40-48GB"
    assert est.observed_tokens_per_sec == pytest.approx(10.0)


async def test_explicit_gpu_overrides_the_catalog_recommendation(tmp_path):
    orch = _orch(tmp_path)
    est = await orch.estimate_cost("qwen3-0.6b", provider="mock", gpu="A40-48GB")
    assert est.gpu_type == "A40-48GB"


async def test_unknown_model_with_no_gpu_and_no_history_raises(tmp_path):
    orch = _orch(tmp_path)
    with pytest.raises(ModelNotFoundError, match="pass a gpu"):
        await orch.estimate_cost("never-heard-of-it", provider="mock")
