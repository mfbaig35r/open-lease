"""Token-usage metering: parse usage from a forwarded response, and derive the utilization +
cost-per-token view (spec §11 extension)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from gpu_orchestrator.core import usage
from gpu_orchestrator.models import CostRecord, DeploymentState
from gpu_orchestrator.store import Store
from tests.fixtures.deployments import make_deployment

_T0 = datetime(2026, 7, 4, 12, 0, 0, tzinfo=UTC)


def test_extract_usage_from_json():
    body = b'{"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":60,"total_tokens":100}}'
    assert usage.extract_usage(body) == (40, 60)


def test_extract_usage_from_sse_final_chunk():
    body = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'
        b"data: [DONE]\n\n"
    )
    assert usage.extract_usage(body) == (10, 5)


def test_extract_usage_absent_returns_none():
    assert usage.extract_usage(b'{"error":{"message":"bad model"}}') is None
    assert usage.extract_usage(b"") is None


def test_record_and_totals_round_trip(tmp_path):
    store = Store(tmp_path / "u.db")
    usage.record(store, "dep-1", b'{"usage":{"prompt_tokens":40,"completion_tokens":60}}', _T0)
    usage.record(store, "dep-1", b'{"usage":{"prompt_tokens":10,"completion_tokens":20}}', _T0)
    usage.record(store, "dep-1", b'{"no":"usage here"}', _T0)  # no-op: nothing to meter
    assert store.get_usage_totals("dep-1") == (2, 50, 80)  # requests, prompt sum, completion sum


def test_summary_computes_cost_per_mtok_and_utilization(tmp_path):
    store = Store(tmp_path / "u.db")
    dep = make_deployment(DeploymentState.READY)
    # 1,000,000 tokens over a closed 1-hour $3.00/hr rental -> $3.00 / M tok, ~278 tok/s.
    store.save_usage_record(dep.id, 400_000, 600_000, _T0)
    store.save_cost_record(
        CostRecord(
            deployment_id=dep.id,
            gpu_hourly_usd=3.0,
            started_at=_T0,
            stopped_at=_T0 + timedelta(hours=1),
        )
    )
    s = usage.summary(store, dep, now=_T0 + timedelta(hours=1))
    assert s.total_tokens == 1_000_000
    assert s.accrued_usd == 3.0
    assert s.cost_per_mtok == 3.0
    assert s.uptime_seconds == 3600.0
    assert s.tokens_per_sec == round(1_000_000 / 3600, 1)


def test_summary_no_traffic_is_zero(tmp_path):
    store = Store(tmp_path / "u.db")
    dep = make_deployment(DeploymentState.READY)
    s = usage.summary(store, dep, now=_T0)
    assert s.total_tokens == 0
    assert s.cost_per_mtok is None  # no divide-by-zero
    assert s.tokens_per_sec == 0.0
