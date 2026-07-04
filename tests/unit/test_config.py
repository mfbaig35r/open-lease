"""Step-1 config tests: precedence, namespace isolation, and secret masking (spec §12, §7.5)."""

from __future__ import annotations

from gpu_orchestrator.config import Config, _default_namespace


def test_default_namespace_is_sanitized(monkeypatch):
    import gpu_orchestrator.config as cfg

    monkeypatch.setattr(cfg.socket, "gethostname", lambda: "Fahads-MBP.local")
    assert _default_namespace() == "fahads-mbp"


def test_instance_name_and_prefix():
    c = Config(namespace="laptop")
    assert c.instance_name("dep-a1b2c3") == "gpu-orch-laptop-dep-a1b2c3"
    assert c.instance_prefix() == "gpu-orch-laptop-"
    # The name must start with the prefix the sweep filters on (spec §7.5).
    assert c.instance_name("dep-a1b2c3").startswith(c.instance_prefix())


def test_init_kwargs_beat_env(monkeypatch):
    # CLI flags (init kwargs) are highest precedence.
    monkeypatch.setenv("GPU_ORCH_NAMESPACE", "from-env")
    c = Config(namespace="from-flag")
    assert c.namespace == "from-flag"


def test_env_beats_default(monkeypatch):
    monkeypatch.setenv("GPU_ORCH_NAMESPACE", "from-env")
    assert Config().namespace == "from-env"


def test_unprefixed_secret_env_names(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value")
    c = Config()
    assert c.runpod_api_key is not None
    assert c.runpod_api_key.get_secret_value() == "secret-value"


def test_secrets_never_appear_in_repr_or_effective(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "secret-value")
    c = Config()
    assert "secret-value" not in repr(c)
    eff = c.effective()
    assert eff["runpod_api_key"] == "***set***"
    assert "secret-value" not in str(eff)


def test_effective_masks_unset_secret_as_none():
    c = Config(namespace="x")
    # With no key set, effective() reports None rather than a fake mask.
    if c.runpod_api_key is None:
        assert c.effective()["runpod_api_key"] is None


def test_timeout_and_loop_defaults_present():
    c = Config()
    assert c.reconcile_interval == 10
    assert c.timeout_download == 1800
    assert c.retry_max_attempts == 3
    assert (c.proxy_host, c.proxy_port) == ("localhost", 8080)
