"""Step-7/#3 process helpers: pidfile liveness + stale cleanup (the guard behind the daemon
lifecycle). Spawn/detach itself launches real processes and is smoke-tested manually, not here."""

from __future__ import annotations

import os

from gpu_orchestrator.cli import process


def test_running_pid_none_when_absent(tmp_path):
    assert process.running_pid(tmp_path / "missing.pid") is None


def test_write_read_clear_own_pid(tmp_path):
    pid_file = tmp_path / "d.pid"
    process.write_pid(pid_file)  # writes this (alive) process's pid
    assert process.running_pid(pid_file) == os.getpid()
    process.clear_pid(pid_file)
    assert process.running_pid(pid_file) is None


def test_running_pid_cleans_stale(tmp_path):
    pid_file = tmp_path / "d.pid"
    pid_file.write_text("999999")  # a pid that does not exist
    assert process.running_pid(pid_file) is None
    assert not pid_file.exists()  # stale file removed


def test_running_pid_ignores_garbage(tmp_path):
    pid_file = tmp_path / "d.pid"
    pid_file.write_text("not-a-pid")
    assert process.running_pid(pid_file) is None
