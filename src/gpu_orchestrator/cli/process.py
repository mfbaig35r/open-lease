"""Pidfile-based process management for the long-running daemon and proxy (CLI concern, not core).

A foreground `gpu daemon` / `gpu proxy` owns its pidfile: it writes it on start and clears it on
exit. The detached (`--detach`) and `gpu up` paths just spawn that same foreground command in a new
session and let the child self-register. ``running_pid`` is the single source of truth for "is it
up?", and it cleans up stale pidfiles (a pid that is no longer alive) so a crashed process never
looks running.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# The command that runs a foreground loop, invoked for detached spawns. `-m` keeps it tied to the
# same interpreter/venv rather than depending on the `gpu` script being on PATH.
_MODULE = "gpu_orchestrator.cli.main"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but owned by another user
    return True


def running_pid(pid_file: Path) -> int | None:
    """The live pid recorded in ``pid_file``, or None. Removes the file if the pid is stale."""
    try:
        pid = int(pid_file.read_text().strip())
    except (OSError, ValueError):
        return None
    if _alive(pid):
        return pid
    pid_file.unlink(missing_ok=True)  # stale (crashed without cleanup)
    return None


def write_pid(pid_file: Path) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))


def clear_pid(pid_file: Path) -> None:
    pid_file.unlink(missing_ok=True)


def spawn_detached(command: list[str], log_file: Path) -> int:
    """Launch ``gpu <command...>`` in a new session, output appended to ``log_file``. Returns the
    child pid. The child (a foreground command) writes its own pidfile."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "a")  # noqa: SIM115 -- handed to the child; closed when it exits
    proc = subprocess.Popen(
        [sys.executable, "-m", _MODULE, *command],
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc.pid


def stop(pid_file: Path, *, timeout: float = 5.0) -> int | None:
    """SIGTERM the process in ``pid_file`` and wait for it to exit. Returns the pid stopped, or None
    if nothing was running. Clears the pidfile."""
    pid = running_pid(pid_file)
    if pid is None:
        return None
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            break
        time.sleep(0.1)
    clear_pid(pid_file)
    return pid
