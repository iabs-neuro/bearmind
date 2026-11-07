"""Start the batch runner as a process of its own (it outlives the window that started it) and ask it to stop.

The runner is the same command a student types: python -m bearmind run SERIES (no --once: it waits for
people and picks their decisions up itself). The series folder goes to the child as one argv element, so
Cyrillic letters, spaces and '&' arrive intact. The child's stage processes make their own job object
(bearmind.isolate); the runner is not put into one here.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from bearmind import monitor, provenance
from bearmind.paths import SeriesPaths


def runner_command(series_root: Path) -> list[str]:
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe":  # the GUI runs under pythonw; the runner needs a real stdout/stderr
        exe = exe.with_name("python.exe")
    return [str(exe), "-m", "bearmind", "run", str(series_root)]


def _flags(breakaway: bool) -> int:
    if os.name != "nt":
        return 0
    # CREATE_NO_WINDOW, not DETACHED_PROCESS: a runner without a console gives every console child (stage
    # processes, CaImAn pool workers) a new visible window; with a hidden console they inherit it
    f = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    return f | subprocess.CREATE_BREAKAWAY_FROM_JOB if breakaway else f


def start_detached(series_root: Path, cmd: list[str] | None = None) -> int:
    """Start the runner of the series; returns its PID. Raises RuntimeError when one is running already."""
    paths = SeriesPaths(Path(series_root).resolve())
    state = monitor.runner_state(paths)
    if state.alive:
        raise RuntimeError(f"batch already running: PID {state.pid}")
    cmd = cmd or runner_command(paths.root)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(provenance.REPO_ROOT)
    env.setdefault("OPENBLAS_CORETYPE", "Sandybridge")
    env["PYTHONUTF8"] = "1"  # series names the redirected (cp1251) console log cannot encode
    with open(paths.logs_dir / "runner_console.log", "ab") as log:
        # a job that forbids breakaway refuses the flag (OSError): start without it, still detached
        for breakaway in (True, False):
            try:
                proc = subprocess.Popen(cmd, cwd=provenance.REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, close_fds=True, env=env,
                                        creationflags=_flags(breakaway))
                return proc.pid
            except OSError:
                if not breakaway or os.name != "nt":
                    raise
    raise AssertionError("unreachable")


def request_stop(paths) -> None:
    """Ask the runner to stop before its next stage (a running stage finishes first)."""
    paths.stop_request.parent.mkdir(parents=True, exist_ok=True)
    paths.stop_request.write_text("stop\n", encoding="utf-8")
