"""Detached start of the runner and the stop request."""
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from bearmind import launcher, monitor
from bearmind.paths import SeriesPaths

REPO = Path(__file__).resolve().parents[2]


def _wait_for(path, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if path.exists() and path.stat().st_size:
            return True
        time.sleep(0.1)
    return False


def _wait_exit(pid, timeout=30):
    try:
        psutil.Process(pid).wait(timeout)
    except psutil.NoSuchProcess:
        pass


def test_runner_command_uses_python_not_pythonw(monkeypatch, tmp_path):
    exe = tmp_path / "env" / "pythonw.exe"
    monkeypatch.setattr(sys, "executable", str(exe))
    cmd = launcher.runner_command(tmp_path / "S")
    assert cmd == [str(exe.with_name("python.exe")), "-m", "bearmind", "run", str(tmp_path / "S")]


def test_runner_command_keeps_python(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert launcher.runner_command(tmp_path)[0] == str(tmp_path / "python.exe")


def test_start_detached_survives_parent(tmp_path):
    root, marker = tmp_path / "series", tmp_path / "marker.txt"
    child = f"import time; time.sleep(2); open({str(marker)!r}, 'w').write('alive')"
    helper = ("import sys; from pathlib import Path; from bearmind import launcher; "
              "print(launcher.start_detached(Path(sys.argv[1]), cmd=[sys.executable, '-c', sys.argv[2]]))")
    out = subprocess.run([sys.executable, "-c", helper, str(root), child], cwd=REPO, capture_output=True,
                         text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    pid = int(out.stdout.strip().splitlines()[-1])
    try:
        assert not marker.exists()  # the helper returned at once, the child is still sleeping
        assert _wait_for(marker), "the detached child died with its parent"
    finally:
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    assert (root / "_bearmind" / "logs" / "runner_console.log").is_file()


def test_start_detached_refuses_when_alive(tmp_path, monkeypatch):
    state = monitor.RunnerState(True, "running", 4242, 1.0, None, None)
    monkeypatch.setattr(monitor, "runner_state", lambda paths: state)
    called = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: called.append(a))
    with pytest.raises(RuntimeError, match="batch already running: PID 4242"):
        launcher.start_detached(tmp_path)
    assert called == []


def test_cyrillic_and_space_path_passed_intact(tmp_path):
    root = tmp_path / "серия тест & co"
    root.mkdir()
    out = tmp_path / "argv.json"
    code = "import sys, json; open(sys.argv[1], 'w', encoding='utf-8').write(json.dumps(sys.argv[2]))"
    pid = launcher.start_detached(root, cmd=[sys.executable, "-c", code, str(out), str(root)])
    try:
        assert _wait_for(out)
        _wait_exit(pid)
    finally:
        try:
            psutil.Process(pid).kill()
        except psutil.NoSuchProcess:
            pass
    assert json.loads(out.read_text(encoding="utf-8")) == str(root)


def test_start_detached_retries_without_breakaway(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("Windows job objects")
    real, flags = subprocess.Popen, []

    def popen(cmd, **kw):
        flags.append(kw["creationflags"])
        if kw["creationflags"] & subprocess.CREATE_BREAKAWAY_FROM_JOB:
            raise PermissionError("job forbids breakaway")
        return real(cmd, **kw)

    monkeypatch.setattr(subprocess, "Popen", popen)
    pid = launcher.start_detached(tmp_path, cmd=[sys.executable, "-c", "pass"])
    _wait_exit(pid)
    assert len(flags) == 2 and not flags[1] & subprocess.CREATE_BREAKAWAY_FROM_JOB
    assert flags[1] & subprocess.CREATE_NO_WINDOW and not flags[1] & subprocess.DETACHED_PROCESS


def test_request_stop_writes_file(tmp_path):
    paths = SeriesPaths(tmp_path)
    launcher.request_stop(paths)
    assert paths.stop_request.is_file()


def test_detached_child_sees_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENBLAS_CORETYPE", raising=False)
    out = tmp_path / "env.json"
    code = ("import os, sys, json; open(sys.argv[1], 'w', encoding='utf-8').write(json.dumps("
            "{k: os.environ.get(k) for k in ('PYTHONPATH', 'OPENBLAS_CORETYPE', 'PYTHONUTF8')} | {'cwd': os.getcwd()}))")
    pid = launcher.start_detached(tmp_path / "s", cmd=[sys.executable, "-c", code, str(out)])
    try:
        assert _wait_for(out)
        _wait_exit(pid)
    finally:
        _kill(pid)
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got == {"PYTHONPATH": str(REPO), "OPENBLAS_CORETYPE": "Sandybridge", "PYTHONUTF8": "1",
                   "cwd": str(REPO)}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows consoles")
def test_children_of_detached_runner_get_no_visible_console(tmp_path):
    """DETACHED_PROCESS would give every console child of the runner (stage processes, pool workers) a new\n    visible window; the runner has a hidden console instead, which they inherit."""
    out = tmp_path / "console.json"
    child = (
        "import ctypes, json, sys\n"
        "def f(path):\n"
        "    k, u = ctypes.windll.kernel32, ctypes.windll.user32\n"
        "    k.GetConsoleWindow.restype = ctypes.c_void_p\n"
        "    u.IsWindowVisible.argtypes = [ctypes.c_void_p]\n"
        "    h = k.GetConsoleWindow()\n"
        "    open(path, 'w').write(json.dumps({'hwnd': h or 0, 'visible': bool(h and u.IsWindowVisible(h))}))\n"
        "if __name__ == '__main__':\n"
        "    import multiprocessing as mp\n"
        "    p = mp.get_context('spawn').Process(target=f, args=(sys.argv[1],))\n"
        "    p.start(); p.join()\n")
    script = tmp_path / "parent.py"
    script.write_text(child, encoding="utf-8")
    pid = launcher.start_detached(tmp_path / "s", cmd=[sys.executable, str(script), str(out)])
    try:
        assert _wait_for(out, 60)
        _wait_exit(pid, 60)
    finally:
        _kill(pid)
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["visible"] is False, got


def _kill(pid):
    try:
        psutil.Process(pid).kill()
    except psutil.NoSuchProcess:
        pass


def test_real_start_command_smoke(tmp_path):
    """The command the Start button runs, on a series whose only session is excluded: the real runner starts,
    finds nothing to do and exits; its console log is UTF-8 with the runner's own summary line, the final
    heartbeat carries the exit reason and the lock is gone. Cyrillic and a space in the path."""
    from bearmind import registry
    from bearmind.registry import STAGES, SessionRecord, StageRecord

    paths = SeriesPaths(tmp_path / "серия тест")
    registry.save_record(paths, SessionRecord(
        sid="LT1_J01_1D", mouse="LT1_J01", mode="bonsai", parts=[], n_frames=100, fps=20.0, fps_info={},
        anomalies=[], decisions=[], excluded=True, stages={s: StageRecord() for s in STAGES}))
    pid = launcher.start_detached(paths.root)  # default runner_command: python -m bearmind run SERIES
    try:
        psutil.Process(pid).wait(120)
    except psutil.NoSuchProcess:
        pass
    finally:
        _kill(pid)
    text = (paths.logs_dir / "runner_console.log").read_bytes().decode("utf-8")  # strict: UTF-8 or it raises
    assert "this run: done 0, failed 0, skipped 0" in text and "excluded 1" in text, text
    assert json.loads(paths.heartbeat.read_text(encoding="utf-8"))["exit"] == "finished"
    assert not paths.lock.exists()
    assert monitor.runner_state(paths).status == "stopped"
