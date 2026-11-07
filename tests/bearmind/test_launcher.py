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
    assert flags[1] & subprocess.DETACHED_PROCESS


def test_request_stop_writes_file(tmp_path):
    paths = SeriesPaths(tmp_path)
    launcher.request_stop(paths)
    assert paths.stop_request.is_file()
