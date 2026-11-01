"""Stage isolation with real child processes: results, exceptions, crashes, a dead pool worker, Ctrl+C.

The fake stages are in isolate_fakes.py. Every test that starts processes checks that none of the child
tree outlives the call: a sampler records every descendant of this process while the stage runs.
"""
import _thread
import json
import os
import pickle
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from bearmind import isolate
from bearmind.paths import SeriesPaths
from bearmind.pool import POOL_WORKER_DIED
from bearmind.registry import Anomaly, SessionRecord
from bearmind.stages import StageResult, WaitingForInput

SID = "NOF_H01_1D"


def spec(name):
    return f"tests.bearmind.isolate_fakes:{name}"


@pytest.fixture
def stage(tmp_path):
    paths = SeriesPaths(tmp_path / "series")
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=10, fps=19.76, fps_info={},
                        anomalies=[], decisions=[], excluded=False, stages={})
    notes = tmp_path / "notes"
    return SimpleNamespace(paths=paths, rec=rec, cfg={"fake": {"dir": str(notes)}}, log=paths.session_log(SID),
                           notes=notes)


def run(s, name):
    return isolate.run_isolated(spec(name), s.paths, s.rec, s.cfg, s.log)


def log_text(s):
    return s.log.read_text(encoding="utf-8", errors="replace")


def tree(s):
    """[pid, creation time] of the processes the fake stage noted (it may start and lose them faster than
    the sampler below looks)."""
    return json.loads((s.notes / "tree.json").read_text(encoding="utf-8"))


def survivors(rows):
    out = []
    for pid, created in rows:
        try:
            if psutil.Process(pid).create_time() == created:
                out.append(pid)
        except psutil.NoSuchProcess:
            pass
    return out


class Descendants:
    """Every process started below this one while the block runs (sampled every 50 ms)."""

    def __enter__(self):
        self.me = psutil.Process()
        self.before = {p.pid for p in self.me.children(recursive=True)}
        self.seen = {}
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        self.t0 = time.monotonic()
        return self

    def _loop(self):
        while not self._stop.is_set():
            try:
                for p in self.me.children(recursive=True):
                    if p.pid not in self.before:
                        self.seen.setdefault(p.pid, p)
            except psutil.Error:
                pass
            self._stop.wait(0.05)

    def __exit__(self, *exc):
        self.elapsed = time.monotonic() - self.t0
        self._stop.set()
        self._t.join()
        return False

    def alive(self):
        return sorted(p.pid for p in self.seen.values() if p.is_running())


def test_result_and_output_come_back(stage):
    with Descendants() as d:
        res = run(stage, "ok")
    assert isinstance(res, StageResult)
    assert res.outputs == {"x": "a/b.txt"}
    assert res.anomalies == [Anomaly("note", "warning", "from the child", data={"k": 1})]
    m = res.metrics
    assert m["pid"] != os.getpid() and m["sid"] == SID and m["root"] == str(stage.paths.root)
    assert m["cwd"] == os.getcwd()  # the old config.py writes config.json into the current folder
    assert m["coretype"] == "Sandybridge" and m["faulthandler_env"] == "1" and m["faulthandler_on"] is True
    text = log_text(stage)
    assert "fake stage: stdout line" in text and "fake stage: stderr line" in text
    assert "INFO fake stage: info record" in text
    assert d.alive() == []


def test_exception_reraised_with_type_message_and_child_traceback(stage):
    with pytest.raises(ValueError) as ei:
        run(stage, "value_error")
    assert str(ei.value) == "x"
    assert "isolate_fakes.py" in str(ei.value.__cause__) and "value_error" in str(ei.value.__cause__)


def test_waiting_for_input_reraised(stage):
    with pytest.raises(WaitingForInput) as ei:
        run(stage, "waiting")
    assert str(ei.value) == "waiting for thresholds: gSig not set" and ei.value.what == "thresholds"


def test_waiting_for_input_pickles():
    w = pickle.loads(pickle.dumps(WaitingForInput("crop_finalize")))
    assert str(w) == "waiting for crop_finalize" and w.what == "crop_finalize"


def test_native_crash_is_stage_crashed_and_the_tree_dies(stage):
    with Descendants() as d:
        with pytest.raises(isolate.StageCrashed) as ei:
            run(stage, "native_crash")
    assert ei.value.exit_code == 0xC0000005
    assert str(ei.value) == "stage process exited with code 0xc0000005"
    assert d.elapsed < 10
    assert survivors(tree(stage)) == [], "a grandchild outlived the crashed stage process"
    assert d.alive() == []
    text = log_text(stage)  # faulthandler: where the stage process crashed
    assert "Windows fatal exception: access violation" in text and "native_crash" in text


def test_abort_exit_code(stage):
    with pytest.raises(isolate.StageCrashed, match=r"^stage process exited with code 0x3$") as ei:
        run(stage, "abort")
    assert ei.value.exit_code == 3


def test_dead_pool_worker_fails_the_stage_instead_of_hanging(stage):
    with Descendants() as d:
        with pytest.raises(isolate.StageCrashed, match=r"^pool worker died$") as ei:
            run(stage, "pool_worker_killed")
    assert ei.value.exit_code == POOL_WORKER_DIED
    assert d.elapsed < 15
    rows = tree(stage)  # the stage process and its two first workers
    assert len(rows) == 3 and survivors(rows) == []
    assert d.alive() == []  # also the replacement worker, if the sampler saw it
    assert f"pool worker died: pid {rows[1][0]}" in log_text(stage)


def test_normal_pool_shutdown_is_not_a_death(stage):
    with Descendants() as d:
        res = run(stage, "pool_normal")
    assert res.metrics == {"map": [1, 2, 3]}
    assert "pool worker died" not in log_text(stage)
    assert d.alive() == []


def test_interrupt_kills_the_tree(stage):
    tree_file = stage.notes / "tree.json"

    def interrupt():
        deadline = time.monotonic() + 20
        while not tree_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        _thread.interrupt_main()  # what Ctrl+C does to the runner's main thread

    t = threading.Thread(target=interrupt, daemon=True)
    with Descendants() as d:
        t.start()
        with pytest.raises(KeyboardInterrupt):
            run(stage, "hang")
    t.join()
    assert tree_file.exists() and d.elapsed < 15
    assert survivors(tree(stage)) == []
    assert d.alive() == []


def test_only_dotted_specs(stage):
    with pytest.raises(TypeError, match="module:function"):
        isolate.run_isolated(lambda *a: None, stage.paths, stage.rec, stage.cfg, stage.log)
    assert not Path(stage.log).exists()


def test_child_exits_right_after_its_result(stage, caplog):
    """Interpreter shutdown of a finished stage can hang (seen with joblib's workers): the child ends at once
    after sending its result instead of being killed EXIT_WAIT_S later."""
    caplog.set_level("INFO", logger="bearmind.isolate")
    with Descendants() as d:
        res = run(stage, "lingering")
    assert res.metrics == {"ok": True}
    assert d.elapsed < 10 and d.alive() == []
    assert any("exit 0x0" in r.getMessage() for r in caplog.records), [r.getMessage() for r in caplog.records]
