"""Fake stages run in a real child process by bearmind.isolate (test_isolate.py, test_runner.py).

They are named by 'module:function' like the real stages. cfg["fake"] says what they need: a folder to
write their process ids to (`dir`), which sessions crash (`crash`).
"""
import faulthandler
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from bearmind.registry import Anomaly
from bearmind.stages import StageResult, WaitingForInput

SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


def _tree(cfg, pids) -> None:
    """Write [pid, creation time] of each process for the test to check that none of them survives."""
    import psutil

    d = Path(cfg["fake"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    rows = [[p, psutil.Process(p).create_time()] for p in pids]
    (d / "tree.json").write_text(json.dumps(rows), encoding="utf-8")


def _calls(cfg, rec, stage) -> None:
    f = Path(cfg["fake"]["dir"]) / "calls.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(f"{rec.sid} {stage}\n")


def ok(paths, rec, cfg):
    print("fake stage: stdout line")
    print("fake stage: stderr line", file=sys.stderr)
    import logging
    logging.getLogger("bearmind.fake").info("fake stage: info record")
    return StageResult(
        outputs={"x": "a/b.txt"},
        metrics={"pid": os.getpid(), "sid": rec.sid, "cwd": os.getcwd(), "root": str(paths.root),
                 "coretype": os.environ.get("OPENBLAS_CORETYPE"),
                 "faulthandler_env": os.environ.get("PYTHONFAULTHANDLER"),
                 "faulthandler_on": faulthandler.is_enabled()},
        anomalies=[Anomaly("note", "warning", "from the child", data={"k": 1})])


def value_error(paths, rec, cfg):
    raise ValueError("x")


def waiting(paths, rec, cfg):
    raise WaitingForInput("thresholds", "gSig not set")


def native_crash(paths, rec, cfg):
    """A grandchild that would outlive its parent, then an access violation (0xC0000005 on Windows)."""
    g = subprocess.Popen(SLEEPER)
    _tree(cfg, [os.getpid(), g.pid])
    faulthandler._read_null()


def abort(paths, rec, cfg):
    os.abort()


def pool_worker_killed(paths, rec, cfg):
    import psutil

    from bearmind import pool

    dview = pool.start_pool(2)
    workers = [p.pid for p in dview._pool]
    _tree(cfg, [os.getpid(), *workers])
    psutil.Process(workers[0]).kill()
    (Path(cfg["fake"]["dir"]) / "killed_at.json").write_text(json.dumps(time.time()), encoding="utf-8")
    dview.map_async(time.sleep, [60] * 4).get(4294967)  # as CaImAn waits: forever without the watchdog


def pool_normal(paths, rec, cfg):
    from bearmind import pool

    dview = pool.start_pool(2)
    try:
        res = dview.map(abs, [-1, -2, 3])
    finally:
        pool.stop_pool(dview)
    time.sleep(3 * pool.WATCH_S)  # a watchdog still running would now see the workers gone
    return StageResult(metrics={"map": res})


def blocked(paths, rec, cfg):
    """A deadlock: no CPU and no output, forever; a grandchild that sleeps as well."""
    import threading

    g = subprocess.Popen(SLEEPER)
    _tree(cfg, [os.getpid(), g.pid])
    threading.Event().wait()


def busy(paths, rec, cfg):
    """Computes without any output for cfg["fake"]["busy_s"] seconds."""
    t0, n = time.monotonic(), 0
    while time.monotonic() - t0 < cfg["fake"]["busy_s"]:
        n += 1
    return StageResult(metrics={"busy_s": time.monotonic() - t0, "n": n})


def hang(paths, rec, cfg):
    g = subprocess.Popen(SLEEPER)
    _tree(cfg, [os.getpid(), g.pid])
    time.sleep(600)


# --- stages for the runner test (crop and mc of a tiny series) ---

def crop(paths, rec, cfg):
    _calls(cfg, rec, "crop")
    return StageResult(metrics={"fov": [2, 3, 10, 9], "frame_shape": [16, 16]})


def mc(paths, rec, cfg):
    _calls(cfg, rec, "mc")
    if rec.sid in cfg["fake"].get("crash", []):
        faulthandler._read_null()
    if rec.sid in cfg["fake"].get("fail", []):
        raise ValueError("boom")
    if rec.sid in cfg["fake"].get("stall", []):
        import threading
        threading.Event().wait()
    s = rec.sid
    out = {}
    for k, p in (("mc_video", paths.mc_video(s)), ("mc_shifts", paths.mc_shifts(s))):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        out[k] = paths.rel(p)
    return StageResult(outputs=out, metrics={"max_shift_px": 2.0, "mean_template_corr": 0.9})


def lingering(paths, rec, cfg):
    """A result, and a non-daemon thread that would keep the interpreter from exiting (as loky's did)."""
    import threading

    threading.Thread(target=time.sleep, args=(600,), daemon=False).start()
    return StageResult(metrics={"ok": True})


def maps_worker_killed(paths, rec, cfg):
    """The parallel threshold maps on a small memmap; a pool worker is killed right after the pool starts."""
    import threading

    import numpy as np
    import psutil

    from bearmind import pool
    from bearmind.stages import thresholds

    d = Path(cfg["fake"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    shape = (600, 48, 48)
    u8 = d / "u8.dat"
    np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8).tofile(u8)
    real = pool.start_pool

    def start(n):
        dview = real(n)
        workers = [p.pid for p in dview._pool]
        _tree(cfg, [os.getpid(), *workers])

        def kill():
            psutil.Process(workers[0]).kill()
            (d / "killed_at.json").write_text(json.dumps(time.time()), encoding="utf-8")

        threading.Thread(target=kill).start()
        return dview

    pool.start_pool = start
    thresholds._parallel_maps(u8, shape, [3, 4, 5], 1, 1.0, 3)
    return StageResult()
