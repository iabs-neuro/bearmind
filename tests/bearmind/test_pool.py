"""Pool watchdog: a dead worker is reported, a normal shutdown is not.

The watchdog's default reaction ends the process (os._exit), so these in-process tests give it a recording
callback; the real reaction inside a stage process is tested in test_isolate.py.
"""
import multiprocessing
import threading
import time

import psutil

from bearmind import pool

WAIT_S = 10.0


def _pool(n=2):
    return multiprocessing.get_context("spawn").Pool(n)


def _wait(event, timeout=WAIT_S):
    return event.wait(timeout)


def _watch(p):
    died, event = [], threading.Event()

    def on_death(pids):
        died.append(pids)
        event.set()

    return pool._Watchdog(p, on_death=on_death, every_s=0.05), died, event


def test_killed_worker_is_reported():
    p = _pool()
    try:
        w, died, event = _watch(p)
        victim = p._pool[0].pid
        psutil.Process(victim).kill()
        assert _wait(event), "the watchdog did not report the killed worker"
        assert died == [[victim]]
    finally:
        w.stop()
        p.terminate()


def test_stop_then_terminate_is_not_a_death():
    p = _pool()
    w, died, _ = _watch(p)
    assert p.map(abs, [-1, -2, 3]) == [1, 2, 3]
    w.stop()
    p.terminate()
    time.sleep(0.3)  # several watchdog periods: a still-running watchdog would see the workers go
    assert died == []
    assert not w._thread.is_alive()


def test_pool_closed_by_its_owner_is_not_a_death():
    """CaImAn closing or terminating the pool itself (state no longer RUN) ends the watch without a report."""
    p = _pool()
    w, died, _ = _watch(p)
    p.close()
    p.join()  # workers exit after close: not a death
    time.sleep(0.3)
    assert died == []
    w._thread.join(WAIT_S)
    assert not w._thread.is_alive()


def test_stop_pool_stops_the_watchdog_first(monkeypatch):
    p = _pool()
    order = []
    w = pool._Watchdog(p, on_death=lambda pids: order.append(("death", pids)), every_s=0.05)
    p._bearmind_watchdog = w
    real_stop, real_terminate = w.stop, p.terminate
    monkeypatch.setattr(w, "stop", lambda: (order.append("stop"), real_stop()))
    monkeypatch.setattr(p, "terminate", lambda: (order.append("terminate"), real_terminate()))
    pool.stop_pool(p)
    time.sleep(0.3)
    assert order == ["stop", "terminate"]
    assert all(x.exitcode is not None for x in p._pool)


def test_start_pool_records_workers_and_watches():
    n = 2
    dview = pool.start_pool(n)
    try:
        w = dview._bearmind_watchdog
        assert w.pids == {x.pid for x in dview._pool} and len(w.pids) == n
        assert w._thread.is_alive() and w._thread.daemon
        assert dview.map(abs, [-1, -2]) == [1, 2]
    finally:
        pool.stop_pool(dview)
    assert not dview._bearmind_watchdog._thread.is_alive()


def _new_children(before):
    return [p for p in psutil.Process().children(recursive=True) if p.pid not in before]


def _stop_in_thread(dview, bound):
    """stop_pool in a daemon thread: a hang fails the test instead of blocking the suite."""
    t = threading.Thread(target=pool.stop_pool, args=(dview,), daemon=True)
    t0 = time.monotonic()
    t.start()
    t.join(bound)
    return not t.is_alive(), time.monotonic() - t0


def test_stop_pool_after_idle_workers_were_killed_does_not_hang(monkeypatch):
    """An idle worker blocks in the task queue's get() holding its read lock; killed there, the lock stays
    taken and Pool.terminate() waits for it forever. A death in the last watchdog period before stop_pool
    must not hang the stage."""
    reported = []
    monkeypatch.setattr(pool, "_die", reported.append)  # a watchdog that fires must not end pytest
    before = {p.pid for p in psutil.Process().children(recursive=True)}
    dview = pool.start_pool(2)
    try:
        assert dview.map(abs, [-1, -2, -3, -4]) == [1, 2, 3, 4]  # workers booted, now idle in get()
        time.sleep(0.3)
        for w in list(dview._pool):  # every idle worker, so the one holding the lock is among them
            p = psutil.Process(w.pid)
            p.kill()
            p.wait(5)
        done, took = _stop_in_thread(dview, 15)
        assert done, f"stop_pool still blocked after {took:.1f} s"
        assert took < 15
        left = [p.pid for p in _new_children(before) if p.is_running()]
        assert left == [], f"pool processes left alive: {left}"
    finally:
        for p in _new_children(before):
            p.kill()


def test_stop_pool_bounds_a_blocked_terminate(monkeypatch):
    p = _pool()
    before = set()
    monkeypatch.setattr(pool, "TERMINATE_S", 0.5)
    w = pool._Watchdog(p, on_death=lambda pids: None, every_s=0.05)
    p._bearmind_watchdog = w
    workers = [psutil.Process(x.pid) for x in p._pool]
    monkeypatch.setattr(p, "terminate", lambda: time.sleep(30))  # stuck for any reason
    done, took = _stop_in_thread(p, 10)
    assert done and took < 5, took
    assert [x.pid for x in workers if x.is_running()] == []


class _FakeProc:
    def __init__(self, pid, exitcode=None):
        self.pid, self.exitcode = pid, exitcode


class _FlakyPool:
    """Pool stand-in whose worker list cannot be read once (as a transient error in the private API)."""
    _state = "RUN"

    def __init__(self):
        self.reads = 0

    @property
    def _pool(self):
        self.reads += 1
        if self.reads == 2:
            raise RuntimeError("transient")
        if self.reads == 1:
            return [_FakeProc(1), _FakeProc(2)]
        return [_FakeProc(1), _FakeProc(2, exitcode=1)]


def test_watchdog_survives_an_error_and_still_reports_the_death(caplog):
    died, event = [], threading.Event()
    w = pool._Watchdog(_FlakyPool(), on_death=lambda pids: (died.append(pids), event.set()), every_s=0.05)
    try:
        assert _wait(event), "the watchdog stopped watching after an error"
        assert died == [[2]]
        assert any("transient" in r.getMessage() or (r.exc_info and "transient" in str(r.exc_info[1]))
                   for r in caplog.records)
    finally:
        w.stop()
