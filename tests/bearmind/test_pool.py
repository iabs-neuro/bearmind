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
