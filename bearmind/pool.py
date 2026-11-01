"""The CaImAn multiprocessing pool of the MC and CNMF-E stages, with a watchdog for dead workers.

multiprocessing.Pool replaces a worker that dies (a native crash, an external kill), but the task that
worker held never returns: CaImAn then waits in map_async(...).get(4294967) forever, with the batch
heartbeat still alive (seen in Task 18). CaImAn creates the pool without maxtasksperchild, so a worker
never exits on its own while the pool runs; any change of the worker set means a worker died. The
watchdog then ends the stage process with POOL_WORKER_DIED: its main thread is blocked inside CaImAn and
cannot be interrupted, and the runner (bearmind.isolate) turns that exit code into a failed stage.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from multiprocessing.pool import RUN

import psutil

POOL_WORKER_DIED = 75  # exit code of a stage process whose pool lost a worker (EX_TEMPFAIL)
WATCH_S = 0.5

log = logging.getLogger("bearmind.pool")


def _die(pids) -> None:
    """End this process: the stage cannot recover a task lost with its worker."""
    try:
        print(f"pool worker died: pid {', '.join(map(str, pids))}", file=sys.stderr, flush=True)
    finally:
        # os._exit skips the atexit handler that would terminate the pool, so the workers go first
        try:
            for p in psutil.Process().children(recursive=True):
                try:
                    p.kill()
                except psutil.Error:
                    pass
        finally:
            os._exit(POOL_WORKER_DIED)


class _Watchdog:
    """Daemon thread comparing the pool's live workers with the ones it started with.

    Uses the private attributes Pool._pool (the worker Process objects) and Pool._state ("RUN" until close
    or terminate): multiprocessing has no public way to see a worker that was replaced. Checked against
    CPython 3.12: _state is set before close/terminate touch the workers, so a shutdown is never seen as a
    death; the watch ends as soon as the state leaves RUN.
    """

    def __init__(self, pool, on_death=_die, every_s: float = WATCH_S):
        self.pool, self.on_death, self.every_s = pool, on_death, every_s
        self.pids = {p.pid for p in pool._pool}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="bearmind-pool-watchdog", daemon=True)
        self._thread.start()

    def _dead(self) -> set:
        workers = list(self.pool._pool)  # a copy: the pool's worker handler thread edits the list
        gone = self.pids - {p.pid for p in workers}
        gone |= {p.pid for p in workers if p.pid in self.pids and p.exitcode is not None}
        # read again after the workers: a close/terminate that started meanwhile is not a death
        return gone if self.pool._state == RUN and not self._stop.is_set() else set()

    def _loop(self):
        while not self._stop.wait(self.every_s):
            if self.pool._state != RUN:
                return
            dead = self._dead()
            if dead:
                self.on_death(sorted(dead))
                return

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not threading.current_thread():
            self._thread.join()


def start_pool(n_processes: int):
    """CaImAn's multiprocessing pool (dview) of n_processes workers, watched for dead workers."""
    from caiman.cluster import setup_cluster

    _, dview, _ = setup_cluster(backend="multiprocessing", n_processes=int(n_processes), single_thread=False)
    dview._bearmind_watchdog = _Watchdog(dview)
    return dview


def stop_pool(dview) -> None:
    """Stop the watchdog, then terminate the pool and wait for its workers.

    cm.stop_server(dview=pool) only calls pool.terminate() for a multiprocessing pool, so this is the
    same shutdown. Errors are logged, not raised: this runs in finally blocks and must not hide the
    stage's own exception.
    """
    watch = getattr(dview, "_bearmind_watchdog", None)
    if watch is not None:
        watch.stop()  # first: the workers leaving below are not deaths
    try:
        dview.terminate()
    except Exception as e:
        log.warning("pool not terminated cleanly: %s: %s", type(e).__name__, e)
