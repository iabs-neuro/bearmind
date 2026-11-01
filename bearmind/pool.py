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
from multiprocessing.pool import RUN, TERMINATE

import psutil

POOL_WORKER_DIED = 75  # exit code of a stage process whose pool lost a worker (EX_TEMPFAIL)
WATCH_S = 0.5
TERMINATE_S = 30.0  # Pool.terminate() normally returns in about a second; longer means it is stuck
KILL_WAIT_S = 10.0

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

    def __init__(self, pool, on_death=None, every_s: float | None = None):
        # resolved here, not in the signature: tests replace the module's _die / WATCH_S
        self.pool = pool
        self.on_death = _die if on_death is None else on_death
        self.every_s = WATCH_S if every_s is None else every_s
        self.pids = {p.pid for p in pool._pool}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="bearmind-pool-watchdog", daemon=True)
        self._thread.start()

    def gone(self) -> set:
        """Recorded workers no longer in the pool or already exited, whatever the pool state."""
        workers = list(self.pool._pool)  # a copy: the pool's worker handler thread edits the list
        gone = self.pids - {p.pid for p in workers}
        return gone | {p.pid for p in workers if p.pid in self.pids and p.exitcode is not None}

    def _dead(self) -> set:
        gone = self.gone()
        # read again after the workers: a close/terminate that started meanwhile is not a death
        return gone if self.pool._state == RUN and not self._stop.is_set() else set()

    def _loop(self):
        logged = set()
        while not self._stop.wait(self.every_s):
            try:
                if self.pool._state != RUN:
                    return
                dead = self._dead()
            except Exception as e:
                # Keep watching: a bookkeeping error must not end a healthy hour-long stage, and a thread
                # that stops here would bring the silent hang back. Logged once per kind of error.
                if repr(e) not in logged:
                    logged.add(repr(e))
                    log.exception("pool watchdog: could not read the pool's workers; still watching")
                continue
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


def _terminate(dview, timeout: float) -> bool:
    """Pool.terminate() in a daemon thread; True when it returned without error within `timeout`."""
    errors = []

    def target():
        try:
            dview.terminate()
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=target, name="bearmind-pool-terminate", daemon=True)
    t.start()
    t.join(timeout)
    for e in errors:
        log.warning("pool not terminated cleanly: %s: %s", type(e).__name__, e)
    return not t.is_alive() and not errors


def _kill(dview) -> None:
    """Kill the pool's workers without Pool.terminate().

    Pool.terminate() first takes the task queue's read lock (Pool._help_stuff_finish), which an idle worker
    holds while it waits for a task; if that worker was killed, the lock is never released and terminate
    waits forever. So: stop the worker handler first (it would start replacements), kill and reap the
    workers, and cancel the pool's exit-time finalizer, which would call terminate() again and block the
    interpreter's exit. Private attributes used for that: Pool._worker_handler, _change_notifier, _pool,
    _terminate (a multiprocessing.util.Finalize). The task and result handler threads are daemons and are
    left to end with the process.
    """
    dview._state = TERMINATE
    handler = dview._worker_handler
    handler._state = TERMINATE
    dview._change_notifier.put(None)  # wakes the handler, which then sees TERMINATE and returns
    handler.join(KILL_WAIT_S)
    workers = list(dview._pool)
    for p in workers:
        if p.exitcode is None:
            p.kill()
    for p in workers:
        p.join(KILL_WAIT_S)
    alive = [p.pid for p in workers if p.exitcode is None]
    if alive:
        log.warning("pool workers still running after kill: %s", alive)
    dview._terminate.cancel()


def stop_pool(dview) -> None:
    """Stop the watchdog, then terminate the pool and wait for its workers; never hangs.

    cm.stop_server(dview=pool) only calls pool.terminate() for a multiprocessing pool, so the normal path is
    the same shutdown. A worker that died in the last watchdog period (not reported yet) may hold the task
    queue's lock that terminate() needs, so then the workers are killed instead; a terminate() that does
    not return within TERMINATE_S is abandoned the same way. Errors are logged, not raised: this runs in
    finally blocks and must not hide the stage's own exception.
    """
    watch = getattr(dview, "_bearmind_watchdog", None)
    dead = set()
    if watch is not None:
        watch.stop()  # first: the workers leaving below are not deaths
        try:
            dead = watch.gone() if dview._state == RUN else set()
        except Exception as e:
            log.warning("pool workers not checked before shutdown: %s: %s", type(e).__name__, e)
    if dead:
        log.warning("pool worker died before the pool was stopped: pid %s; workers killed instead of "
                    "Pool.terminate()", ", ".join(map(str, sorted(dead))))
        _kill(dview)
    elif not _terminate(dview, TERMINATE_S):
        log.warning("Pool.terminate() did not finish within %.0f s; workers killed", TERMINATE_S)
        _kill(dview)
