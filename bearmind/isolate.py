"""Run one pipeline stage in its own child process: a native crash or a dead CaImAn pool worker fails that
stage instead of killing or hanging the batch.

The child is started with multiprocessing 'spawn' and named by the stage's 'module:function' spec. It
writes its stdout/stderr (and those of the processes it starts, such as CaImAn's pool workers) to the
session log, with faulthandler on, so a native crash leaves its Python traceback there. The parent polls
(at most POLL_S between checks), so its heartbeat thread keeps running and Ctrl+C is handled; whatever
ends the wait, the whole child process tree is killed and waited for before run_isolated returns.

On Windows the child is put into a job object that kills every process in it when its handle closes:
psutil cannot find the orphaned pool workers of a child that has already died, and the job also takes the
stage tree down with the runner if the runner itself is killed.
"""
from __future__ import annotations

# Only the standard library, psutil and light bearmind modules here: the child imports this module
# before anything else, and numpy / CaImAn must load only after faulthandler is on (bearmind's
# __init__ has set OPENBLAS_CORETYPE by then).
import faulthandler
import importlib
import logging
import multiprocessing
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import psutil

from bearmind.pool import POOL_WORKER_DIED
from bearmind.stages import WaitingForInput

POLL_S = 1.0
EXIT_WAIT_S = 30.0  # after sending its result the child gets this long to exit on its own
KILL_WAIT_S = 10.0

log = logging.getLogger("bearmind.isolate")


class StageCrashed(RuntimeError):
    """The stage process ended without a result: a native crash, a kill, or a dead pool worker."""

    def __init__(self, message: str, exit_code: int | None = None):
        super().__init__(message)
        self.exit_code = exit_code

    def __reduce__(self):
        return type(self), (str(self), self.exit_code)


class _RemoteTraceback(Exception):
    """The child's traceback text, chained as __cause__ of the exception re-raised in the parent."""

    def __str__(self):
        return self.args[0]


def resolve(spec: str):
    """The function named by a 'module:function' spec."""
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


# --- Windows job object ---

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    _k32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD)
    _k32.QueryInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.c_void_p)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    _k32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)

    _JOB_BASIC_ACCOUNTING, _JOB_EXTENDED_LIMITS = 1, 9  # JOBOBJECTINFOCLASS
    _KILL_ON_JOB_CLOSE = 0x2000
    _PROCESS_TERMINATE, _PROCESS_SET_QUOTA = 0x0001, 0x0100

    class _BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class _ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", _BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class _Accounting(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

    def _check(ok):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return ok

    class _Job:
        """A job object that kills all its processes when its (non-inheritable) handle is closed."""

        def __init__(self, pid: int):
            self.handle = _check(_k32.CreateJobObjectW(None, None))
            try:
                info = _ExtendedLimits()
                info.BasicLimitInformation.LimitFlags = _KILL_ON_JOB_CLOSE
                _check(_k32.SetInformationJobObject(self.handle, _JOB_EXTENDED_LIMITS, ctypes.byref(info),
                                                    ctypes.sizeof(info)))
                proc = _check(_k32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, False, pid))
                try:
                    _check(_k32.AssignProcessToJobObject(self.handle, proc))
                finally:
                    _k32.CloseHandle(proc)
            except BaseException:
                self.close()
                raise

        def _query(self, cls, info):
            _check(_k32.QueryInformationJobObject(self.handle, cls, ctypes.byref(info), ctypes.sizeof(info),
                                                  None))
            return info

        def peak_gb(self) -> float:
            """Peak commit of all processes of the job together."""
            return self._query(_JOB_EXTENDED_LIMITS, _ExtendedLimits()).PeakJobMemoryUsed / 2 ** 30

        def kill(self, timeout: float) -> int:
            """Terminate every process of the job; the number still running after `timeout`."""
            _k32.TerminateJobObject(self.handle, 1)
            deadline = time.monotonic() + timeout
            while True:
                active = self._query(_JOB_BASIC_ACCOUNTING, _Accounting()).ActiveProcesses
                if not active or time.monotonic() > deadline:
                    return active
                time.sleep(0.05)

        def close(self):
            if self.handle:
                _k32.CloseHandle(self.handle)
                self.handle = None
else:
    _Job = None


class _Tree:
    """The child and every process it starts."""

    def __init__(self, pid: int):
        try:
            self.root = psutil.Process(pid)
        except psutil.NoSuchProcess:  # died before we looked: it had not started anything yet
            self.root = None
        self.seen: dict[int, psutil.Process] = {}
        self.job = None
        if _Job is not None and self.root is not None:
            try:
                self.job = _Job(pid)
            except OSError as e:  # e.g. a job that forbids nesting: fall back to the processes seen
                log.warning("stage process not put into a job object (%s); its tree is killed by pid", e)

    def track(self) -> None:
        """Remember the current descendants: they cannot be found from the child once it has died."""
        if self.root is None:
            return
        try:
            for p in self.root.children(recursive=True):
                self.seen.setdefault(p.pid, p)
        except psutil.Error:
            pass

    def kill(self) -> None:
        self.track()
        procs = [p for p in [self.root, *self.seen.values()] if p is not None]
        left = self.job.kill(KILL_WAIT_S) if self.job is not None else 0
        for p in procs:
            try:
                p.kill()  # psutil checks the creation time: a reused pid is not touched
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(procs, timeout=KILL_WAIT_S)
        if left or alive:
            log.warning("stage process tree not fully killed: %d in the job, pids %s still running", left,
                        [p.pid for p in alive])

    def close(self) -> None:
        if self.job is not None:
            self.job.close()


# --- the child ---

_LOG_FILE = None  # kept open for the life of the child: faulthandler writes to its descriptor


def _setup_child(log_path: str) -> None:
    global _LOG_FILE
    os.environ["PYTHONFAULTHANDLER"] = "1"  # processes the stage starts (CaImAn's pool) dump crashes too
    _LOG_FILE = open(log_path, "ab", buffering=0)
    faulthandler.enable(_LOG_FILE, all_threads=True)
    for fd in (1, 2):  # also the process's standard handles: native code and child processes write here
        os.dup2(_LOG_FILE.fileno(), fd)
    sys.stdout = open(1, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    sys.stderr = open(2, "w", encoding="utf-8", errors="backslashreplace", buffering=1, closefd=False)
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    lg = logging.getLogger("bearmind")
    lg.addHandler(h)
    lg.setLevel(logging.INFO)


def _pack(e: BaseException) -> tuple:
    tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
    try:
        data = pickle.dumps(e)
        pickle.loads(data)  # an exception whose __init__ does not match its args fails here, not in the parent
    except Exception:
        data = None
    return data, f"{type(e).__module__}.{type(e).__qualname__}", str(e), tb


def _child(func_spec: str, blob: bytes, log_path: str, conn) -> None:
    _setup_child(log_path)
    if conn.recv() != "go":  # the parent has put this process into its job: nothing escapes it now
        return
    try:
        paths, rec, cfg = pickle.loads(blob)
        msg = ("ok", resolve(func_spec)(paths, rec, cfg))
    except BaseException as e:
        msg = ("error", *_pack(e))
    try:
        conn.send(msg)
    except Exception as e:  # a result that cannot be pickled
        conn.send(("error", *_pack(e)))
    sys.stdout.flush()
    sys.stderr.flush()
    conn.close()


# --- the parent ---

def _unpack(data, name, text, tb) -> tuple[BaseException, str]:
    exc = None
    if data is not None:
        try:
            exc = pickle.loads(data)
        except Exception:  # e.g. a class the parent cannot import
            exc = None
    if exc is None:
        exc = RuntimeError(f"{name}: {text}")
    return exc, tb


def _wait(proc, conn, tree):
    """The child's message, or None when it ended without sending one."""
    try:
        conn.send("go")
        while True:
            if conn.poll(POLL_S):
                return conn.recv()
            if not proc.is_alive():
                return conn.recv() if conn.poll(0) else None  # a message sent just before it exited
            tree.track()
    except (EOFError, OSError):  # the child closed its end of the pipe: it is gone
        return None


def run_isolated(func_spec: str, paths, rec, cfg, log_path: Path):
    """Run `func_spec(paths, rec, cfg)` in a child process and return its StageResult.

    An exception of the stage is raised here with the same type and message (the child's traceback is its
    __cause__); a child that ends without a result raises StageCrashed with its exit code.
    """
    if not isinstance(func_spec, str):
        raise TypeError(f"an isolated stage is named by a 'module:function' string, not {func_spec!r}")
    blob = pickle.dumps((paths, rec, cfg))  # unpickled in the child only after faulthandler is on
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ctx = multiprocessing.get_context("spawn")
    here, there = ctx.Pipe()
    proc = ctx.Process(target=_child, args=(func_spec, blob, str(log_path), there),
                       name=f"bearmind {func_spec}")
    t0 = time.monotonic()
    proc.start()
    there.close()  # else the parent's copy keeps the pipe open and a dead child is never noticed
    tree = _Tree(proc.pid)
    try:
        msg = _wait(proc, here, tree)
        if msg is not None:
            proc.join(EXIT_WAIT_S)
        tree.kill()  # a child stuck on exit, workers a stage did not stop
        peak = tree.job.peak_gb() if tree.job is not None else None
    except BaseException:  # Ctrl+C, or an error while waiting: the tree must not outlive this call
        tree.kill()
        raise
    finally:
        tree.close()
        here.close()
    proc.join(KILL_WAIT_S)
    code = proc.exitcode
    exit_text = "unknown" if code is None else f"{code:#x}"

    def report():
        log.info("stage process %s: exit %s after %.0f s%s", func_spec, exit_text, time.monotonic() - t0,
                 "" if peak is None else f", peak commit of its process tree {peak:.1f} GB")

    if msg is None:
        report()
        raise StageCrashed("pool worker died" if code == POOL_WORKER_DIED
                           else f"stage process exited with code {exit_text}", code)
    kind, *rest = msg
    if kind == "ok":
        report()
        return rest[0]
    exc, tb = _unpack(*rest)
    if not isinstance(exc, WaitingForInput):  # a waiting stage is polled: one line per poll is noise
        report()
    raise exc from _RemoteTraceback(tb)
