"""Safe writes of files that several processes read and write (registry, configs, heartbeat, summary).

Windows specifics: os.replace fails with PermissionError while another process has the target open
(a GUI reading the registry, Excel holding summary.csv), so replaces are retried; the temp file name is
unique per writer, so two writers never share it.
"""
from __future__ import annotations

import os
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import psutil

LOCK_TIMEOUT_S = 10.0  # wait for a live writer at most this long
LOCK_STALE_S = 60.0  # fallback: a lock this old is broken even if its writer looks alive (writes take ms)
LOCK_STEP_S = 0.02
REPLACE_TRIES = 20
REPLACE_STEP_S = 0.25


class LockTimeout(TimeoutError):
    """A live writer held the lock longer than LOCK_TIMEOUT_S."""


_ME: dict = {}


def _me() -> str:
    pid = os.getpid()
    if pid not in _ME:
        _ME[pid] = f"{pid} {psutil.Process(pid).create_time()!r}"
    return _ME[pid]


def _holder_alive(text: str) -> bool:
    """Lock text is '<pid> <process start time> <nonce>'; empty or foreign text counts as alive."""
    parts = text.split()
    try:
        pid = int(parts[0])
        start = float(parts[1]) if len(parts) > 1 else None
    except (IndexError, ValueError):
        return True  # being written right now, or not ours to judge: the age rule decides
    try:
        if not psutil.pid_exists(pid):
            return False
        return start is None or abs(psutil.Process(pid).create_time() - start) < 1.0  # PID reuse
    except psutil.NoSuchProcess:
        return False
    except psutil.Error:
        return True


BREAK_STALE_S = 5.0  # a break mutex older than this was left by a contender that died while breaking


def _break(lock: Path, seen: str) -> bool:
    """Remove a dead writer's lock. Returns False when it could not be done now (retry shortly).

    Under a short `<lock>.break` mutex the lock is reread: while the dead lock file exists nobody can create
    a new one (O_EXCL), so if it still holds the text judged dead, renaming it to a unique name removes
    exactly that file and never a live lock that replaced it.
    """
    mutex = lock.with_name(lock.name + ".break")
    try:
        os.close(os.open(mutex, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        try:
            if time.time() - mutex.stat().st_mtime > BREAK_STALE_S:
                os.unlink(mutex)
        except OSError:
            pass
        return False
    except PermissionError:
        return False
    try:
        if _read_lock(lock) != seen:
            return True  # released or replaced meanwhile: look again
        grave = lock.with_name(f"{lock.name}.{uuid.uuid4().hex}.stale")
        try:
            os.replace(lock, grave)
        except FileNotFoundError:
            return True
        except PermissionError:
            return False  # open in a reader right now
        _unlink(grave)
        return True
    finally:
        _unlink(mutex)


def _unlink(f, tries: int = 50, step: float = 0.02) -> None:
    """os.unlink, retried while another process has the file open (Windows sharing violation)."""
    for i in range(tries):
        try:
            os.unlink(f)
            return
        except FileNotFoundError:
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(step)


def _read_lock(lock: Path) -> str | None:
    try:
        return lock.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (PermissionError, UnicodeDecodeError):
        return ""  # being written: counts as held


@contextmanager
def file_lock(target):
    """Short exclusive lock `<target>.lock` (O_CREAT | O_EXCL) around a read-modify-write of `target`.

    A lock whose writer process is gone (PID not running, or running with another start time) is broken at
    once; any lock older than LOCK_STALE_S is broken as a fallback. LockTimeout after LOCK_TIMEOUT_S.
    """
    target = Path(target)
    lock = target.with_name(target.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    token = f"{_me()} {uuid.uuid4().hex}"
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            seen = _read_lock(lock)
            if seen is None:
                continue  # released between the two calls
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue
            if (not _holder_alive(seen) or age > LOCK_STALE_S) and _break(lock, seen):
                continue
        except PermissionError:
            pass  # Windows: the previous holder's lock file is still being deleted
        else:
            os.write(fd, token.encode())
            os.close(fd)
            break
        if time.monotonic() >= deadline:
            raise LockTimeout(f"lock is held by another writer: {lock}")
        time.sleep(LOCK_STEP_S)
    try:
        yield
    finally:
        # remove only our own lock: if it was broken meanwhile, the file belongs to someone else now
        if _read_lock(lock) == token:
            _unlink(lock)


def replace(src, dst, tries: int | None = None) -> None:
    """os.replace, retried while the target is open in another process (PermissionError on Windows)."""
    tries = REPLACE_TRIES if tries is None else tries
    for i in range(tries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(REPLACE_STEP_S)


def write_atomic(path, write, newline=None, replace_tries: int | None = None) -> None:
    """Call write(text_handle) on a unique temp file next to `path`, fsync, then replace `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # suffix .tmp: readers glob '*.json' / '*.yaml' and never see a half-written file
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        replace(tmp, path, replace_tries)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def read_text(path, tries: int = 5, step: float = 0.05) -> str:
    """read_text, retried briefly: on Windows opening a file that is being replaced can raise PermissionError."""
    for i in range(tries):
        try:
            return Path(path).read_text(encoding="utf-8")
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(step)
