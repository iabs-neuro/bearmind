"""Safe writes of files that several processes read and write (registry, configs, heartbeat, summary).

Windows specifics: os.replace fails with PermissionError while another process has the target open
(a GUI reading the registry, Excel holding summary.csv), so replaces are retried; the temp file name is
unique per writer, so two writers never share it.
"""
from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_TIMEOUT_S = 10.0  # wait for another writer at most this long
LOCK_STALE_S = 60.0  # a lock older than this was left by a dead writer (writes take milliseconds)
LOCK_STEP_S = 0.02
REPLACE_TRIES = 20
REPLACE_STEP_S = 0.25


@contextmanager
def file_lock(target):
    """Short exclusive lock `<target>.lock` (O_CREAT | O_EXCL) around a read-modify-write of `target`."""
    target = Path(target)
    lock = target.with_name(target.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + LOCK_TIMEOUT_S
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except FileNotFoundError:
                continue  # released between the two calls
            if age > LOCK_STALE_S:
                try:
                    lock.unlink()
                except (FileNotFoundError, PermissionError):
                    pass
                continue
        except PermissionError:
            pass  # Windows: the previous holder's lock file is still being deleted
        else:
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"lock is held by another writer: {lock}")
        time.sleep(LOCK_STEP_S)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def replace(src, dst) -> None:
    """os.replace, retried while the target is open in another process (PermissionError on Windows)."""
    for i in range(REPLACE_TRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == REPLACE_TRIES - 1:
                raise
            time.sleep(REPLACE_STEP_S)


def write_atomic(path, write, newline=None) -> None:
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
        replace(tmp, path)
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
