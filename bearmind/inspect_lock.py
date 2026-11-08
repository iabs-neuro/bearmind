"""The inspection server's lock as other processes read it: is a server of the series running, at which address,
and is it starting or stopping. No Bokeh, Tornado or CaImAn here: the window imports this module at its start,
and bearmind.inspect_server, which the window does not import, keeps these names for its callers.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

import psutil

# A lock that cannot be read is being written by its new holder (it empties the file, then writes itself in, at
# once); one older than this was left by a holder that died in between, and is not waited for.
UNREADABLE_LOCK_STALE_S = 2.0


def session_url(url: str, sid: str) -> str:
    return f"{url.rstrip('/')}/?sid={quote(sid, safe='')}"


@dataclass
class ServerState:
    alive: bool
    url: str | None
    pid: int | None
    current: list[str] = field(default_factory=list)  # sessions open in a browser tab
    prepared: str | None = None  # the session prepared in the background
    # no address yet: its process runs but does not listen yet, or the lock is being written (a new holder
    # empties it before it writes itself in); worth waiting for
    starting: bool = False
    stopping: bool = False  # its process runs, but no longer serves pages (it finishes the side files)

    @property
    def available(self) -> bool:
        """A server of the series answers at url."""
        return self.alive and self.url is not None and not self.stopping


def _lock_info(lock: Path, tries: int = 5) -> dict | None:
    """The lock's content; {} without a lock file; None when it exists but cannot be read as a lock (its holder
    is writing it, or nobody will)."""
    for _ in range(tries):
        try:
            info = json.loads(lock.read_text(encoding="utf-8"))
            return info if isinstance(info, dict) else None
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            time.sleep(0.02)  # its server is rewriting it: look again
    return None


def same_process(pid, created) -> bool:
    """The process `pid` is alive and started at `created` (a reused PID has another start time)."""
    if not isinstance(pid, int) or isinstance(pid, bool) or created is None:
        return False
    try:
        return abs(psutil.Process(pid).create_time() - float(created)) < 1.0
    except (psutil.Error, TypeError, ValueError):
        return False


def server_state(paths) -> ServerState:
    """The inspection server of the series as its lock describes it; alive: its process runs. Use available
    before opening its address; starting and stopping mean "look again shortly" (wait_ready)."""
    info = _lock_info(paths.inspection_lock)
    if info is None:
        try:
            age = time.time() - paths.inspection_lock.stat().st_mtime
        except OSError:
            age = 0.0  # gone or held right now: look again
        return ServerState(alive=False, url=None, pid=None, starting=age < UNREADABLE_LOCK_STALE_S)
    pid, url, cur, prep = info.get("pid"), info.get("url"), info.get("current"), info.get("prepared")
    alive = same_process(pid, info.get("create_time"))
    url = url if isinstance(url, str) else None
    stopping = alive and bool(info.get("stopping"))
    return ServerState(alive=alive, url=url, pid=pid if isinstance(pid, int) else None,
                       current=[s for s in cur if isinstance(s, str)] if isinstance(cur, list) else [],
                       prepared=prep if isinstance(prep, str) else None,
                       starting=alive and url is None and not stopping, stopping=stopping)
