"""One inspection server per series: the manual-inspection window of the sessions that wait at the gate.

python -m bearmind inspect SERIES [SID] starts it, or reuses the running one. It is one long-lived Bokeh server
per series, so CaImAn and driada are imported once per series, not once per session. Its lock is
_bearmind/inspect.lock: the runner's OS-lock mechanism on a file of its own, so a server runs while a batch runs
and neither waits for the other. A browser tab opens one session at /?sid=<SID>; the page without sid lists
the waiting sessions as links. While a session is open, the next waiting one is prepared in a background thread,
so Next after Save results only builds the window. Save results writes the session's manual result
(SeriesPaths.manual_result) and records the gate decision "cleaned": the batch then picks the session up for
autoinspection. A save is refused when CNMF-E was recomputed after the session was opened. Without an open
browser tab for inspection.idle_exit_s the server stops and removes its lock.

Page texts are English and name only an exception's type and file: a Windows error text is localized, and the
series log keeps the whole text. Console lines are ASCII.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import weakref
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from urllib.parse import quote

import numpy as np
import psutil
from bokeh.document import without_document_lock
from bokeh.layouts import column, row
from bokeh.models import Button, Div
from bokeh.server.server import Server
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.web import OutputTransform

from bearmind import actions, config, naming, provenance, registry, runner
# the lock as other processes read it is in a module without Bokeh (the window imports it); its names stay here
from bearmind.inspect_lock import (UNREADABLE_LOCK_STALE_S, ServerState, _lock_info,  # noqa: F401
                                   same_process as _same_process, server_state, session_url)
from bearmind.paths import SeriesPaths
from bearmind.registry import State
from bearmind.stages import gate
from bearmind.timestamps import session_timestamps

log = logging.getLogger("bearmind.inspect")

OK, BAD_INPUT, BUSY = 0, 2, 3

# --- texts: the page's (English) and the console's (ASCII) ---
TITLE = "{sid} - manual inspection"
INDEX_TITLE = "Manual inspection"
INDEX_HEAD = "Sessions waiting for manual inspection:"
NONE_WAITING = "No session waits for manual inspection."
NEXT_LABEL = "Next: {sid} →"
NO_MORE = "No more sessions wait for manual inspection."
OPENING = "Opening {sid}..."
SAVED = "Saved. Decision “cleaned” recorded for {sid}. You can close this tab."
OUTDATED = "Result is outdated: CNMF-E was recomputed after this session was opened. Open it again."
SAVE_FAILED = "Save failed: {what}. Details in the series log."
DECISION_FAILED = "Result file written, but the decision could not be recorded: {what}. Save again."
PROGRESS_SAVED = "Progress saved: {name}"
SIDE_FILES_FAILED = ("Side files of {sid} not all written: {what}. The decision stays; details in the series "
                     "log.")
SECOND_TAB = "{sid} is already open in another browser tab."
NOT_DONE = "CNMF-E of {sid} is not done yet."
UNKNOWN = "Unknown session {sid}."
SHORT_TIMESTAMPS = ("{sid} cannot be opened: it has {n_ts} timestamps for {n_frames} frames of traces. Check its "
                    "timestamps file, then rescan the series.")
CANNOT_OPEN = "{sid} cannot be opened: {what}. Details in the series log."

STARTED = "[OK] inspection server: {url}"
ALREADY_RUNNING = "[FAILED] an inspection server of this series is already running: {url}"
NOT_DONE_CONSOLE = "[FAILED] CNMF-E of {sid} is not done yet"

DEFAULT_FPS = 20  # timestamps come from the registry; make_app needs a value, which the first view ignores
TICK_MS = 500  # how often open tabs are counted and the lock is updated
# A page whose browser has not connected yet (the page is loading) holds its session this long; once connected,
# it holds the session while it is connected, so a reloaded tab is not refused as a second one.
CONNECT_GRACE_S = 15.0
# Bokeh discards a session this long after its last tab closed (its document and the window's copy of the
# estimates go with it).
UNUSED_SESSION_MS = 15_000
CHECK_UNUSED_MS = 5_000
SIDE_FILES_WAIT_S = 600  # a stopping server waits this long for the side files it queued
URL_WAIT_S = 10.0  # a server that holds the lock writes its address right after it listens
STOP_WAIT_S = SIDE_FILES_WAIT_S + 60  # how long the inspect command waits for a stopping server to end
# A reload asks for the new page while the old page's websocket is still open (it closes a moment later): a
# session held by another tab is asked for again after this long, and refused only if that tab is still there.
RETRY_OPEN_MS = 1500


# --- small helpers ---

def _say(text: str) -> None:
    # paths and session names may hold letters a Windows console cannot encode
    print(str(text).encode("ascii", "backslashreplace").decode("ascii"), flush=True)


def _file_of(e) -> str | None:
    f = getattr(e, "filename2", None) or getattr(e, "filename", None)
    if isinstance(f, (str, bytes, os.PathLike)):
        return Path(os.fsdecode(f)).name
    return None


def page_error(e: BaseException) -> str:
    """What a page says of an error: its type and the file it concerns. The message itself stays out: a Windows
    error text is in the system's language, and a path can hold any letters; the series log has both."""
    x = e.__cause__ if e.__cause__ is not None else e
    f = _file_of(x) or _file_of(e)
    return f"{type(x).__name__} on {f}" if f else type(x).__name__


_bm_module = None


def _bm():
    """bm_examinator, imported on first use: it imports CaImAn and driada (about 10 s), which the CLI's other
    commands and server_state must not pay for."""
    global _bm_module
    if _bm_module is None:
        root = str(provenance.REPO_ROOT)
        if root not in sys.path:  # it imports the repo's top-level modules (config, polygon, ...)
            sys.path.insert(0, root)
        try:
            # torch's shm.dll fails to load (WinError 127) when torch comes after what bm_examinator imports first
            import torch  # noqa: F401
        except ImportError:
            pass
        import bm_examinator
        _bm_module = bm_examinator
    return _bm_module


def session_time_s(paths, rec) -> np.ndarray:
    """The session's frame times in seconds from its first frame: the stitched timestamps of the registry
    (timestamps.session_timestamps), one per frame; rec.n_frames of them when the timestamps cover every frame."""
    t = np.asarray(session_timestamps(paths, rec).t_ms, dtype=np.float64)
    return (t - t[0]) / 1000.0 if len(t) else t


# --- waiting for the series' server (its lock is read by bearmind.inspect_lock) ---

def wait_ready(paths, start_timeout: float | None = None, stop_timeout: float | None = None,
               on_stopping=None, deadline: float | None = None) -> ServerState:
    """The server state once it is settled: available, or no server at all. A starting server is waited for
    start_timeout (URL_WAIT_S), a stopping one (it ends once the side files of its last saves are written)
    stop_timeout (STOP_WAIT_S), never past deadline (time.monotonic()); on_stopping(state) is called once
    before waiting for a stopping one. After the time it returns the state as it is then (not available)."""
    start_timeout = URL_WAIT_S if start_timeout is None else start_timeout
    stop_timeout = STOP_WAIT_S if stop_timeout is None else stop_timeout
    t0 = time.monotonic()
    told = False
    while True:
        st = server_state(paths)
        if st.available or not (st.starting or st.stopping):
            return st
        if st.stopping and not told and on_stopping is not None:
            on_stopping(st)
            told = True
        limit = t0 + (stop_timeout if st.stopping else start_timeout)
        if time.monotonic() >= (limit if deadline is None else min(limit, deadline)):
            return st
        time.sleep(0.1)


def wait_url(paths, timeout: float = URL_WAIT_S) -> str | None:
    """The address of the series' running server, waiting while it starts; None when none is available."""
    st = wait_ready(paths, timeout, timeout)
    return st.url if st.available else None


# --- what a prepared session must still match when it is opened or saved ---

@dataclass(frozen=True)
class _Stamp:
    """The estimates file and the CNMF-E record a session was loaded from; any change means CNMF-E ran again."""
    size: int | None
    mtime_ns: int | None
    cnmf_state: str | None
    cnmf_finished: str | None


def _estimates_file(paths, rec) -> Path:
    return gate._estimates_file(paths, rec)  # the gate's choice: R28 compares the result with this file


def _stamp_of(paths, rec) -> _Stamp:
    try:
        st = _estimates_file(paths, rec).stat()
        size, mtime = st.st_size, st.st_mtime_ns
    except OSError:
        size = mtime = None
    cnmf = rec.stages.get("cnmf")
    return _Stamp(size, mtime, cnmf.state.value if cnmf is not None else None,
                  cnmf.finished if cnmf is not None else None)


class _NotDone(Exception):
    def __init__(self, sid):
        super().__init__(sid)
        self.sid = sid


class _ShortTimestamps(Exception):
    def __init__(self, sid, n_ts, n_frames):
        super().__init__(f"{sid}: {n_ts} timestamps for {n_frames} frames")
        self.sid, self.n_ts, self.n_frames = sid, n_ts, n_frames


# LoadEstimates' refusal of timestamps shorter than the traces; the page names the two lengths, not the path
_SHORT_RE = re.compile(r"(\d+) timestamps for (\d+) frames$")
_SID_RE = re.compile(r"[\w-]+")


# --- background preparation ---

class Preparer:
    """Prepares sessions in one background thread, one at a time, and holds at most one prepared session (an
    hour-long one is hundreds of MB).

    build(sid) -> PreparedSession with .stamp; stamp(sid) -> the current stamp. take() hands the prepared
    session over, or waits for the one being prepared, or prepares it in the caller's thread; a prepared session
    whose stamp changed meanwhile (CNMF-E ran again) is prepared again."""

    def __init__(self, build, stamp):
        self._build, self._stamp = build, stamp
        self._cv = threading.Condition()
        self._wanted: str | None = None
        self._busy: str | None = None
        self._ready: dict = {}
        self._closed = False
        self._thread = threading.Thread(target=self._work, name="bearmind-inspect-prepare", daemon=True)
        self._thread.start()

    @property
    def prepared(self) -> str | None:
        with self._cv:
            return next(iter(self._ready), None)

    def prepare(self, sid: str) -> None:
        with self._cv:
            if self._closed or sid in self._ready or sid == self._busy:
                return
            self._wanted = sid
            self._cv.notify_all()

    def take(self, sid: str):
        with self._cv:
            while self._busy == sid:
                self._cv.wait()
            if self._wanted == sid:
                self._wanted = None  # not started yet: prepared here, now
            prep = self._ready.pop(sid, None)
        if prep is not None and prep.stamp == self._stamp(sid):
            return prep
        return self._build(sid)

    def _work(self):
        while True:
            with self._cv:
                while not self._closed and self._wanted is None:
                    self._cv.wait()
                if self._closed:
                    return
                sid, self._wanted, self._busy = self._wanted, None, self._wanted
                self._ready.clear()  # one prepared session at a time
            prep = None
            try:
                prep = self._build(sid)
            except Exception:
                # take() prepares it again in the page's thread, which shows the error
                log.warning("%s: background preparation failed", sid, exc_info=True)
            with self._cv:
                self._busy = None
                if prep is not None and not self._closed:
                    self._ready = {sid: prep}
                self._cv.notify_all()

    def close(self, timeout: float = 120) -> None:
        with self._cv:
            self._closed = True
            self._ready.clear()
            self._cv.notify_all()
        self._thread.join(timeout)


# --- one browser tab ---

class _Page:
    """One browser tab: one session at a time; Next replaces the document's content in place."""

    def __init__(self, server, doc):
        self.server, self.doc = server, doc
        self.sid: str | None = None  # the session shown (or being opened)
        self.claim: str | None = None  # the session Next is opening
        self.created = time.monotonic()
        self.connected = False
        self.busy = False
        self.notes: list[str] = []  # side-file failures; kept when Next opens another session
        self.prepared = None
        self.next_button = self.next_text = self.notes_div = self.wave = None

    # -- state the server reads --
    def session(self):
        try:
            ctx = self.doc.session_context
        except Exception:
            return None
        return getattr(ctx, "session", None) if ctx is not None else None

    def holding(self) -> str | None:
        """How a browser shows this page: "connected" now, "loading" (never connected yet, for at most
        CONNECT_GRACE_S since it was asked for), or None (the tab left it)."""
        s = self.session()
        if s is not None and not s.destroyed and s.connection_count > 0:
            self.connected = True
            return "connected"
        if not self.connected and time.monotonic() - self.created < CONNECT_GRACE_S:
            return "loading"
        return None

    def is_open(self) -> bool:
        return self.holding() is not None

    # -- content --
    def show_message(self, text: str, title: str = INDEX_TITLE) -> None:
        self.sid = None
        self.doc.clear()
        self.doc.title = title
        self.doc.add_root(Div(text=html.escape(text, quote=False), name="message"))

    def show_index(self) -> None:
        waiting = self.server.waiting()
        self.doc.clear()
        self.doc.title = INDEX_TITLE
        if not waiting:
            text = html.escape(NONE_WAITING)
        else:
            links = "".join(f'<li><a href="/?sid={quote(s, safe="")}">{html.escape(s)}</a></li>' for s in waiting)
            text = f"<b>{html.escape(INDEX_HEAD)}</b><ul>{links}</ul>"
        self.doc.add_root(Div(text=text, name="index"))

    def open(self, sid: str) -> None:
        """The page of a session requested by the browser (the document has no session yet: no lock needed)."""
        srv = self.server
        # the name comes from the address bar: a session name, never a path into another folder
        if not _SID_RE.fullmatch(sid) or not srv.paths.registry_file(sid).is_file():
            return self.show_message(UNKNOWN.format(sid=sid))
        if srv.claim_status(self, sid) != "claimed":
            # most often a reload: the browser asks for the new page while the old page is still connected; or
            # the other page was left while it loaded and will never connect
            self.show_message(OPENING.format(sid=sid), TITLE.format(sid=sid))
            self.doc.add_timeout_callback(partial(self._claim_again, sid), RETRY_OPEN_MS)
            return
        try:
            prep = srv.preparer.take(sid)
        except Exception as e:
            srv.release(self)
            return self.show_message(srv.open_failure(sid, e))
        self.show(sid, prep)

    def _claim_again(self, sid: str) -> None:
        """RETRY_OPEN_MS after a refused claim (under the document lock). A tab connected to the session now is
        another tab; a page that has not connected yet is waited for until it connects (then refused) or its
        CONNECT_GRACE_S is over (then the session is taken)."""
        status = self.server.claim_status(self, sid)
        if status == "connected":
            self.show_message(SECOND_TAB.format(sid=sid))
        elif status == "loading":
            self.doc.add_timeout_callback(partial(self._claim_again, sid), RETRY_OPEN_MS)
        else:
            self.doc.add_next_tick_callback(without_document_lock(partial(self._take, sid)))

    def _take(self, sid: str) -> None:
        """Without the document lock: the session may still be prepared."""
        prep = err = None
        try:
            prep = self.server.preparer.take(sid)
        except Exception as e:
            self.server.release(self)
            err = self.server.open_failure(sid, e)
        self._next_tick(self._taken, sid, prep, err)

    def _taken(self, sid: str, prep, err) -> None:
        if err is not None:
            self.show_message(err)
            return
        try:
            self.show(sid, prep)
        except Exception as e:
            log.exception("%s: cannot open", sid)
            self.show_message(CANNOT_OPEN.format(sid=sid, what=page_error(e)))

    def show(self, sid: str, prep) -> None:
        """Replace the document's content with the window of the prepared session (under the document lock, or
        before the document has a session)."""
        srv = self.server
        self.doc.clear()
        self.sid, self.claim, self.busy, self.prepared = sid, None, False, prep
        self.doc.title = TITLE.format(sid=sid)
        self.doc.add_root(self._header(sid))
        app = _bm().make_app(prep.fname, prep.options["default_fps"], srv.bkapp_kwargs, prepared=prep,
                             final_save=partial(srv.final_save, self, sid, prep.stamp),
                             progress_save=partial(srv.progress_save, sid), refuse_autoinspected=True)
        app(self.doc)
        log.info("%s opened", sid)
        srv.after_open(self, sid)

    def _header(self, sid):
        self.next_button = Button(label=NEXT_LABEL.format(sid=""), button_type="primary", visible=False,
                                  name="next", width_policy="fit")
        self.next_button.on_event("button_click", self._on_next)
        self.next_text = Div(text="", name="next_text", visible=False)
        self.wave = Div(text="", name="wave", visible=False)  # the calcium-wave hint (task 6)
        self.notes_div = Div(text=self._notes_html(), name="notes", visible=bool(self.notes))
        return column(row(Div(text=f"<b>{html.escape(sid)}</b>", name="session"), self.next_button,
                          self.next_text), self.wave, self.notes_div, name="inspection_header")

    def _notes_html(self) -> str:
        return "<br>".join(html.escape(n, quote=False) for n in self.notes)

    def add_note(self, text: str) -> None:
        """A line on the page (under the document lock: through add_next_tick_callback)."""
        self.notes.append(text)
        if self.notes_div is not None:
            self.notes_div.text = self._notes_html()
            self.notes_div.visible = True

    def _next_tick(self, fn, *args) -> None:
        """Run fn(*args) under the document lock; nothing if the tab is gone (the series log has what matters)."""
        try:
            self.doc.add_next_tick_callback(partial(fn, *args))
        except Exception:
            log.debug("page gone, not updated", exc_info=True)

    # -- after Save results --
    def saved(self, sid: str) -> None:
        """Save results of sid succeeded (in its callback, under the document lock): offer the next session."""
        if self.next_button is not None:
            self.next_button.visible = False
        if self.next_text is not None:
            self.next_text.visible = False
        self.server.background(self._find_next, sid)

    def _find_next(self, sid: str) -> None:
        nxt = self.server.next_waiting(sid, self)
        if nxt is not None:
            self.server.preparer.prepare(nxt)
        self._next_tick(self._offer_next, sid, nxt)

    def _offer_next(self, sid: str, nxt: str | None) -> None:
        if self.sid != sid or self.busy:
            return  # the page has moved on meanwhile
        if nxt is None:
            self.next_button.visible = False
            self.next_text.text = html.escape(NO_MORE)
            self.next_text.visible = True
        else:
            self.next_button.label = NEXT_LABEL.format(sid=nxt)
            self.next_button.disabled = False
            self.next_button.visible = True
            self.next_text.visible = False

    def side_files_done(self, sid: str, future) -> None:
        """Done callback of the side-file writer (its thread): a failure goes on the page through the document's
        next tick, never from this thread; the writer has logged it with its traceback."""
        e = None if future.cancelled() else future.exception()
        if e is not None:
            self._next_tick(self.add_note, SIDE_FILES_FAILED.format(sid=sid, what=page_error(e)))

    # -- Next --
    def _on_next(self, event) -> None:
        if self.busy:
            return
        self.busy = True
        self.next_button.disabled = True
        self.doc.add_next_tick_callback(without_document_lock(self._next_work))

    def _next_work(self) -> None:
        """Without the document lock: the next session may still be prepared; the page stays responsive."""
        srv, current = self.server, self.sid
        nxt = prep = err = None
        try:
            # the waiting list now, not when the button was offered: a session decided meanwhile is skipped
            nxt = srv.next_waiting(current, self)
            if nxt is not None:
                if srv.claim(self, nxt, next_=True):
                    prep = srv.preparer.take(nxt)
                else:
                    err = SECOND_TAB.format(sid=nxt)
        except Exception as e:
            srv.release_claim(self)
            err = srv.open_failure(nxt, e) if nxt else CANNOT_OPEN.format(sid=current, what=page_error(e))
        self._next_tick(self._next_done, current, nxt, prep, err)

    def _next_done(self, current, nxt, prep, err) -> None:
        self.busy = False
        if self.sid != current:
            self.server.release_claim(self)
            return
        if err is not None or nxt is None:
            self.server.release_claim(self)
            self.next_button.visible = nxt is not None
            self.next_button.disabled = False
            self.next_text.text = html.escape(err if err is not None else NO_MORE)
            self.next_text.visible = True
            return
        try:
            self.show(nxt, prep)
        except Exception as e:
            log.exception("%s: cannot open", nxt)
            self.server.release_claim(self)
            self.show_message(CANNOT_OPEN.format(sid=nxt, what=page_error(e)))


# --- the server ---

class InspectServer:
    """serve() of one series; tests keep the object to stop it and to reach its pages."""

    def __init__(self, series_root, sid: str | None = None, port: int | None = None, open_browser: bool = True,
                 idle_exit_s: float | None = None, quiet_busy: bool = False):
        self.paths = SeriesPaths(Path(series_root).resolve())
        self.sid, self.port, self.open_browser, self.idle_exit_s = sid, port, open_browser, idle_exit_s
        self.quiet_busy = quiet_busy  # code 3 without its console line: the inspect command reuses that server
        self.url: str | None = None
        self.started = threading.Event()
        self.stopped = threading.Event()
        self._stop_requested = threading.Event()
        self._lock = threading.RLock()
        self._pages: dict = {}
        self._side_futures: set = set()  # side-file jobs not done yet (a stopping server waits for them)
        self._side_last: dict = {}  # sid -> its last job not done yet
        self._cfg = None
        self._io_loop = None
        self._lock_fh = None
        self._lock_info: dict = {}
        self._lock_len = 0
        self._last_open = time.monotonic()
        self.preparer: Preparer | None = None
        self._bg: ThreadPoolExecutor | None = None

    # -- config --
    @property
    def cfg(self) -> dict:
        if self._cfg is None:
            self._cfg = config.resolve(self.paths)["inspection"]
        return self._cfg

    @property
    def bkapp_kwargs(self) -> dict:
        c = self.cfg
        return {"mode": "legacy", "end_frame": None, "enable_gpu_backend": bool(c.get("webgl", True)),
                "trace_points": int(c.get("trace_points", 2000))}

    # -- sessions --
    def stamp(self, sid: str) -> _Stamp:
        return _stamp_of(self.paths, registry.load_record(self.paths, sid))

    def build(self, sid: str):
        """The PreparedSession of sid, with .sid and .stamp (taken before the file is read: a CNMF-E that writes
        new estimates meanwhile makes a later save refuse)."""
        paths = self.paths
        rec = registry.load_record(paths, sid)
        cnmf = rec.stages.get("cnmf")
        if cnmf is None or cnmf.state is not State.DONE:
            raise _NotDone(sid)
        stamp = _stamp_of(paths, rec)
        fname = _estimates_file(paths, rec)
        time_s = session_time_s(paths, rec)
        kwargs = self.bkapp_kwargs
        contours = self._cached_contours(sid, stamp, kwargs)
        bm = _bm()
        try:
            # forward slashes: the window titles its image with the part of the path after the last "/"
            prep = bm.prepare_session(fname.as_posix(), time=time_s, default_fps=DEFAULT_FPS, bkapp_kwargs=kwargs,
                                      contours=contours)
        except ValueError as e:
            m = _SHORT_RE.search(str(e))
            if m:
                raise _ShortTimestamps(sid, int(m.group(1)), int(m.group(2))) from e
            raise
        if contours is None and prep.contours is not None:
            self._write_contours(sid, stamp, kwargs, prep.contours)
        prep.sid, prep.stamp = sid, stamp
        return prep

    def _cached_contours(self, sid, stamp, kwargs):
        """Contours of an earlier preparation of the same estimates file (size, mtime) and cthr, or None."""
        f = self.paths.inspection_cache(sid)
        if stamp.size is None or not f.is_file():
            return None
        try:
            with np.load(f, allow_pickle=False) as z:
                key = (int(z["size"]), int(z["mtime_ns"]), float(z["cthr"]))
                if key != (stamp.size, stamp.mtime_ns, float(kwargs.get("cthr", 0.3))):
                    return None
                coords, lengths = z["coords"], z["lengths"]
        except Exception:
            log.warning("%s: contour cache not readable, computed again", sid, exc_info=True)
            return None
        return list(np.split(coords, np.cumsum(lengths)[:-1])) if len(lengths) else []

    def _write_contours(self, sid, stamp, kwargs, contours) -> None:
        f = self.paths.inspection_cache(sid)
        tmp = f.with_name(f.name + ".partial")
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            lengths = np.array([len(c) for c in contours], dtype=np.int64)
            coords = (np.concatenate([np.asarray(c, dtype=np.float64) for c in contours])
                      if contours else np.zeros((0, 2)))
            with open(tmp, "wb") as fh:
                np.savez(fh, size=np.int64(stamp.size), mtime_ns=np.int64(stamp.mtime_ns),
                         cthr=np.float64(kwargs.get("cthr", 0.3)), coords=coords, lengths=lengths)
            os.replace(tmp, f)
        except Exception:
            # only a cache: the next preparation computes the contours again
            log.warning("%s: contour cache not written", sid, exc_info=True)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def open_failure(self, sid: str, e: BaseException) -> str:
        if isinstance(e, _NotDone):
            return NOT_DONE.format(sid=sid)
        if isinstance(e, _ShortTimestamps):
            log.error("%s cannot be opened: %s", sid, e, exc_info=e)
            return SHORT_TIMESTAMPS.format(sid=sid, n_ts=e.n_ts, n_frames=e.n_frames)
        log.error("%s cannot be opened", sid, exc_info=e)
        return CANNOT_OPEN.format(sid=sid, what=page_error(e))

    def waiting(self) -> list[str]:
        """Sessions waiting for manual inspection, in Monitor's order (mouse, then session)."""
        return sorted(actions.waiting_gate_sids(self.paths), key=lambda s: (naming.mouse_id(s), s))

    def next_waiting(self, after: str | None, page=None) -> str | None:
        """The first waiting session after `after` in Monitor's order (from the start when none is after it),
        not `after` itself and not one another tab holds."""
        with self._lock:
            held = {s for p in self._pages.values() if p is not page and p.is_open() for s in (p.sid, p.claim) if s}
        order = [s for s in self.waiting() if s != after and s not in held]
        if after is None:
            return order[0] if order else None
        key = (naming.mouse_id(after), after)
        later = [s for s in order if (naming.mouse_id(s), s) > key]
        return (later or order or [None])[0]

    # -- tabs --
    def claim_status(self, page, sid: str, next_: bool = False) -> str:
        """page holds sid ("claimed") unless another page does: "connected" when one of them is connected,
        "loading" when they have only not connected yet (Page.holding). The check and the claim are one step."""
        with self._lock:
            loading = False
            for p in self._pages.values():
                if p is page or sid not in (p.sid, p.claim):
                    continue
                state = p.holding()
                if state == "connected":
                    return "connected"
                loading = loading or state == "loading"
            if loading:
                return "loading"
            if next_:
                page.claim = sid
            else:
                page.sid = sid
            return "claimed"

    def claim(self, page, sid: str, next_: bool = False) -> bool:
        """page holds sid unless another open tab does (False then)."""
        return self.claim_status(page, sid, next_) == "claimed"

    def release(self, page) -> None:
        with self._lock:
            page.sid = None

    def release_claim(self, page) -> None:
        with self._lock:
            page.claim = None

    def pages(self) -> list:
        with self._lock:
            return list(self._pages.values())

    def page_of(self, session_id: str):
        with self._lock:
            return self._pages.get(session_id)

    def _make_doc(self, doc) -> None:
        """A browser asked for a page. Bokeh builds documents in a worker thread, so a slow preparation does not
        hold up the other tabs. Never a dead page: every failure becomes a message."""
        page = _Page(self, doc)
        doc.template = _page_template()
        ctx = doc.session_context
        key = ctx.id if ctx is not None else str(id(page))
        with self._lock:
            self._pages[key] = page

        def destroyed(session_context):
            with self._lock:
                self._pages.pop(session_context.id, None)
        doc.on_session_destroyed(destroyed)
        try:
            sid = _sid_argument(doc)
            if not sid:
                page.show_index()
            else:
                page.open(sid)
        except Exception as e:
            log.exception("page not built")
            try:
                page.show_message(CANNOT_OPEN.format(sid=_sid_argument(doc) or "the page", what=page_error(e)))
            except Exception:
                log.exception("message page not built")

    def after_open(self, page, sid: str) -> None:
        self.background(lambda: self._prepare_after(page, sid))

    def _prepare_after(self, page, sid):
        nxt = self.next_waiting(sid, page)
        if nxt is not None:
            self.preparer.prepare(nxt)

    def background(self, fn, *args) -> None:
        """Small jobs off the pages' threads (the waiting list reads every registry file)."""
        bg = self._bg
        if bg is None:
            return
        try:
            bg.submit(self._logged, fn, *args)
        except RuntimeError:
            pass  # stopping

    @staticmethod
    def _logged(fn, *args):
        try:
            fn(*args)
        except Exception:
            log.exception("background job failed")

    # -- saves --
    def final_save(self, page, sid: str, opened: _Stamp, est) -> str:
        """Save results: the result, then the decision "cleaned"; side files in the background. Never raises:
        the page shows the returned text."""
        try:
            now = self.stamp(sid)
        except Exception as e:
            log.exception("%s: save not checked, registry not read", sid)
            return SAVE_FAILED.format(what=page_error(e))
        if now != opened:
            log.warning("%s: result not saved, CNMF-E was recomputed after the session was opened (%s -> %s)",
                        sid, opened, now)
            _say(f"[WARN] {sid}: result not saved; CNMF-E was recomputed after the session was opened")
            return OUTDATED
        bm = _bm()
        try:
            bm.write_result(est, self.paths.manual_result(sid))
        except Exception as e:
            log.exception("%s: save failed", sid)
            return SAVE_FAILED.format(what=page_error(e))
        # the result is on disk, but without the decision the batch does not take it: the person saves again
        try:
            res = actions.gate_cleaned(self.paths, sid, actions.current_user())
        except Exception as e:
            log.exception("%s: result written, decision not recorded", sid)
            return DECISION_FAILED.format(what=page_error(e))
        if sid in res.failed:
            code, detail = res.failed[sid]
            log.error("%s: result written, decision not recorded: %s %s", sid, code, detail)
            return DECISION_FAILED.format(what=code)
        log.info("%s: manual result saved, decision cleaned recorded", sid)
        _say(f"[OK] {sid}: result saved, decision cleaned recorded")
        self._queue_side_files(page, sid, est)
        page.saved(sid)
        return SAVED.format(sid=sid)

    def _queue_side_files(self, page, sid: str, est) -> None:
        """The side files of this save after those already being written. A queued job of an earlier save of the
        session is dropped: only the last save's files are kept, and each job holds the large arrays."""
        bm = _bm()
        with self._lock:
            earlier = self._side_last.get(sid)
            try:
                future = bm.write_side_files_in_background(est, self.paths.inspection_dir(sid), f"{sid}_")
            except Exception as e:  # the executor is shut down: the result and the decision stand
                log.exception("%s: side files not queued", sid)
                page.add_note(SIDE_FILES_FAILED.format(sid=sid, what=page_error(e)))
                return  # the earlier save's job stays: its files are the newest that will be written
            self._side_futures.add(future)
            self._side_last[sid] = future
            if earlier is not None and earlier.cancel():
                log.info("%s: side files of an earlier save dropped, this save replaces them", sid)
        # the page through a weak reference: a finished Future keeps its callbacks, and a closed tab's page holds
        # its document and the session's data
        future.add_done_callback(partial(self._side_files_done, weakref.ref(page), sid))

    def _side_files_done(self, page_ref, sid: str, future) -> None:
        """Done callback (the writer's thread, or here when cancelled): forget the job; a failure goes on the
        page if the tab is still there."""
        with self._lock:
            self._side_futures.discard(future)
            if self._side_last.get(sid) is future:
                del self._side_last[sid]
        page = page_ref()
        if page is not None:
            page.side_files_done(sid, future)

    def progress_save(self, sid: str, est) -> str:
        """Save progress: a copy in 5_Inspection/<SID>/progress, the date in its name; no decision."""
        name = f"{sid}_{datetime.now():%Y-%m-%d_%H-%M-%S}_estimates.pickle"
        try:
            _bm().write_result(est, self.paths.inspection_progress_dir(sid) / name)
        except Exception as e:
            log.exception("%s: progress not saved", sid)
            return SAVE_FAILED.format(what=page_error(e))
        log.info("%s: progress saved: %s", sid, name)
        return PROGRESS_SAVED.format(name=name)

    # -- running --
    def request_stop(self) -> None:
        """Stop the server from any thread."""
        self._stop_requested.set()
        loop = self._io_loop
        if loop is not None:
            loop.add_callback(loop.stop)

    def run(self) -> int:
        """0 normal exit; 2 bad input; 3 already running."""
        try:
            fh = runner._take_lock(self.paths.inspection_lock, {"url": None, "current": [], "prepared": None})
            if fh is None:
                if not self.quiet_busy:
                    _say(ALREADY_RUNNING.format(url=wait_url(self.paths) or "address unknown"))
                return BUSY
            self._lock_fh = fh
            self._lock_len = fh.tell()
            me = psutil.Process()
            self._lock_info = {"pid": me.pid, "create_time": me.create_time(),
                               "time": datetime.now().isoformat(timespec="seconds"), "url": None, "current": [],
                               "prepared": None}
            try:
                code = self._check_start()
                if code:
                    return code
                return self._serve()
            finally:
                # On Windows the file cannot be deleted while a reader has it open, and it would stay with this
                # (live) process in it: "no holder" first, so nobody waits for it
                self._rewrite_lock(pid=None, create_time=None, url=None, stopping=False, current=[], prepared=None)
                self._lock_fh = None
                runner._drop_lock(self.paths.inspection_lock, fh)
        finally:
            self.stopped.set()

    def _check_start(self) -> int:
        try:
            self.cfg  # noqa: B018  a broken series config is named before anything listens
        except Exception as e:
            _say(f"[FAILED] the series config cannot be read: {actions.config_problem(self.paths, e)}")
            return BAD_INPUT
        if self.sid is None:
            return OK
        try:
            rec = registry.load_record(self.paths, self.sid)
        except FileNotFoundError:
            _say(f"[FAILED] unknown session {self.sid}")
            return BAD_INPUT
        cnmf = rec.stages.get("cnmf")
        if cnmf is None or cnmf.state is not State.DONE:
            _say(NOT_DONE_CONSOLE.format(sid=self.sid))
            return BAD_INPUT
        return OK

    def _listen(self, io_loop) -> Server:
        """Server on localhost; port 0 takes a free one. The websocket origins name the port, so a free port is
        found first (and found again if another program takes it before the server binds it)."""
        want = int(self.cfg.get("port", 0) if self.port is None else self.port)
        for attempt in range(5):
            port = want or _free_port()
            try:
                return Server({"/": self._make_doc}, io_loop=io_loop, port=port, address="localhost",
                              allow_websocket_origin=[f"localhost:{port}", f"127.0.0.1:{port}"],
                              check_unused_sessions_milliseconds=CHECK_UNUSED_MS,
                              unused_session_lifetime_milliseconds=UNUSED_SESSION_MS,
                              # passed on to Tornado's Application: every response goes through them
                              transforms=[_NoStorePages])
            except OSError as e:
                if want or attempt == 4:
                    raise OSError(f"cannot listen on port {port}") from e
        raise AssertionError("unreachable")

    def _serve(self) -> int:
        handler = _attach_series_log(self.paths)
        level = log.level
        log.setLevel(logging.INFO)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        io_loop = IOLoop.current()
        server = tick = None
        self.preparer = Preparer(self.build, self.stamp)
        self._bg = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bearmind-inspect-bg")
        try:
            try:
                server = self._listen(io_loop)
            except OSError as e:
                # the port is in use (or not allowed): the cause's text is the system's, in its language
                _say(f"[FAILED] {e}: {type(e.__cause__).__name__}; give another port with --port")
                log.error("cannot listen", exc_info=True)
                return BAD_INPUT
            self.url = f"http://localhost:{server.port}"
            self._rewrite_lock(url=self.url)
            server.start()
            idle = self.cfg.get("idle_exit_s", 600) if self.idle_exit_s is None else self.idle_exit_s
            self._idle_s = float(idle)
            self._last_open = time.monotonic()
            tick = PeriodicCallback(self._tick, TICK_MS)
            tick.start()
            if self.sid:
                self.preparer.prepare(self.sid)
            else:
                self.background(_bm)  # CaImAn and driada load while the browser shows the list of sessions
            log.info("inspection server started: %s", self.url)
            _say(STARTED.format(url=self.url))
            if self.open_browser:
                webbrowser.open(session_url(self.url, self.sid) if self.sid else self.url + "/")
            self._io_loop = io_loop
            self.started.set()
            if not self._stop_requested.is_set():
                io_loop.start()
            return OK
        finally:
            self._io_loop = None
            if tick is not None:
                tick.stop()
            # no address any more, but the lock stays until the side files are written: the inspect command waits
            # for this server to end instead of opening an address that no longer answers
            self._rewrite_lock(url=None, stopping=True, current=[], prepared=None)
            if server is not None:
                server.stop()
            loop.run_until_complete(loop.shutdown_default_executor())
            self._bg.shutdown(wait=True)
            self.preparer.close()
            with self._lock:
                pending = list(self._side_futures)
                self._pages.clear()
            # the side files of the last saves are finished before the lock goes: a new server of the series
            # must not write the same files at the same time
            wait_futures(pending, timeout=SIDE_FILES_WAIT_S)
            io_loop.close()
            asyncio.set_event_loop(None)
            log.info("inspection server stopped")
            log.setLevel(level)
            _detach_series_log(handler)

    def _tick(self) -> None:
        """Every TICK_MS on the IOLoop: idle exit, and the open sessions and the prepared one in the lock."""
        if self._stop_requested.is_set():
            self._io_loop.stop()
            return
        now = time.monotonic()
        with self._lock:
            open_pages = [p for p in self._pages.values() if p.is_open()]
        if open_pages:
            self._last_open = now
        elif now - self._last_open >= self._idle_s:
            log.info("no open browser tab for %.0f s: stopping", self._idle_s)
            _say(f"[OK] inspection server stopped: no open browser tab for {self._idle_s:g} s")
            self._io_loop.stop()
            return
        current = sorted({p.sid for p in open_pages if p.sid})
        prepared = self.preparer.prepared if self.preparer is not None else None
        if current != self._lock_info.get("current") or prepared != self._lock_info.get("prepared"):
            self._rewrite_lock(current=current, prepared=prepared)

    def _rewrite_lock(self, **fields) -> None:
        """New text in the held lock file, in place: the OS lock stays (it covers a byte past the text). A shorter
        text is padded with spaces, so a reader never finds the end of a longer old one after it."""
        fh = self._lock_fh
        if fh is None:
            return
        self._lock_info.update(fields)
        data = json.dumps(self._lock_info).encode("utf-8")
        try:
            fh.seek(0)
            fh.write(data + b" " * max(0, self._lock_len - len(data)))
            self._lock_len = max(self._lock_len, len(data))
        except OSError:
            log.warning("inspection lock not updated", exc_info=True)


class _NoStorePages(OutputTransform):
    """Cache-Control: no-store on the HTML pages. A page kept in the browser's back-forward cache keeps its
    websocket, and with it its session, open after the tab has left it: the session would stay held (a second
    tab refused, Next skipping it) and the server would never be idle."""

    def transform_first_chunk(self, status_code, headers, chunk, finishing):
        if headers.get("Content-Type", "").startswith("text/html"):
            headers["Cache-Control"] = "no-store"
        return status_code, headers, chunk


# Before BokehJS, which opens the websocket: a browser that keeps a page in its back-forward cache anyway (Chrome
# may, despite no-store) gets its sockets closed when the page is put there, and a page shown again from there
# is loaded anew, so it asks for its session again.
_PAGE_SCRIPT = """<script>
(function () {
  const Native = window.WebSocket, sockets = [];
  window.WebSocket = class extends Native {
    constructor(...args) { super(...args); sockets.push(this); }
  };
  window.addEventListener("pagehide", (event) => {
    if (event.persisted) for (const s of sockets) { try { s.close(); } catch (e) { } }
  });
  window.addEventListener("pageshow", (event) => { if (event.persisted) location.reload(); });
})();
</script>"""

_template = None


def _page_template():
    """Bokeh's page with _PAGE_SCRIPT in its head, before Bokeh's own scripts."""
    global _template
    if _template is None:
        from bokeh.core.templates import get_env
        _template = get_env().from_string('{% extends "file.html.jinja" %}{% block preamble %}{% raw %}'
                                          + _PAGE_SCRIPT + '{% endraw %}{% endblock %}')
    return _template


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _sid_argument(doc) -> str:
    ctx = doc.session_context
    req = getattr(ctx, "request", None) if ctx is not None else None
    values = (req.arguments if req is not None else {}).get("sid") or []
    v = values[0] if values else ""
    return (v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)).strip()


_LOGGERS = ("bearmind.inspect", "bm_examinator")  # the server's and the window's (save and side-file failures)


def _attach_series_log(paths) -> logging.Handler | None:
    try:
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(paths.series_log, encoding="utf-8")
    except OSError:
        _say(f"[WARN] the series log cannot be written: {paths.series_log}")
        return None
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    for name in _LOGGERS:
        logging.getLogger(name).addHandler(handler)
    return handler


def _detach_series_log(handler) -> None:
    if handler is None:
        return
    for name in _LOGGERS:
        logging.getLogger(name).removeHandler(handler)
    handler.close()


def serve(series_root: Path, sid: str | None = None, port: int | None = None, open_browser: bool = True,
          idle_exit_s: float | None = None, *, quiet_busy: bool = False) -> int:
    """Run the inspection server of the series in this thread until it is idle (or Ctrl+C).
    0 normal exit; 2 bad input; 3 already running (with quiet_busy without its console line)."""
    return InspectServer(series_root, sid, port, open_browser, idle_exit_s, quiet_busy).run()
