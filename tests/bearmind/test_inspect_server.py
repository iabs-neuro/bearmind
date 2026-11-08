"""The inspection server of a series (bearmind.inspect_server) and the inspect command.

Servers run in threads on port 0; a browser tab is a bokeh.client session (a websocket that pulls the page's
document). Every test stops its server and joins its thread in `finally`, so a failing test leaves no port or
thread behind. The series folder has Cyrillic letters: no page text may carry a path."""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll fails to load (WinError 127)
# unless torch is imported first.
import torch  # noqa: F401

import asyncio
import gc
import html
import json
import os
import pickle
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
import weakref
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pytest
from bokeh.client import pull_session
from bokeh.events import ButtonClick
from bokeh.models import Button, ColumnDataSource, Div
from tornado.ioloop import IOLoop

import bm_examinator
from bearmind import actions, cli, config, inspect_server, monitor, runner, scan, side_files, wave
from bearmind.paths import SeriesPaths
from bearmind.registry import State, load_record, update_record
from bearmind.stages import gate
from tests.bearmind.current_stages import make_current, mark_current
from tests.examinator_helpers import make_estimates

S1, S2, S3 = "NOF_H01_1D", "NOF_H02_1D", "NOF_H03_1D"
REPO = Path(__file__).resolve().parents[2]
CYRILLIC = re.compile("[Ѐ-ӿ]")
DENIED = "Отказано в доступе"  # localized WinError 5


# --- fixtures and helpers ---

@pytest.fixture
def series_est(make_series, tmp_path):
    """Three sessions of 300 frames with CNMF-E done and synthetic estimates in 4_Estimates, all waiting at the
    gate; the series folder has Cyrillic letters."""
    root = tmp_path / "серия 1"
    for sid in (S1, S2, S3):
        make_series(root, sid=sid, chunk_frames=100)  # 3 x 100 frames: the length of make_estimates' traces
    scan.scan_series(root)
    paths = SeriesPaths(root)
    finished = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="milliseconds")
    for sid in (S1, S2, S3):
        folder = tmp_path / f"t4_est_{sid}"
        folder.mkdir()
        est = paths.estimates(sid)
        est.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(make_estimates(folder)), est)

        def done(rec, est=est):
            st = rec.stages["cnmf"]
            st.state, st.finished, st.outputs = State.DONE, finished, {"estimates": paths.rel(est)}
        update_record(paths, sid, done)
        # up to date by the batch's rule: the server lists and saves only such sessions (I3)
        make_current(paths, sid, "crop", "mc", "thresholds", "cnmf")
    assert actions.waiting_gate_sids(paths) == [S1, S2, S3]
    return paths


def _set_cnmf(paths, sid, **fields):
    def change(rec):
        for k, v in fields.items():
            setattr(rec.stages["cnmf"], k, v)
    update_record(paths, sid, change)


@contextmanager
def running(paths, sid=None, idle_exit_s=600):
    srv = inspect_server.InspectServer(paths.root, sid=sid, port=0, open_browser=False, idle_exit_s=idle_exit_s)
    srv.result = {}
    t = threading.Thread(target=lambda: srv.result.setdefault("code", srv.run()), name="t4-inspect-server")
    t.start()
    try:
        assert srv.started.wait(120) and srv.url, srv.result
        yield srv
    finally:
        srv.request_stop()
        t.join(180)
        assert not t.is_alive(), "server thread still running"


@contextmanager
def tab(srv, sid=None):
    """A browser tab on the session (its websocket stays open until the block ends). Its document is the page
    as pulled; later changes are read on the server's side (bokeh.client drops patches that arrive while it
    waits for a reply, so its copy does not follow them)."""
    # Its own loop, closed afterwards with its selector thread. The thread's asyncio loop while the tab is open:
    # the client writes to its websocket outside the running loop (close), and asyncio has no loop to give then.
    io = IOLoop(make_current=False)
    _TAB_LOOPS.append(io.asyncio_loop)
    asyncio.set_event_loop(io.asyncio_loop)
    try:
        session = pull_session(url=srv.url + "/", io_loop=io, arguments={"sid": sid} if sid else None)
        try:
            if sid:
                _settle(srv.page_of(session.id), sid)
            yield session
        finally:
            session.close()
    finally:
        _TAB_LOOPS.pop()
        asyncio.set_event_loop(_TAB_LOOPS[-1] if _TAB_LOOPS else None)
        io.close(all_fds=True)


_TAB_LOOPS = []  # the loops of the open tabs, innermost last

# Wall-clock bounds: the fast suite runs with loose ones (at least 3 times the time measured on an idle machine:
# the suite also runs on a loaded one), each still well below the time the behaviour under test must not take;
# the strict targets run with the slow tests.
STRICT = [False, pytest.param(True, marks=pytest.mark.slow, id="strict")]


def until(cond, timeout=30.0, step=0.01):
    deadline = time.monotonic() + timeout
    while True:
        value = cond()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError("condition not met in time")
        time.sleep(step)


def locked(srv, page, fn):
    """fn() under the page's document lock, as Bokeh runs a browser's click."""
    session = page.session()
    future = asyncio.run_coroutine_threadsafe(session.with_document_locked(fn), srv._io_loop.asyncio_loop)
    return future.result(120)


def _models(doc, kind=object):
    return [m for root in doc.roots for m in root.references() if isinstance(m, kind)]


def _button(doc, label):
    found = [b for b in _models(doc, Button) if b.label == label]
    assert len(found) == 1, [b.label for b in _models(doc, Button)]
    return found[0]


def _div(doc, name):
    found = [d for d in _models(doc, Div) if d.name == name]
    assert len(found) == 1, name
    return found[0]


def click(doc, label):
    b = _button(doc, label)
    b._trigger_event(ButtonClick(b))


def delete_row(doc, row):
    """Select one row of the list on screen and click Delete sel."""
    src, = [s for s in _models(doc, ColumnDataSource) if "idx" in s.data]
    rows, = {r.view.filter for r in _models(doc) if getattr(r, "data_source", None) is src}
    src.selected.indices = [rows.indices[row]]
    click(doc, "Delete sel.")


def save(srv, page):
    """Save results as the browser clicks it; returns the status text."""
    return locked(srv, page, lambda: (click(page.doc, "Save results"), _div(page.doc, "status").text)[1])


def _decisions(paths, sid):
    return [d.value for d in load_record(paths, sid).decisions if d.what == "manual_inspection"]


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _texts(doc):
    """Everything a page shows as text: its title, the Divs and the button labels."""
    return [doc.title] + [d.text for d in _models(doc, Div)] + [b.label for b in _models(doc, Button)]


def _opened(page, sid):
    return page.sid == sid and any(b.label == "Save results" for b in _models(page.doc, Button))


def _settle(page, sid, timeout=120):
    """Wait while the page is opening sid for itself: a session that is not prepared is built after the page
    answered with "Opening" (K1c). A page refused the session (another tab holds it) does not hold it and is
    not waited for; its own retry goes on."""
    # opened: the window, or the text the window shows instead (an autoinspected input), below the header
    until(lambda: page.sid != sid or _opened(page, sid) or len(page.doc.roots) >= 2, timeout=timeout)


# --- tests ---

def test_session_time_s(make_series, tmp_path):
    root = make_series(tmp_path / "s")  # bonsai: 30 frames, 500000 ticks (50 ms) apart
    rec = scan.scan_series(root)[0]
    t = inspect_server.session_time_s(SeriesPaths(root), rec)
    assert len(t) == rec.n_frames == 30 and t[0] == 0.0
    assert np.allclose(t, np.arange(30) * 0.05)


def test_session_url():
    assert inspect_server.session_url("http://localhost:5006", S1) == f"http://localhost:5006/?sid={S1}"
    assert inspect_server.session_url("http://localhost:5006/", S1) == f"http://localhost:5006/?sid={S1}"


def test_refuses_without_cnmf(series_est, capsys):
    _set_cnmf(series_est, S2, state=State.PENDING)
    code = inspect_server.serve(series_est.root, S2, port=0, open_browser=False, idle_exit_s=5)
    assert code == 2
    assert f"[FAILED] CNMF-E of {S2} is not done yet" in capsys.readouterr().out
    assert not series_est.inspection_lock.exists()


def test_server_serves_page(series_est):
    with running(series_est) as srv:
        with urllib.request.urlopen(inspect_server.session_url(srv.url, S1), timeout=120) as r:
            assert r.status == 200
            body = r.read().decode("utf-8", "replace")
        assert "bokeh" in body.lower()
        st = inspect_server.server_state(series_est)
        assert st.alive and st.url == srv.url and st.pid == os.getpid()
    assert srv.result["code"] == 0
    assert not series_est.inspection_lock.exists()
    assert not inspect_server.server_state(series_est).alive


def test_final_save_records_cleaned(series_est):
    paths = series_est
    estimates_before = sorted(os.listdir(paths.root / "4_Estimates"))
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert _opened(page, S1) and page.doc.title == inspect_server.TITLE.format(sid=S1)
        locked(srv, page, lambda: delete_row(page.doc, 2))
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        result = paths.manual_result(S1)
        saved = _load(result)
        assert not hasattr(saved, "idx_components_bad_original") and saved.C.dtype == np.float64
        assert list(saved.idx_components_bad) == [2] and saved.A.shape[1] == 8
        assert _decisions(paths, S1) == ["cleaned"]
        assert load_record(paths, S1).decisions[-1].who == actions.current_user()
        # the batch takes it: the gate passes with this decision
        cfg = config.resolve(paths, S1)
        assert gate.run(paths, load_record(paths, S1), cfg).metrics["gate"] == "cleaned"
        assert S1 not in actions.waiting_gate_sids(paths)
        # Next is offered once the waiting list is read
        nxt = until(lambda: locked(srv, page, lambda: page.next_button.visible and page.next_button.label))
        assert nxt == inspect_server.NEXT_LABEL.format(sid=S2)
    assert sorted(os.listdir(paths.root / "4_Estimates")) == estimates_before
    # side files of the save, written before the server stopped
    assert sorted(os.listdir(paths.inspection_dir(S1))) == sorted(
        [result.name, f"{S1}_filters", f"{S1}_filters_session.mat", f"{S1}_traces.csv", f"{S1}_inspect_cache.npz"])


def test_second_final_save_overwrites(series_est):
    paths = series_est
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        locked(srv, page, lambda: delete_row(page.doc, 0))
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
    folder = paths.inspection_dir(S1)
    assert [p.name for p in folder.glob("*manual*")] == [paths.manual_result(S1).name]
    assert not list(folder.glob("*.partial"))
    assert list(_load(paths.manual_result(S1)).idx_components_bad) == [0]  # the last save's
    assert _decisions(paths, S1) == ["cleaned", "cleaned"]


@pytest.mark.parametrize("change", ["cnmf_finished", "estimates_file"])
def test_save_after_cnmf_rerun_is_stale(series_est, capsys, change):
    paths = series_est
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        if change == "cnmf_finished":
            _set_cnmf(paths, S1, finished=datetime.now().isoformat(timespec="milliseconds"))
        else:  # CNMF-E wrote new estimates
            est = paths.estimates(S1)
            st = est.stat()
            os.utime(est, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        assert save(srv, page) == inspect_server.OUTDATED
        assert not paths.manual_result(S1).exists()
        assert _decisions(paths, S1) == []
        assert not locked(srv, page, lambda: page.next_button.visible)
    out = capsys.readouterr().out
    assert f"[WARN] {S1}: result not saved" in out
    assert S1 in actions.waiting_gate_sids(paths)


def _divs(srv, page):
    """The Div texts of a tab's page on the server, read under its document lock."""
    return locked(srv, page, lambda: [d.text for d in _models(page.doc, Div)])


def _get(srv, sid):
    """A browser's request for the page of sid (no websocket): the server's new page."""
    before = {id(p) for p in srv.pages()}
    with urllib.request.urlopen(inspect_server.session_url(srv.url, sid), timeout=120) as r:
        assert r.status == 200
    page, = [p for p in srv.pages() if id(p) not in before]
    _settle(page, sid)
    return page


def test_second_tab_same_session_refused(series_est):
    opening, refused = inspect_server.OPENING.format(sid=S1), inspect_server.SECOND_TAB.format(sid=S1)
    with running(series_est) as srv:
        with tab(srv, S1) as a:
            first = srv.page_of(a.id)
            assert _opened(first, S1)
            with tab(srv, S1) as b:
                # first "opening": a reload asks for the new page before the old one has closed
                assert [d.text for d in _models(b.document, Div)] == [opening]
                assert not _models(b.document, Button)
                second = srv.page_of(b.id)
                until(lambda: _divs(srv, second) == [refused], timeout=10)
                assert second.sid is None and not _models(second.doc, Button)
            # another session in a second tab is fine
            with tab(srv, S2) as c:
                assert _opened(srv.page_of(c.id), S2)
        # the tab closed: the session can be opened again
        until(lambda: not first.is_open())
        with tab(srv, S1) as d:
            assert _opened(srv.page_of(d.id), S1)


def test_reload_gets_the_session(series_est):
    """I1: a reload asks for the new page while the old page's websocket is still open; the old one closes
    right after, and the new page opens the session."""
    with running(series_est) as srv:
        with tab(srv, S1) as a:
            assert _opened(srv.page_of(a.id), S1)
            new = _get(srv, S1)
            assert _divs(srv, new) == [inspect_server.OPENING.format(sid=S1)]
        until(lambda: _opened(new, S1), timeout=10)
        assert new.doc.title == inspect_server.TITLE.format(sid=S1)


def test_open_tab_keeps_the_session(series_est):
    """I1: the other tab stays connected: the new page is refused once the retry finds it still there."""
    with running(series_est) as srv:
        with tab(srv, S1) as a:
            holder = srv.page_of(a.id)
            new = _get(srv, S1)
            until(lambda: _divs(srv, new) == [inspect_server.SECOND_TAB.format(sid=S1)], timeout=10)
            assert new.sid is None and _opened(holder, S1)


def test_second_server_refused(series_est, capsys):
    with running(series_est) as srv:
        capsys.readouterr()
        code = inspect_server.serve(series_est.root, port=0, open_browser=False, idle_exit_s=5)
        assert code == 3
        assert inspect_server.ALREADY_RUNNING.format(url=srv.url) in capsys.readouterr().out
        assert inspect_server.server_state(series_est).alive  # the first one is untouched
        with urllib.request.urlopen(srv.url + "/", timeout=120) as r:
            assert r.status == 200


def test_port_in_use(series_est, capsys):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        port = taken.getsockname()[1]
        assert inspect_server.serve(series_est.root, port=port, open_browser=False, idle_exit_s=5) == 2
    out = capsys.readouterr().out
    assert f"[FAILED] cannot listen on port {port}" in out and out.isascii()
    assert not series_est.inspection_lock.exists()


@pytest.mark.parametrize("strict", STRICT)
def test_server_exits_when_idle(series_est, strict):
    """The tab is closed without saving: no decision, the session still waits, the server stops after
    idle_exit_s and removes its lock."""
    paths = series_est
    idle = 2.0
    srv = inspect_server.InspectServer(paths.root, sid=S1, port=0, open_browser=False, idle_exit_s=idle)
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("code", srv.run()), name="t4-inspect-server")
    t.start()
    try:
        assert srv.started.wait(120)
        with tab(srv, S1) as s:
            page = srv.page_of(s.id)
            assert _opened(page, S1)
            until(lambda: page.connected)
            assert inspect_server.server_state(paths).current == [S1]
            time.sleep(idle + 1)  # an open tab keeps the server
            assert t.is_alive()
        closed = time.monotonic()
        bound = idle + 5 if strict else idle + 30  # measured: idle + 1.2 s (3.16 s)
        t.join(bound)
        took = time.monotonic() - closed
        print(f"[timing] stopped {took:.2f} s after the tab closed (idle_exit_s {idle})")
        assert not t.is_alive() and took <= bound
        assert result["code"] == 0
        assert not paths.inspection_lock.exists()
        assert _decisions(paths, S1) == [] and S1 in actions.waiting_gate_sids(paths)
    finally:
        srv.request_stop()
        t.join(180)
        assert not t.is_alive()


def test_next_session_prepared_in_background(series_est):
    paths = series_est
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        until(lambda: inspect_server.server_state(paths).prepared == S2, timeout=30)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        until(lambda: locked(srv, page, lambda: page.next_button.visible))
        t0 = time.perf_counter()
        locked(srv, page, lambda: click(page.doc, inspect_server.NEXT_LABEL.format(sid=S2)))
        until(lambda: _opened(page, S2), timeout=30, step=0.002)
        dt = time.perf_counter() - t0
        print(f"[speed] next prepared session opened in {dt:.3f} s")
        assert dt <= 1.0, dt
        assert page.doc.title == inspect_server.TITLE.format(sid=S2)
        # the tab now holds S2, and the next one is prepared
        until(lambda: inspect_server.server_state(paths).current == [S2])
        until(lambda: inspect_server.server_state(paths).prepared == S3, timeout=30)


def test_build_widens_blas_threads(series_est, monkeypatch):
    """K1a: the server prepares a session with PREPARE_THREADS BLAS threads (at most 8)."""
    seen = []
    real = bm_examinator.prepare_session
    monkeypatch.setattr(bm_examinator, "prepare_session", lambda *a, **k: seen.append(k.get("threads")) or real(*a, **k))
    inspect_server.InspectServer(series_est.root).build(S1)
    assert seen == [inspect_server.PREPARE_THREADS] and seen[0] == min(8, os.cpu_count())


def test_unprepared_session_opens_off_the_request(series_est, monkeypatch):
    """K1c: a session that is not prepared answers the browser's request at once with "Opening" and is built
    off it; the window replaces the text when the session is ready."""
    release, building = threading.Event(), threading.Event()
    real = inspect_server.InspectServer.build

    def slow(self, sid):
        if sid == S1:
            building.set()
            assert release.wait(60)
        return real(self, sid)

    monkeypatch.setattr(inspect_server.InspectServer, "build", slow)
    answered = {}
    with running(series_est) as srv:
        def get():
            with urllib.request.urlopen(inspect_server.session_url(srv.url, S1), timeout=120) as r:
                answered["status"] = r.status
        t = threading.Thread(target=get)
        try:
            t.start()
            assert building.wait(30)
            t.join(10)
            assert answered.get("status") == 200, "the request waited for the build"
            page, = srv.pages()
            assert _divs(srv, page) == [inspect_server.OPENING.format(sid=S1)] and page.sid == S1
            assert page.doc.title == inspect_server.TITLE.format(sid=S1)
        finally:
            release.set()
            t.join(120)
        until(lambda: _opened(page, S1), timeout=60)


def test_next_session_skips_decided(series_est):
    paths = series_est
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        label = until(lambda: locked(srv, page, lambda: page.next_button.visible and page.next_button.label))
        assert label == inspect_server.NEXT_LABEL.format(sid=S2)
        actions.gate_not_needed(paths, [S2], "ann")  # decided in the window or the CLI meanwhile
        locked(srv, page, lambda: click(page.doc, label))
        until(lambda: _opened(page, S3), timeout=60)
        # after S3 is saved, nothing waits any more
        assert save(srv, page) == inspect_server.SAVED.format(sid=S3)
        text = until(lambda: locked(srv, page, lambda: page.next_text.visible and page.next_text.text))
        assert text == inspect_server.NO_MORE and not page.next_button.visible


def test_next_skips_session_open_in_another_tab(series_est):
    with running(series_est) as srv, tab(srv, S2) as other, tab(srv, S1) as s:
        assert _opened(srv.page_of(other.id), S2)
        page = srv.page_of(s.id)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        label = until(lambda: locked(srv, page, lambda: page.next_button.visible and page.next_button.label))
        assert label == inspect_server.NEXT_LABEL.format(sid=S3)


def test_contour_cache_reused(series_est, monkeypatch):
    paths = series_est
    calls = []
    real = bm_examinator.cm.utils.visualization.get_contours

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(bm_examinator.cm.utils.visualization, "get_contours", counted)
    srv = inspect_server.InspectServer(paths.root)
    first = srv.build(S1)
    assert len(calls) == 1 and paths.inspection_cache(S1).is_file()
    again = srv.build(S1)
    assert len(calls) == 1  # from the cache
    assert len(again.contours) == len(first.contours) == 8
    for a, b in zip(again.contours, first.contours):
        assert np.array_equal(a, b, equal_nan=True)
    for key in first.data:
        np.testing.assert_equal(again.data[key], first.data[key], err_msg=key)  # NaN equals NaN here
    est = paths.estimates(S1)
    st = est.stat()
    os.utime(est, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # CNMF-E wrote the file again
    srv.build(S1)
    assert len(calls) == 2


def _estimates_of(paths, sid, n):
    """sid's estimates replaced by make_estimates' session of n components (the fixture's have 8, fewer than
    the wave hint needs); returns them."""
    with tempfile.TemporaryDirectory(prefix="t6c_") as d:
        est = _load(make_estimates(d, n=n))
    _dump(est, paths.estimates(sid))
    return est


def _dump(est, path):
    with open(path, "wb") as fh:
        pickle.dump(est, fh)


def _add_wave(paths, sid, n=12, start=100, frames=30):
    """A calcium wave in sid's estimates (n components): at frame `start` every component rises to 20 times the
    largest noise SD and decays (10 frames) over `frames` frames. Returns the session's wave share."""
    est = _estimates_of(paths, sid, n)
    est.C = est.C.copy()
    est.C[:, start:start + frames] += 20.0 * np.max(est.neurons_sn) * np.exp(-np.arange(frames) / 10.0)
    _dump(est, paths.estimates(sid))
    frac = wave.wave_frac(est.C, est.idx_components, sn=wave.component_noise(est))
    assert frac >= 10 / est.C.shape[1]  # at least its first 10 frames
    return frac


def _set_wave(paths, **fields):
    config.update_yaml(paths.series_config, {"inspection": {"wave": fields}})


def test_wave_banner_only_in_page(series_est):
    """Task 6: a session whose share of wave frames is above inspection.wave.warn_frac has the hint in the
    server's line above the window; Monitor's cells and warnings never mention it."""
    paths = series_est
    frac = _add_wave(paths, S1)
    _set_wave(paths, warn_frac=frac / 2)
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert _opened(page, S1)
        assert page.prepared.wave_frac == frac  # computed while the session was prepared
        banner = _div(page.doc, "wave")
        assert banner.visible and banner.text == html.escape(wave.wave_text(frac, 0.3))
        header = page.doc.roots[0]
        assert header.name == "inspection_header" and banner in header.references()
        assert not CYRILLIC.search(banner.text)
    view = monitor.series_view(paths)
    shown = [f"{c.text} {c.tooltip}" for r in view.rows for c in r.cells.values()]
    shown += [f"{w.kind} {w.text}" for w in view.warnings]
    assert shown and not any("wave" in t.lower() for t in shown)


def test_no_banner_without_threshold(series_est):
    paths = series_est
    frac = _add_wave(paths, S1)
    _set_wave(paths, warn_frac=None)
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert _opened(page, S1) and page.prepared.wave_frac == frac
        banner = _div(page.doc, "wave")
        assert not banner.visible and banner.text == ""


def test_no_banner_on_ordinary_sessions(series_est):
    """With the default threshold the synthetic sessions (independent sparse transients) show no hint."""
    warn = config.load_defaults()["inspection"]["wave"]["warn_frac"]
    assert warn is not None
    for sid in (S1, S2):
        _estimates_of(series_est, sid, 12)
    with running(series_est) as srv:
        for sid in (S1, S2):
            with tab(srv, sid) as s:
                page = srv.page_of(s.id)
                assert _opened(page, sid) and page.prepared.wave_frac <= warn
                banner = _div(page.doc, "wave")
                assert not banner.visible and banner.text == ""


@pytest.mark.parametrize("n", [3, 9, 10])
def test_no_hint_below_ten_components(series_est, n):
    """V1: with fewer than 10 kept components the share is not computed and the page has no hint, whatever the
    threshold; the series log says why."""
    paths = series_est
    _add_wave(paths, S1, n=n)
    _set_wave(paths, warn_frac=0.0)
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert _opened(page, S1)
        banner = _div(page.doc, "wave")
        if n < 10:
            assert page.prepared.wave_frac is None and not banner.visible
        else:
            assert page.prepared.wave_frac > 0 and banner.visible
    # S2 (8 components, prepared as the next session) logs it too: only S1's line counts here
    logged = f"{S1}: too few components for the wave hint" in paths.series_log.read_text(encoding="utf-8")
    assert logged == (n < 10)


def test_wave_share_cached(series_est, monkeypatch):
    """W4: the share is kept in the session's inspect cache with the contours: a later preparation of the same
    estimates file with the same rule reads it; another file or another k_mad computes it again."""
    paths = series_est
    frac = _add_wave(paths, S1)
    calls = []
    real = wave.wave_frac

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(wave, "wave_frac", counted)
    srv = inspect_server.InspectServer(paths.root)
    assert srv.build(S1).wave_frac == frac and len(calls) == 1
    assert srv.build(S1).wave_frac == frac and len(calls) == 1  # from the cache
    _set_wave(paths, k_mad=4.0)  # another rule, in the series config: read at each preparation
    assert srv.build(S1).wave_frac != frac and len(calls) == 2
    assert len(srv.build(S1).contours) == 12 and len(calls) == 2  # cached again, with the contours
    est = paths.estimates(S1)
    st = est.stat()
    os.utime(est, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))  # CNMF-E wrote the file again
    srv.build(S1)
    assert len(calls) == 3
    # an inspect cache written before the share was kept (contours only) gets the share added
    with np.load(paths.inspection_cache(S1)) as z:
        old = {k: z[k] for k in ("size", "mtime_ns", "cthr", "coords", "lengths")}
    with open(paths.inspection_cache(S1), "wb") as fh:
        np.savez(fh, **old)
    srv.build(S1)
    assert len(calls) == 4
    srv.build(S1)
    assert len(calls) == 4


@pytest.mark.parametrize("change", ["active_frac", "rule_version", "corrupt_field", "truncated"])
def test_wave_share_cache_key(series_est, monkeypatch, change):
    """V5: a cached share is used only with the same active_frac and rule version; a broken cache that has the
    wave fields is computed again (contours too) and written anew."""
    paths = series_est
    _add_wave(paths, S1)
    calls = []
    real = wave.wave_frac
    monkeypatch.setattr(wave, "wave_frac", lambda *a, **k: calls.append(1) or real(*a, **k))
    srv = inspect_server.InspectServer(paths.root)
    first = srv.build(S1)
    assert len(calls) == 1
    f = paths.inspection_cache(S1)
    with np.load(f) as z:
        assert {"wave_key", "wave_frac"} <= set(z.files)
        fields = {k: z[k] for k in z.files}
    if change == "active_frac":
        _set_wave(paths, active_frac=0.5)
    elif change == "rule_version":
        monkeypatch.setattr(wave, "RULE_VERSION", wave.RULE_VERSION + 1)
    elif change == "corrupt_field":
        with open(f, "wb") as fh:
            np.savez(fh, **{**fields, "wave_frac": np.array([], dtype=np.float64)})  # not one number
    else:
        data = f.read_bytes()
        f.write_bytes(data[:len(data) // 2])
    again = srv.build(S1)
    assert len(calls) == 2
    if change in ("corrupt_field", "truncated"):
        assert again.wave_frac == first.wave_frac and len(again.contours) == len(first.contours)
    srv.build(S1)
    assert len(calls) == 2  # the cache was written anew with the share


def test_wave_hint_rule():
    w = {"k_mad": 3.0, "active_frac": 0.3, "warn_frac": 0.02}
    assert inspect_server.wave_hint(0.03, w) == wave.wave_text(0.03, 0.3)
    assert inspect_server.wave_hint(0.02, w) is None  # above the threshold, not at it
    assert inspect_server.wave_hint(0.03, {**w, "warn_frac": None}) is None
    assert inspect_server.wave_hint(None, w) is None  # not computed
    assert inspect_server.wave_hint(0.03, {**w, "warn_frac": "2%"}) is None  # a typo does not block the page


def test_progress_save(series_est):
    paths = series_est
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        text = locked(srv, page, lambda: (click(page.doc, "Save progress"), _div(page.doc, "status").text)[1])
    saved = list(paths.inspection_progress_dir(S1).iterdir())
    assert len(saved) == 1 and re.fullmatch(rf"{S1}_\d{{4}}-\d\d-\d\d_\d\d-\d\d-\d\d_estimates\.pickle",
                                            saved[0].name)
    assert text == inspect_server.PROGRESS_SAVED.format(name=saved[0].name)
    assert _decisions(paths, S1) == [] and not paths.manual_result(S1).exists()


def test_short_timestamps_named_on_the_page(series_est):
    """K1: timestamps shorter than the traces: a page that names the session and both lengths, never a dead
    page; the other sessions open as usual."""
    paths = series_est
    rec = load_record(paths, S3)
    ts = paths.root / rec.parts[0].timestamps_file
    ts.write_text("".join(f"{i * 500000}\n" for i in range(250)), encoding="utf-8")
    with running(paths) as srv:
        with tab(srv, S3) as s:
            texts = _divs(srv, srv.page_of(s.id))
            assert texts == [inspect_server.SHORT_TIMESTAMPS.format(sid=S3, n_ts=250, n_frames=300)]
            assert "250" in texts[0] and "300" in texts[0] and S3 in texts[0]
        with tab(srv, S1) as s:
            assert _opened(srv.page_of(s.id), S1)
    assert "250 timestamps for 300 frames" in paths.series_log.read_text(encoding="utf-8")


def test_side_files_failure_on_page(series_est, monkeypatch):
    """K3: a side-file failure in the background shows one line on the page; the result and the decision
    stay. K2: the line names the type and the file, the series log has the localized text."""
    paths = series_est
    target = paths.inspection_dir(S1) / f"{S1}_traces.csv"

    def failing(source, folder, base, sigma=3):  # the child's report of a failure, as _side_files_child raises it
        raise bm_examinator.SideFilesFailed(f"side files in {folder} not all written ({target.name}: {DENIED})",
                                            f"PermissionError on {target.name}")

    monkeypatch.setattr(bm_examinator, "_side_files_child", failing)
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        note = until(lambda: locked(srv, page, lambda: page.notes_div.visible and page.notes_div.text))
        assert note == inspect_server.SIDE_FILES_FAILED.format(sid=S1, what=f"PermissionError on {target.name}")
        assert paths.manual_result(S1).is_file() and _decisions(paths, S1) == ["cleaned"]
    assert DENIED in paths.series_log.read_text(encoding="utf-8")


def _no_decision(paths, sid, who):
    raise PermissionError(13, DENIED)  # the registry file is held: a localized system text


def test_decision_not_recorded_after_result_written(series_est, monkeypatch):
    """M6: the result file is written, the decision cannot be: the page says so, nothing else is done."""
    paths = series_est
    monkeypatch.setattr(actions, "gate_cleaned", _no_decision)
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        text = save(srv, page)
        assert text == inspect_server.DECISION_FAILED.format(what="PermissionError")
        assert paths.manual_result(S1).is_file() and _decisions(paths, S1) == []
        assert not srv._side_futures  # no side files for a save that did not count
        time.sleep(0.5)
        assert not locked(srv, page, lambda: page.next_button.visible or page.next_text.visible)
        monkeypatch.undo()
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)  # "Save again"
        assert _decisions(paths, S1) == ["cleaned"]


def _gated_child(started, release, written):
    """A stand-in for the side-file child process: it reads the result as the child does, notes its components,
    holds the first job until `release`, then writes the files in this process."""
    def child(source, folder, base, sigma=3):
        est = side_files.load_result(source)
        written.append([int(c) for c in side_files.kept_for_side_files(est)])
        if len(written) == 1:
            started.set()
            assert release.wait(60)
        side_files.write_side_files(est, folder, base, sigma)
    return child


def test_newer_save_skips_queued_side_files(series_est, monkeypatch):
    """M5: side files of a save that has not started yet are dropped when the same session is saved again;
    only the last save's files matter."""
    paths = series_est
    started, release, written = threading.Event(), threading.Event(), []
    monkeypatch.setattr(bm_examinator, "_side_files_child", _gated_child(started, release, written))
    try:
        with running(paths) as srv, tab(srv, S1) as s:
            page = srv.page_of(s.id)
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            assert started.wait(60)  # the first save's files are being written
            for row in (0, 0):
                locked(srv, page, lambda: delete_row(page.doc, row))
                assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            assert len(srv._side_futures) == 2  # the running one and the last one; the middle one dropped
            release.set()
    finally:
        release.set()
    assert written == [list(range(8)), list(range(2, 8))]
    assert pd.read_csv(paths.inspection_dir(S1) / f"{S1}_traces.csv").shape[1] == 1 + 6


def test_queued_side_files_kept_when_the_new_job_cannot_start(series_est, monkeypatch):
    """N4: the earlier save's queued job is dropped only after the new one is queued: a new job that cannot be
    queued leaves it."""
    paths = series_est
    started, release, written = threading.Event(), threading.Event(), []

    def cannot_queue(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(bm_examinator, "_side_files_child", _gated_child(started, release, written))
    try:
        with running(paths) as srv, tab(srv, S1) as s:
            page = srv.page_of(s.id)
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            assert started.wait(60)
            locked(srv, page, lambda: delete_row(page.doc, 0))
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)  # queued behind the first
            real_queue = bm_examinator.write_side_files_in_background
            monkeypatch.setattr(bm_examinator, "write_side_files_in_background", cannot_queue)
            locked(srv, page, lambda: delete_row(page.doc, 0))
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            monkeypatch.setattr(bm_examinator, "write_side_files_in_background", real_queue)
            note = until(lambda: locked(srv, page, lambda: page.notes_div.visible and page.notes_div.text))
            assert note == inspect_server.SIDE_FILES_FAILED.format(sid=S1, what="RuntimeError")
            release.set()
    finally:
        release.set()
    # the second save's job was still run; it reads the result on disk when it runs, which is the third save's
    assert written == [list(range(8)), list(range(2, 8))]


def test_closed_tab_is_freed(series_est, monkeypatch):
    """I2: after Save results the tab is closed and Bokeh discards its session: nothing of the server keeps the
    page, its document or its prepared session (hundreds of MB on an hour session)."""
    monkeypatch.setattr(inspect_server, "UNUSED_SESSION_MS", 500)
    monkeypatch.setattr(inspect_server, "CHECK_UNUSED_MS", 200)
    with running(series_est) as srv:
        with tab(srv, S1) as s:
            key, page = s.id, srv.page_of(s.id)
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            until(lambda: not srv._side_futures, timeout=60)  # written, and forgotten
            refs = [weakref.ref(page), weakref.ref(page.prepared), weakref.ref(page.doc)]
            del page
        until(lambda: srv.page_of(key) is None, timeout=30)
        until(lambda: (gc.collect(), all(r() is None for r in refs))[1], timeout=30)


def test_stopping_server_is_waited_for(series_est, monkeypatch, capsys):
    """M3: a server that stops (here: while the last save's side files are written) has no address in its lock;
    the inspect command waits for it to end, then starts a new server, never a dead address."""
    paths = series_est
    release = threading.Event()
    real = bm_examinator._side_files_child

    def slow(*args, **kwargs):
        assert release.wait(60)
        return real(*args, **kwargs)

    monkeypatch.setattr(bm_examinator, "_side_files_child", slow)
    srv = inspect_server.InspectServer(paths.root, port=0, open_browser=False, idle_exit_s=600)
    t = threading.Thread(target=srv.run, name="t4-inspect-server")
    t.start()
    served, out = [], {}
    cmd = threading.Thread(target=lambda: out.setdefault("code", cli.main(["inspect", str(paths.root),
                                                                          "--no-browser"])))
    try:
        assert srv.started.wait(120)
        with tab(srv, S1) as s:
            assert save(srv, srv.page_of(s.id)) == inspect_server.SAVED.format(sid=S1)
        srv.request_stop()
        st = until(lambda: (lambda x: x.stopping and x)(inspect_server.server_state(paths)), timeout=30)
        assert st.alive and st.url is None and not st.available
        monkeypatch.setattr(inspect_server, "serve", lambda *a, **k: served.append(a) or 0)
        cmd.start()
        time.sleep(1.5)
        assert cmd.is_alive() and served == []  # it waits
        release.set()
        cmd.join(60)
        assert not cmd.is_alive() and out["code"] == 0 and len(served) == 1
        assert not t.is_alive()  # the old one had ended
        assert "[OK] inspection server is running" not in capsys.readouterr().out
    finally:
        release.set()
        srv.request_stop()
        t.join(180)
        cmd.join(180)
        assert not t.is_alive() and not cmd.is_alive()


def test_stopping_server_waits_for_side_files_with_a_cap(series_est, monkeypatch):
    """K1e: a stopping server waits for the side files it queued until they are written, with a line in the
    series log every SIDE_FILES_LOG_S; past SIDE_FILES_CAP_S it stops the writer processes and ends."""
    paths = series_est
    monkeypatch.setattr(inspect_server, "SIDE_FILES_LOG_S", 0.5)
    monkeypatch.setattr(inspect_server, "SIDE_FILES_CAP_S", 2.0)
    release, stopped = threading.Event(), []

    def stuck(*args, **kwargs):  # a writer that does not finish by itself
        assert release.wait(60)

    def stop_side_files():
        stopped.append(1)
        release.set()  # as killing the child ends its job
        return 1

    monkeypatch.setattr(bm_examinator, "_side_files_child", stuck)
    monkeypatch.setattr(bm_examinator, "stop_side_files", stop_side_files)
    try:
        with running(paths) as srv, tab(srv, S1) as s:
            assert save(srv, srv.page_of(s.id)) == inspect_server.SAVED.format(sid=S1)
            t0 = time.monotonic()
    finally:
        release.set()
    assert 2.0 <= time.monotonic() - t0 <= 2.0 + 30  # the cap, then the end (the thread joins in `running`)
    assert stopped == [1]
    text = paths.series_log.read_text(encoding="utf-8")
    assert text.count("waiting for the side files of 1 saves") >= 3  # at 0, 0.5, 1.0, 1.5 s
    assert re.search(r"side files of 1 saves not written after \d+ s: stopped \(1 writer processes\)", text)
    # R6: the folder says its side files are incomplete, until a complete write
    marker = paths.inspection_dir(S1) / f"{S1}_side_files_INCOMPLETE.txt"
    note = marker.read_text(encoding="utf-8")
    assert S1 in note and "incomplete" in note and re.search(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", note)
    side_files.write_side_files(side_files.load_result(paths.manual_result(S1)), paths.inspection_dir(S1), f"{S1}_")
    assert not marker.exists()


def test_two_inspect_commands_at_once(series_est, monkeypatch, capsys):
    """M4: the other command took the lock between this one's look and its own start: this one opens the
    running server instead of failing."""
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url))
    with running(series_est) as srv:
        real, calls = inspect_server.server_state, []

        def nothing_at_first(paths):
            calls.append(1)
            return inspect_server.ServerState(False, None, None) if len(calls) == 1 else real(paths)

        monkeypatch.setattr(inspect_server, "server_state", nothing_at_first)
        capsys.readouterr()
        assert cli.main(["inspect", str(series_est.root), S1]) == 0
        out = capsys.readouterr().out
        assert f"[OK] inspection server is running: {srv.url}" in out and "[FAILED]" not in out
        assert opened == [inspect_server.session_url(srv.url, S1)]


@pytest.mark.parametrize("strict", STRICT)
def test_empty_lock_means_wait(tmp_path, strict):
    """M4: a lock read just after its new holder emptied it (before it wrote itself in) is "wait", not
    "no server"."""
    paths = SeriesPaths(tmp_path / "s")
    lock = paths.inspection_lock
    lock.parent.mkdir(parents=True)
    lock.write_text("", encoding="utf-8")
    st = inspect_server.server_state(paths)
    assert st.starting and not st.alive and not st.available
    me = psutil.Process()

    def holder_writes():
        time.sleep(0.5)
        lock.write_text(json.dumps({"pid": me.pid, "create_time": me.create_time(), "url": "http://localhost:1",
                                    "current": [], "prepared": None}), encoding="utf-8")

    writer = threading.Thread(target=holder_writes)
    writer.start()
    try:
        st = inspect_server.wait_ready(paths)
    finally:
        writer.join(10)
    assert st.available and st.url == "http://localhost:1"
    lock.write_text(json.dumps({"pid": 999_999_999, "create_time": 1.0, "url": "http://localhost:1"}),
                    encoding="utf-8")
    t0 = time.monotonic()
    st = inspect_server.wait_ready(paths)  # a dead holder: nothing to wait for
    took = time.monotonic() - t0
    print(f"[timing] a dead holder answered in {took:.3f} s")
    # waiting for it would take URL_WAIT_S (10 s)
    assert not st.alive and not st.starting and not st.available
    assert took < (2 if strict else inspect_server.URL_WAIT_S / 2)


def test_old_empty_lock_is_stale(tmp_path):
    """N5: an empty lock nobody writes into (its holder died between emptying it and writing itself in) is not
    waited for once it is older than a moment."""
    paths = SeriesPaths(tmp_path / "s")
    lock = paths.inspection_lock
    lock.parent.mkdir(parents=True)
    lock.write_text("", encoding="utf-8")
    old = time.time() - 5
    os.utime(lock, (old, old))
    st = inspect_server.server_state(paths)
    assert not st.starting and not st.alive and not st.available
    t0 = time.monotonic()
    inspect_server.wait_ready(paths)
    assert time.monotonic() - t0 < 1


@pytest.mark.parametrize("strict", STRICT)
def test_lock_left_by_a_stopped_server(series_est, strict):
    """N3: a reader holds the lock file while the server stops, so Windows cannot delete it: what is left says
    "no holder", and nobody waits for it."""
    paths = series_est
    srv = inspect_server.InspectServer(paths.root, port=0, open_browser=False, idle_exit_s=600)
    t = threading.Thread(target=srv.run, name="t4-inspect-server")
    t.start()
    try:
        assert srv.started.wait(120)
        with open(paths.inspection_lock, "rb"):  # a reader (Monitor, another command) at the wrong moment
            srv.request_stop()
            t.join(180)
            assert not t.is_alive()
        if paths.inspection_lock.exists():  # left behind (on Windows)
            st = inspect_server.server_state(paths)
            assert not st.alive and not st.stopping and not st.starting
        t0 = time.monotonic()
        assert not inspect_server.wait_ready(paths).available
        took = time.monotonic() - t0
        print(f"[timing] a lock without holder answered in {took:.3f} s")
        assert took < (1 if strict else inspect_server.URL_WAIT_S / 2)  # a wait would take 10 s or more
        # a new server takes the series
        with running(paths) as srv2:
            assert inspect_server.server_state(paths).available and srv2.url
    finally:
        srv.request_stop()
        t.join(180)
        assert not t.is_alive()


@pytest.mark.parametrize("strict", STRICT)
def test_stopping_wait_is_bounded_once(series_est, monkeypatch, capsys, strict):
    """N5: a server that does not finish stopping: the inspect command waits once, for STOP_WAIT_S, then fails
    naming the lock file (not a second wait, not a second server)."""
    paths = series_est
    monkeypatch.setattr(inspect_server, "STOP_WAIT_S", 1.0)

    def must_not_serve(*args, **kwargs):
        raise AssertionError("a server was started while another one holds the series")

    monkeypatch.setattr(inspect_server, "serve", must_not_serve)
    waits, real_wait = [], inspect_server.wait_ready
    monkeypatch.setattr(inspect_server, "wait_ready", lambda *a, **k: waits.append(1) or real_wait(*a, **k))
    fh = runner._take_lock(paths.inspection_lock, {"url": None, "current": [], "prepared": None, "stopping": True})
    assert fh is not None
    try:
        assert inspect_server.server_state(paths).stopping
        capsys.readouterr()
        t0 = time.monotonic()
        assert cli.main(["inspect", str(paths.root), "--no-browser"]) == 3
        took = time.monotonic() - t0
        print(f"[timing] the inspect command gave up after {took:.2f} s (STOP_WAIT_S 1.0)")
        out = capsys.readouterr().out
        assert "[FAILED]" in out and str(paths.inspection_lock) in out
        assert waits == [1] and out.count("is stopping") == 1  # one wait, one notice
        assert took < (1.9 if strict else 5.0)  # measured: at most 1.02 s
    finally:
        runner._drop_lock(paths.inspection_lock, fh)


def _session_id(srv, page):
    with srv._lock:
        return next(k for k, p in srv._pages.items() if p is page)


def test_waits_for_a_page_that_never_connects(series_est, monkeypatch):
    """N1: the session's other page never connects (its request was left while the page was built): the new
    page shows "Opening" and gets the session once the other page's connect grace is over."""
    monkeypatch.setattr(inspect_server, "CONNECT_GRACE_S", 3.0)
    with running(series_est) as srv:
        first = _get(srv, S1)  # holds S1, never connects
        assert _opened(first, S1)
        new = _get(srv, S1)
        assert _divs(srv, new) == [inspect_server.OPENING.format(sid=S1)]
        until(lambda: _opened(new, S1), timeout=15)
        assert time.monotonic() - first.created >= 3.0


def test_waiting_page_refused_when_the_other_connects(series_est, monkeypatch):
    """N1: the other page connects while the new one waits: the new page is refused."""
    monkeypatch.setattr(inspect_server, "CONNECT_GRACE_S", 3.0)
    with running(series_est) as srv:
        first = _get(srv, S1)
        new = _get(srv, S1)
        io = IOLoop(make_current=False)
        _TAB_LOOPS.append(io.asyncio_loop)
        asyncio.set_event_loop(io.asyncio_loop)
        try:
            # the browser of the first page connects late, within its grace
            session = pull_session(session_id=_session_id(srv, first), url=srv.url + "/", io_loop=io)
            try:
                until(lambda: _divs(srv, new) == [inspect_server.SECOND_TAB.format(sid=S1)], timeout=10)
                assert new.sid is None and _opened(first, S1)
            finally:
                session.close()
        finally:
            _TAB_LOOPS.pop()
            asyncio.set_event_loop(_TAB_LOOPS[-1] if _TAB_LOOPS else None)
            io.close(all_fds=True)


def test_pages_not_kept_by_the_browser(series_est):
    """N2: a page kept in the browser's back-forward cache keeps its websocket open after the tab has left it:
    pages are sent with Cache-Control: no-store, and their script closes the sockets of a page put in that
    cache and loads a page restored from it again."""
    with running(series_est) as srv:
        for url in (inspect_server.session_url(srv.url, S1), srv.url + "/"):
            with urllib.request.urlopen(url, timeout=120) as r:
                assert r.headers["Cache-Control"] == "no-store", url
                body = r.read().decode("utf-8")
            assert "pagehide" in body and "pageshow" in body and "event.persisted" in body, url
            # before BokehJS: its websocket is made by the page script's WebSocket
            assert body.index("pagehide") < body.index("static/js/bokeh"), url


def test_page_texts_are_english(series_est, monkeypatch, capsys):
    """Titles, status texts and refusals have no Cyrillic, also when the system's error text is localized
    (K2) and the series folder is Cyrillic; console lines are ASCII."""
    paths = series_est
    _set_cnmf(paths, S2, state=State.PENDING)
    texts = []
    real_write = bm_examinator.write_result

    def denied(est, path):
        raise PermissionError(13, DENIED, str(path))

    with running(paths) as srv:
        with tab(srv) as index:
            texts += _texts(index.document)
            assert S1 in index.document.roots[0].text and S3 in index.document.roots[0].text
        for sid in (S2, "NOF_X99_9D", "../registry/" + S1):  # CNMF-E not done; not a session; not a name
            with tab(srv, sid) as s:
                page = srv.page_of(s.id)
                texts += locked(srv, page, lambda: _texts(page.doc))
        with tab(srv, S1) as s, tab(srv, S1) as second:
            page = srv.page_of(s.id)
            texts += locked(srv, page, lambda: _texts(page.doc)) + _texts(second.document)  # "opening", refused
            refused = srv.page_of(second.id)
            until(lambda: _divs(srv, refused) == [inspect_server.SECOND_TAB.format(sid=S1)], timeout=10)
            texts += locked(srv, refused, lambda: _texts(refused.doc))
            monkeypatch.setattr(actions, "gate_cleaned", _no_decision)
            texts.append(save(srv, page))
            monkeypatch.undo()
            monkeypatch.setattr(bm_examinator, "write_result", denied)
            failed = save(srv, page)
            assert failed == inspect_server.SAVE_FAILED.format(
                what=f"PermissionError on {paths.manual_result(S1).name}")
            assert _decisions(paths, S1) == []
            monkeypatch.setattr(bm_examinator, "write_result", real_write)
            texts += [failed, save(srv, page)]
            until(lambda: locked(srv, page, lambda: page.next_button.visible))
            texts += _texts(page.doc)
            _set_cnmf(paths, S1, finished=datetime.now().isoformat(timespec="milliseconds"))
            texts.append(save(srv, page))
    assert inspect_server.OUTDATED in texts and inspect_server.SECOND_TAB.format(sid=S1) in texts
    assert inspect_server.OPENING.format(sid=S1) in texts
    assert inspect_server.DECISION_FAILED.format(what="PermissionError") in texts
    assert inspect_server.NOT_DONE.format(sid=S2) in texts
    assert inspect_server.UNKNOWN.format(sid="NOF_X99_9D") in texts
    assert html.escape(inspect_server.UNKNOWN.format(sid="../registry/" + S1)) in texts
    for text in texts:
        assert not CYRILLIC.search(text), text
    assert DENIED in paths.series_log.read_text(encoding="utf-8")  # the whole error text is logged
    out = capsys.readouterr().out
    assert out.isascii(), out


def test_cli_inspect_help():
    r = subprocess.run([sys.executable, "-m", "bearmind", "inspect", "--help"], cwd=REPO, capture_output=True,
                       text=True, env={**os.environ, "PYTHONPATH": str(REPO)}, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "--no-browser" in r.stdout and "--port" in r.stdout and "SID" in r.stdout


def test_cli_inspect_reuses_running_server(series_est, monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url))

    def must_not_serve(*args, **kwargs):
        raise AssertionError("a second server was started")

    with running(series_est) as srv:
        monkeypatch.setattr(inspect_server, "serve", must_not_serve)
        capsys.readouterr()
        assert cli.main(["inspect", str(series_est.root), S1]) == 0
        assert f"[OK] inspection server is running: {srv.url}" in capsys.readouterr().out
        assert opened == [inspect_server.session_url(srv.url, S1)]
        assert cli.main(["inspect", str(series_est.root), "--no-browser"]) == 0
        assert opened == [inspect_server.session_url(srv.url, S1)]


def test_cli_inspect_starts_a_server(series_est, monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)  # the command sets it for its server: undone after the test
    calls = []
    monkeypatch.setattr(inspect_server, "serve", lambda *a, **k: calls.append((a, k)) or 0)
    assert cli.main(["inspect", str(series_est.root), S2, "--port", "5123", "--no-browser"]) == 0
    assert calls == [((series_est.root, S2, 5123), {"open_browser": False, "quiet_busy": True})]
    assert cli.main(["inspect", str(series_est.root), "NOPE_X01_1D"]) == 2  # unknown session
    assert cli.main(["inspect", str(series_est.root), "--port", "70000"]) == 2
    assert len(calls) == 1


# --- the page never loses work silently; minors of the final fix wave ---

def _visible_next(srv, page):
    return locked(srv, page, lambda: page.next_button.visible and page.next_button.label)


def test_change_after_save_hides_next(series_est):
    """I1: an edit after Save results says so in the status line and hides Next until the next save (Next would
    drop the edit), also when the offer of Next was still on its way; a Revert change back to the saved state
    and a new save bring Next back."""
    with running(series_est) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        status = lambda: locked(srv, page, lambda: _div(page.doc, "status").text)  # noqa: E731

        # the edit right after the save, before the offer of Next arrives
        locked(srv, page, lambda: (click(page.doc, "Save results"), delete_row(page.doc, 2)))
        assert status() == inspect_server.CHANGED_SINCE_SAVE
        time.sleep(1.0)  # the offer has arrived meanwhile
        assert not locked(srv, page, lambda: page.next_button.visible or page.next_text.visible)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        assert until(lambda: _visible_next(srv, page)) == inspect_server.NEXT_LABEL.format(sid=S2)

        # an edit after the offer
        locked(srv, page, lambda: delete_row(page.doc, 0))
        assert status() == inspect_server.CHANGED_SINCE_SAVE
        assert not locked(srv, page, lambda: page.next_button.visible)
        # back to what was saved: Next again
        locked(srv, page, lambda: click(page.doc, "Revert change"))
        assert status() == inspect_server.SAVED.format(sid=S1)
        assert until(lambda: _visible_next(srv, page)) == inspect_server.NEXT_LABEL.format(sid=S2)
        locked(srv, page, lambda: click(page.doc, "Discard all"))
        assert status() == inspect_server.CHANGED_SINCE_SAVE and not _visible_next(srv, page)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        assert until(lambda: _visible_next(srv, page)) == inspect_server.NEXT_LABEL.format(sid=S2)
    assert list(_load(series_est.manual_result(S1)).idx_components_bad) == []  # the last save: Discard all


def test_saved_result_named_when_opened_again(series_est):
    """I2: a session whose "cleaned" decision counts and whose result exists says so above the window when it is
    opened again: the window shows the CNMF-E output, and Save results replaces that result."""
    paths = series_est
    with running(paths) as srv:
        with tab(srv, S1) as s:
            page = srv.page_of(s.id)
            assert not _div(page.doc, "saved_before").visible
            locked(srv, page, lambda: delete_row(page.doc, 2))
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            assert not _div(page.doc, "saved_before").visible
        d = load_record(paths, S1).decisions[-1]
        with tab(srv, S1) as s:  # a reload, or a reopen from the Inspection tab
            page = srv.page_of(s.id)
            until(lambda: _opened(page, S1), timeout=30)  # after the closed tab's socket is gone
            notice = _div(page.doc, "saved_before")
            when = datetime.fromisoformat(d.when).strftime("%Y-%m-%d %H:%M")
            assert notice.visible and notice.text == html.escape(
                inspect_server.SAVED_BEFORE.format(sid=S1, when=when, who=d.who), quote=False)
            assert len(locked(srv, page, lambda: _src(page.doc).data["idx"])) == 8  # the CNMF-E output
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            assert not locked(srv, page, lambda: notice.visible)  # replaced by this save
    # not-needed or no decision: nothing to name
    with running(paths) as srv, tab(srv, S2) as s:
        assert not _div(srv.page_of(s.id).doc, "saved_before").visible


def _src(doc):
    src, = [m for m in _models(doc, ColumnDataSource) if "idx" in m.data]
    return src


def test_out_of_date_cnmf_not_listed_and_not_saved(series_est):
    """I3: a session whose CNMF-E the batch would recompute (here: its parameters changed) is not in the waiting
    list and not offered by Next; opened by its address, it says so, and Save results writes nothing."""
    paths = series_est
    _set_cnmf(paths, S2, params_hash="an older run")
    text = inspect_server.CNMF_OUT_OF_DATE.format(sid=S2)
    with running(paths) as srv:
        assert srv.waiting() == [S1, S3]
        with tab(srv) as index:
            assert S2 not in index.document.roots[0].text and S3 in index.document.roots[0].text
        with tab(srv, S2) as s:
            page = srv.page_of(s.id)
            notice = _div(page.doc, "out_of_date")
            assert notice.visible and notice.text == html.escape(text, quote=False)
            assert save(srv, page) == text
            assert not paths.manual_result(S2).exists() and _decisions(paths, S2) == []
        with tab(srv, S1) as s:
            page = srv.page_of(s.id)
            assert not _div(page.doc, "out_of_date").visible
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            assert until(lambda: _visible_next(srv, page)) == inspect_server.NEXT_LABEL.format(sid=S3)


@pytest.mark.parametrize("state, text", [
    ("done", inspect_server.OUTDATED), ("running", inspect_server.OUTDATED_RUNNING),
    ("pending", inspect_server.OUTDATED_RESET), ("failed", inspect_server.OUTDATED_FAILED),
    ("gone", inspect_server.OUTDATED_GONE)])
def test_outdated_text_per_case(series_est, state, text):
    """M1: Save results after CNMF-E changed says what happened: recomputed ("Open it again": a new page opens the
    new output), running, reset or failed (no page can open it now: no "open again"), its output gone."""
    paths = series_est
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        later = datetime.now().isoformat(timespec="milliseconds")
        if state == "gone":
            paths.estimates(S1).unlink()
        else:
            _set_cnmf(paths, S1, state=State(state), finished=later)
        assert save(srv, page) == text
        assert ("Open it again" in text) == (state == "done")
        assert not paths.manual_result(S1).exists() and _decisions(paths, S1) == []


def _break_estimates(paths, sid):
    """sid's estimates file cut in half: CNMF-E done, but the file cannot be read."""
    f = paths.estimates(sid)
    data = f.read_bytes()
    f.write_bytes(data[:len(data) // 2])


def test_next_skips_a_session_that_cannot_be_opened(series_est, monkeypatch):
    """M2: S2 cannot be opened (its file is broken). Prepared in the background, its failure is remembered:
    after Save results Next skips it with a note and offers S3; it is not read again until its file changes."""
    paths = series_est
    _break_estimates(paths, S2)
    builds = []
    real = inspect_server.InspectServer.build
    monkeypatch.setattr(inspect_server.InspectServer, "build",
                        lambda self, sid: builds.append(sid) or real(self, sid))
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        until(lambda: srv.preparer.failed(S2) is not None, timeout=60)  # the background preparation failed
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        assert until(lambda: _visible_next(srv, page)) == inspect_server.NEXT_LABEL.format(sid=S3)
        note = locked(srv, page, lambda: page.notes_div.visible and page.notes_div.text)
        what = inspect_server.CANNOT_OPEN.format(sid=S2, what="UnpicklingError")
        assert note == html.escape(inspect_server.SKIPPED.format(text=what), quote=False), note
        locked(srv, page, lambda: click(page.doc, inspect_server.NEXT_LABEL.format(sid=S3)))
        until(lambda: _opened(page, S3), timeout=60)
        assert builds.count(S2) == 1  # remembered, not read again
        with pytest.raises(Exception):
            srv.preparer.take(S2)
        assert builds.count(S2) == 1
        shutil.copy(paths.estimates(S1), paths.estimates(S2))  # CNMF-E wrote it again: a new stamp
        assert srv.preparer.failed(S2) is None
        # read again (here or by a background preparation that saw the new stamp first)
        assert srv.preparer.take(S2).sid == S2 and builds.count(S2) >= 2


def test_next_click_skips_a_session_that_fails_then(series_est, monkeypatch):
    """M2: a failure found only when Next is clicked (nothing was prepared) leaves the page where it is, with a
    note, and offers the session after it."""
    paths = series_est
    _break_estimates(paths, S2)
    with running(paths) as srv:
        monkeypatch.setattr(srv.preparer, "prepare", lambda sid: None)  # no background preparation
        with tab(srv, S1) as s:
            page = srv.page_of(s.id)
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            label = until(lambda: _visible_next(srv, page))
            assert label == inspect_server.NEXT_LABEL.format(sid=S2)  # not known yet
            locked(srv, page, lambda: click(page.doc, label))
            assert until(lambda: _visible_next(srv, page) == inspect_server.NEXT_LABEL.format(sid=S3), timeout=60)
            assert page.sid == S1 and not page.busy
            note = locked(srv, page, lambda: page.notes_div.text)
            assert S2 in note and note.endswith(html.escape("Next skips it.", quote=False))
            locked(srv, page, lambda: click(page.doc, inspect_server.NEXT_LABEL.format(sid=S3)))
            until(lambda: _opened(page, S3), timeout=60)


def test_remaining_sessions_open_in_other_tabs(series_est):
    """M4: no free session left, but others wait in other tabs: the page names them instead of "No more"."""
    with running(series_est) as srv, tab(srv, S2) as a, tab(srv, S3) as b, tab(srv, S1) as s:
        assert _opened(srv.page_of(a.id), S2) and _opened(srv.page_of(b.id), S3)
        page = srv.page_of(s.id)
        assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
        text = until(lambda: locked(srv, page, lambda: page.next_text.visible and page.next_text.text))
        assert text == inspect_server.OPEN_ELSEWHERE_MANY.format(sids=f"{S2}, {S3}")
        assert not locked(srv, page, lambda: page.next_button.visible)
    with running(series_est) as srv:
        actions.gate_not_needed(series_est, [S3], "ann")
        with tab(srv, S2) as a, tab(srv, S1) as s:
            page = srv.page_of(s.id)
            locked(srv, page, lambda: delete_row(page.doc, 0))  # S1 was saved above: a change, then a save
            assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
            text = until(lambda: locked(srv, page, lambda: page.next_text.visible and page.next_text.text))
            assert text == inspect_server.OPEN_ELSEWHERE.format(sids=S2)


def test_autoinspected_input_refused_in_server_words(series_est):
    """M3: an autoinspected CNMF-E output is refused with the server's text (no "open 4_Estimates instead")."""
    paths = series_est
    est = _load(paths.estimates(S1))
    est.metrics_df = pd.DataFrame({"component_idx": range(8)})
    _dump(est, paths.estimates(S1))
    make_current(paths, S1, "cnmf")  # the file changed; the record follows it
    with running(paths) as srv, tab(srv, S1) as s:
        page = srv.page_of(s.id)
        texts = _divs(srv, page)
        assert html.escape(inspect_server.AUTOINSPECTED.format(sid=S1), quote=False) in texts
        assert "4_Estimates" not in " ".join(texts) and not _models(page.doc, Button)[1:]  # Next only


def test_part_timestamps_named_on_the_page(series_est, monkeypatch):
    """M7: session_timestamps' refusal (a part with fewer timestamp rows than frames) is caught with the other
    preparation errors and named in English with both numbers; its message holds a folder, never shown."""
    def short(paths, rec):
        raise ValueError("1_Raw/папка: 150 frames but 100 timestamp rows; rescan the series")

    monkeypatch.setattr(inspect_server, "session_timestamps", short)
    with running(series_est) as srv, tab(srv, S1) as s:
        texts = _divs(srv, srv.page_of(s.id))
    assert texts == [inspect_server.PART_TIMESTAMPS.format(sid=S1, n_frames=150, n_rows=100)]


def test_config_read_at_each_preparation(series_est):
    """M8: a new trace_points or warn_frac in the series config takes effect at the next preparation, without a
    restart; the page uses the settings its session was prepared with."""
    paths = series_est
    frac = _add_wave(paths, S1)
    srv = inspect_server.InspectServer(paths.root)
    first = srv.build(S1)
    assert first.bkapp_kwargs["trace_points"] == 2000 and first.wave_cfg["warn_frac"] == 0.025
    config.update_yaml(paths.series_config, {"inspection": {"trace_points": 500, "wave": {"warn_frac": frac / 2}}})
    again = srv.build(S1)
    assert again.bkapp_kwargs["trace_points"] == 500 and again.wave_cfg["warn_frac"] == frac / 2
    assert inspect_server.wave_hint(again.wave_frac, again.wave_cfg) is not None
    paths.series_config.write_text("inspection: [broken\n", encoding="utf-8")  # cannot be read now
    assert srv.build(S1).bkapp_kwargs["trace_points"] == 500  # the settings read last


def test_take_rebuilds_a_prepared_session_whose_stamp_changed():
    """M11: a prepared session whose stamp changed before take() (CNMF-E ran again) is built again."""
    stamps, built = {"A": 1}, []

    class Prep:
        def __init__(self, sid):
            self.sid, self.stamp = sid, stamps[sid]

    def build(sid):
        built.append(sid)
        return Prep(sid)

    p = inspect_server.Preparer(build, lambda sid: stamps[sid])
    try:
        p.prepare("A")
        until(lambda: p.prepared == "A", timeout=10)
        stamps["A"] = 2
        prep = p.take("A")
        assert built == ["A", "A"] and prep.stamp == 2
        p.prepare("A")
        until(lambda: p.prepared == "A", timeout=10)
        assert p.take("A").stamp == 2 and built == ["A", "A", "A"]  # unchanged: handed over as prepared
    finally:
        p.close(10)


def test_gate_cleaned_reads_one_record(series_est, monkeypatch):
    """M5: the server's decision reads only the session's record; an unknown or not-a-name session is refused
    as before, before any write."""
    paths = series_est
    paths.manual_result(S1).parent.mkdir(parents=True, exist_ok=True)
    paths.manual_result(S1).write_bytes(b"result")
    real = actions.registry.all_records
    monkeypatch.setattr(actions.registry, "all_records", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    res = actions.gate_cleaned(paths, S1, "ann")
    assert res.ok == [S1] and _decisions(paths, S1) == ["cleaned"]
    monkeypatch.setattr(actions.registry, "all_records", real)
    for sid in ("NOF_X99_9D", "../registry/" + S1, S1.lower()):  # R9: the file system ignores case; names don't
        with pytest.raises(actions.UnknownSession):
            actions.gate_cleaned(paths, sid, "ann")
    assert _decisions(paths, S1) == ["cleaned"]


# --- residual fixes of the final review ---

def test_edit_while_next_loads_keeps_the_page(series_est, monkeypatch):
    """R1: an edit made while Next is loading the next session keeps the page and the edit: the claim on the next
    session is let go and the session goes back to the Preparer; Save results, then Next opens it."""
    paths = series_est
    with running(paths) as srv:
        gate, entered = threading.Event(), threading.Event()
        real_take = srv.preparer.take

        def slow_take(sid):
            if sid == S2 and not gate.is_set():
                entered.set()
                assert gate.wait(60)
            return real_take(sid)

        monkeypatch.setattr(srv.preparer, "take", slow_take)
        try:
            with tab(srv, S1) as s:
                page = srv.page_of(s.id)
                status = lambda: locked(srv, page, lambda: _div(page.doc, "status").text)  # noqa: E731
                assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
                label = until(lambda: _visible_next(srv, page))
                assert label == inspect_server.NEXT_LABEL.format(sid=S2)
                locked(srv, page, lambda: click(page.doc, label))
                assert entered.wait(30)
                locked(srv, page, lambda: delete_row(page.doc, 0))  # while S2 loads
                assert status() == inspect_server.CHANGED_SINCE_SAVE
                gate.set()
                until(lambda: not page.busy, timeout=60)
                time.sleep(0.5)
                assert page.sid == S1 and page.claim is None and _opened(page, S1)
                assert status() == inspect_server.CHANGED_SINCE_SAVE and not _visible_next(srv, page)
                assert srv.preparer.prepared == S2  # given back, not lost
                assert list(_load(paths.manual_result(S1)).idx_components_bad) == []  # the save before the edit
                assert save(srv, page) == inspect_server.SAVED.format(sid=S1)
                label = until(lambda: _visible_next(srv, page))
                locked(srv, page, lambda: click(page.doc, label))
                until(lambda: _opened(page, S2), timeout=60)
        finally:
            gate.set()
    assert list(_load(paths.manual_result(S1)).idx_components_bad) == [0]  # the edit was saved


def _tb_len(e):
    n, tb = 0, e.__traceback__
    while tb is not None:
        n, tb = n + 1, tb.tb_next
    return n


def test_remembered_failure_pins_nothing(series_est, monkeypatch):
    """R2: a remembered preparation failure keeps no frame of the build: the estimates the build had loaded
    (LoadEstimates refuses short timestamps after unpickling) are freed, and a re-raise does not lengthen the
    traceback."""
    paths = series_est
    rec = load_record(paths, S3)
    ts = paths.root / rec.parts[0].timestamps_file
    ts.write_text("".join(f"{i * 500000}\n" for i in range(250)), encoding="utf-8")
    loaded, real_load = [], bm_examinator.pickle.load

    def load(*args, **kwargs):
        obj = real_load(*args, **kwargs)
        if type(obj).__name__ == "Estimates":
            loaded.append(weakref.ref(obj))
        return obj

    srv = inspect_server.InspectServer(paths.root)
    p = inspect_server.Preparer(srv.build, srv.stamp, srv.failure_key)
    lengths = []
    try:
        monkeypatch.setattr(bm_examinator.pickle, "load", load)
        for _ in range(3):  # the build, then the remembered failure twice
            try:
                p.take(S3)
            except inspect_server._ShortTimestamps as e:
                lengths.append(_tb_len(e))
                assert (e.n_ts, e.n_frames) == (250, 300)
        monkeypatch.undo()
        assert len(loaded) == 1 and len(lengths) == 3  # read once
        assert lengths[1] == lengths[2]
        gc.collect()
        assert loaded[0]() is None, "the remembered failure keeps the estimates"
    finally:
        p.close(10)


def test_unschedulable_session_says_so(series_est):
    """R3: a session the batch cannot schedule (here: its config cannot be read) is not called out of date: the
    page and Save results name the problem, in the series' own terms (no path of the series folder)."""
    paths = series_est
    paths.session_config(S2).parent.mkdir(parents=True, exist_ok=True)
    paths.session_config(S2).write_text("cnmf: [broken\n", encoding="utf-8")
    with running(paths) as srv:
        assert srv.waiting() == [S1, S3]
        with tab(srv, S2) as s:
            page = srv.page_of(s.id)
            notice = _div(page.doc, "out_of_date")
            text = html.unescape(notice.text)
            assert notice.visible and text.startswith(f"{S2} cannot be scheduled now: config: ConfigError: ")
            assert text.endswith(". A result saved here may not count; fix the problem in Monitor first.")
            assert "out of date" not in text and not CYRILLIC.search(text)
            assert save(srv, page) == notice.text  # both escaped for the page
            assert not paths.manual_result(S2).exists() and _decisions(paths, S2) == []


def test_reload_while_opening_builds_once(series_est, monkeypatch):
    """R4: a reload while the page says "Opening" waits for the build already running instead of starting a
    second one (each holds the whole session in memory and writes the same cache)."""
    paths = series_est
    monkeypatch.setattr(inspect_server, "CONNECT_GRACE_S", 1.0)  # the first GET page never connects
    real = inspect_server.InspectServer.build
    gate = threading.Event()
    active, peak, builds, lock = [0], [0], [], threading.Lock()

    def slow(self, sid):
        if sid != S1:
            return real(self, sid)
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            builds.append(sid)
        try:
            assert gate.wait(60)
            return real(self, sid)
        finally:
            with lock:
                active[0] -= 1

    monkeypatch.setattr(inspect_server.InspectServer, "build", slow)
    try:
        with running(paths) as srv:
            pages = []
            for _ in range(2):  # the first request, then the reload
                before = {id(p) for p in srv.pages()}
                with urllib.request.urlopen(inspect_server.session_url(srv.url, S1), timeout=60) as r:
                    assert r.status == 200
                pages += [p for p in srv.pages() if id(p) not in before]
            second = pages[1]
            until(lambda: second.sid == S1, timeout=30)  # it took the session once the first page's grace ended
            time.sleep(1.0)
            with lock:
                assert active[0] == 1
            gate.set()
            until(lambda: _opened(second, S1), timeout=60)
        assert peak[0] == 1 and builds == [S1]
    finally:
        gate.set()


# --- speed on hour-long sessions (slow) ---

HOUR = 107_107


def _hour_estimates(n=194, dims=(608, 608), T=HOUR):
    """tests/test_examinator_speed.py's synthetic session (the size of the real file the phase 5 page measured):
    n round footprints on a grid, AR(1) traces of sparse spikes."""
    from caiman.source_extraction.cnmf.estimates import Estimates
    from caiman.source_extraction.cnmf.params import CNMFParams
    from scipy.signal import lfilter
    from scipy.sparse import csc_matrix
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(n)))
    step = dims[0] / (side + 1)
    r, sigma = 12, 4.0
    rows, cols, vals = [], [], []
    for k in range(n):
        cy, cx = int(step * (1 + k // side)), int(step * (1 + k % side))
        yy, xx = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        rows.append((yy + xx * dims[0]).ravel())
        cols.append(np.full(g.size, k))
        vals.append(g.ravel())
    A = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dims[0] * dims[1], n))
    spikes = (rng.random((n, T)) < 0.01).astype(float)
    C = lfilter([1.0], [1.0, -0.95], spikes, axis=1) + 0.01 * np.arange(n)[:, None]
    est = Estimates(A=A, C=C, b=None, f=None, dims=dims)
    est.YrA = 0.01 * rng.normal(size=(n, T))
    est.R = est.YrA
    est.S = spikes
    k = np.arange(n)
    est.bl, est.c1, est.neurons_sn = 0.01 * k, 0.02 * k, 1.0 + 0.1 * k
    est.g = (0.9 - 0.001 * k)[:, None]
    est.idx_components = np.arange(n)
    est.idx_components_bad = np.array([], dtype=int)
    est.SNR_comp = 1.0 + 5.0 * rng.random(n)
    est.r_values = rng.random(n)
    est.imax = np.ones(dims)
    est.cnmf_dict = CNMFParams(params_dict={"p": 1, "fr": 20}).to_dict()
    return est


def _hour_series(root):
    """S1 and S2 of an hour each: the synthetic session of test_examinator_speed (194 components on 608 x 608,
    107 107 frames), registered with CNMF-E done; no video (the server reads none)."""
    from bearmind import naming
    from bearmind.registry import STAGES, RawPart, SessionRecord, StageRecord, save_record
    paths = SeriesPaths(root)
    est = _hour_estimates()
    finished = (datetime.now() - timedelta(minutes=5)).isoformat(timespec="milliseconds")
    for sid in (S1, S2):
        ts = root / "0_TimeStamps" / f"{sid}_Mini_TS.csv"
        ts.parent.mkdir(parents=True, exist_ok=True)
        ts.write_text("".join(f"{i * 500000}\n" for i in range(HOUR)), encoding="utf-8")
        f = paths.estimates(sid)
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(est, fh, protocol=pickle.HIGHEST_PROTOCOL)
        part = RawPart(folder=f"1_Raw/{sid}", chunks=[], n_frames=HOUR, shape=(608, 608),
                       timestamps_file=paths.rel(ts), ts_format="bonsai_ticks", start_ms=None, metadata_file=None)
        rec = SessionRecord(sid=sid, mouse=naming.mouse_id(sid), mode="bonsai", parts=[part], n_frames=HOUR,
                            fps=20.0, fps_info={}, anomalies=[], decisions=[], excluded=False,
                            stages={s: StageRecord() for s in STAGES})
        rec.stages["cnmf"] = StageRecord(state=State.DONE, finished=finished, outputs={"estimates": paths.rel(f)})
        mark_current(paths, rec, ("crop", "mc", "thresholds", "cnmf"))  # the server lists only current ones
        save_record(paths, rec)
    return paths


@pytest.mark.slow
def test_speed_hour_session(tmp_path, monkeypatch):
    """Opening, Save results and Next on hour-long sessions on a running server: the Python time of the page's
    handlers, no browser (a page opened by GET has a session without a connection). Next and the second Save
    results run while the side files of the first save are still written; the writer competes for the GIL."""
    from bokeh.protocol import Protocol
    from tests.examinator_helpers import message_bytes

    monkeypatch.setattr(inspect_server, "CONNECT_GRACE_S", 0.0)  # a page opened by GET never holds its session
    monkeypatch.setattr(inspect_server, "UNUSED_SESSION_MS", 3_600_000)  # and is not discarded meanwhile
    paths = _hour_series(tmp_path / "hour")
    numbers = {}
    with running(paths) as srv:
        def get(sid):
            """(page, seconds from the request until the window is on the page): a session that is not
            prepared answers with "Opening" at once and is built after it (K1c)."""
            before = {id(p) for p in srv.pages()}
            t0 = time.perf_counter()
            with urllib.request.urlopen(inspect_server.session_url(srv.url, sid), timeout=600) as r:
                assert r.status == 200
            page, = [p for p in srv.pages() if id(p) not in before]
            until(lambda: _opened(page, sid), timeout=600, step=0.002)
            return page, time.perf_counter() - t0

        def timed_save(page):
            def run():
                t0 = time.perf_counter()
                click(page.doc, "Save results")
                return time.perf_counter() - t0, _div(page.doc, "status").text
            dt, text = locked(srv, page, run)
            assert text == inspect_server.SAVED.format(sid=page.sid), text
            return dt

        page, numbers["open_cold_s"] = get(S1)  # no contour cache, not prepared
        numbers["first_document_bytes"] = message_bytes(Protocol().create("PULL-DOC-REPLY", "0", page.doc))
        until(lambda: srv.preparer.prepared == S2, timeout=300)
        _, numbers["open_prepared_s"] = get(S2)
        until(lambda: srv.preparer.prepared == S1, timeout=300)  # the next after S2
        srv.preparer.take(S1)  # dropped: the next GET prepares it, with the contour cache
        page, numbers["open_cached_s"] = get(S1)
        until(lambda: srv.preparer.prepared == S2, timeout=300)

        numbers["save_s"] = timed_save(page)
        writer, = srv._side_futures
        until(lambda: locked(srv, page, lambda: page.next_button.visible), timeout=60)
        t0 = time.perf_counter()
        locked(srv, page, lambda: click(page.doc, inspect_server.NEXT_LABEL.format(sid=S2)))
        until(lambda: _opened(page, S2), timeout=120, step=0.002)
        numbers["next_s"] = time.perf_counter() - t0
        numbers["writer_busy_after_next"] = not writer.done()
        numbers["save_during_side_files_s"] = timed_save(page)
        numbers["writer_busy_after_second_save"] = not writer.done()
        t0 = time.perf_counter()
        srv.next_waiting(S2)
        numbers["next_waiting_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        writer.result(timeout=600)
        numbers["side_files_left_s"] = time.perf_counter() - t0
    print("[speed] " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in numbers.items()))
    assert numbers["writer_busy_after_next"] and numbers["writer_busy_after_second_save"]
    assert numbers["open_cold_s"] <= 1.5, numbers  # the calcium-wave share is computed on this path
    assert numbers["open_prepared_s"] <= 1.5, numbers
    assert numbers["next_s"] <= 1.0, numbers
    assert numbers["save_during_side_files_s"] <= 0.5, numbers
