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
from bearmind import actions, cli, config, inspect_server, monitor, runner, scan, wave
from bearmind.paths import SeriesPaths
from bearmind.registry import State, load_record, update_record
from bearmind.stages import gate
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
            yield session
        finally:
            session.close()
    finally:
        _TAB_LOOPS.pop()
        asyncio.set_event_loop(_TAB_LOOPS[-1] if _TAB_LOOPS else None)
        io.close(all_fds=True)


_TAB_LOOPS = []  # the loops of the open tabs, innermost last


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


def test_server_exits_when_idle(series_est):
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
        t.join(idle + 5)
        assert not t.is_alive() and time.monotonic() - closed <= idle + 5
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


def _add_wave(paths, sid, start=100, frames=30):
    """A calcium wave in sid's estimates: at frame `start` every component rises to 20 times the largest noise SD
    and decays (10 frames) over `frames` frames. Returns the session's wave share."""
    f = paths.estimates(sid)
    est = _load(f)
    est.C = est.C.copy()
    est.C[:, start:start + frames] += 20.0 * np.max(est.neurons_sn) * np.exp(-np.arange(frames) / 10.0)
    with open(f, "wb") as fh:
        pickle.dump(est, fh)
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
        banner = _div(s.document, "wave")
        assert banner.visible and banner.text == html.escape(wave.wave_text(frac, 0.3))
        header = s.document.roots[0]
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
        banner = _div(s.document, "wave")
        assert not banner.visible and banner.text == ""


def test_no_banner_on_ordinary_sessions(series_est):
    """With the default threshold the synthetic sessions (independent sparse transients) show no hint."""
    warn = config.load_defaults()["inspection"]["wave"]["warn_frac"]
    assert warn is not None
    with running(series_est) as srv:
        for sid in (S1, S2):
            with tab(srv, sid) as s:
                page = srv.page_of(s.id)
                assert _opened(page, sid) and page.prepared.wave_frac <= warn
                banner = _div(s.document, "wave")
                assert not banner.visible and banner.text == ""


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
    srv.cfg["wave"] = {**srv.wave_cfg, "k_mad": 4.0}  # another rule
    assert srv.build(S1).wave_frac != frac and len(calls) == 2
    assert len(srv.build(S1).contours) == 8 and len(calls) == 2  # cached again, with the contours
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
            texts = [d.text for d in _models(s.document, Div)]
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

    def failing(est, folder, base, sigma=3):
        try:
            raise PermissionError(13, DENIED, str(target))
        except PermissionError as e:
            raise RuntimeError(f"side files in {folder} not all written ({target.name}: {DENIED})") from e

    monkeypatch.setattr(bm_examinator, "write_side_files", failing)
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


def test_newer_save_skips_queued_side_files(series_est, monkeypatch):
    """M5: side files of a save that has not started yet are dropped when the same session is saved again;
    only the last save's files matter."""
    paths = series_est
    started, release, written = threading.Event(), threading.Event(), []
    real = bm_examinator.write_side_files

    def gated(est, folder, base, sigma=3):
        written.append([int(c) for c in bm_examinator.kept_for_side_files(est)])
        if len(written) == 1:
            started.set()
            assert release.wait(60)
        real(est, folder, base, sigma)

    monkeypatch.setattr(bm_examinator, "write_side_files", gated)
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
    real = bm_examinator.write_side_files

    def gated(est, folder, base, sigma=3):
        written.append([int(c) for c in bm_examinator.kept_for_side_files(est)])
        if len(written) == 1:
            started.set()
            assert release.wait(60)
        real(est, folder, base, sigma)

    def cannot_queue(*args, **kwargs):
        raise RuntimeError("cannot schedule new futures after shutdown")

    monkeypatch.setattr(bm_examinator, "write_side_files", gated)
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
    assert written == [list(range(8)), list(range(1, 8))]  # the second save's files were still written


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
    real = bm_examinator.write_side_files

    def slow(*args, **kwargs):
        assert release.wait(60)
        return real(*args, **kwargs)

    monkeypatch.setattr(bm_examinator, "write_side_files", slow)
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


def test_empty_lock_means_wait(tmp_path):
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
    assert not st.alive and not st.starting and not st.available and time.monotonic() - t0 < 2


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


def test_lock_left_by_a_stopped_server(series_est):
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
        assert not inspect_server.wait_ready(paths).available and time.monotonic() - t0 < 1
        # a new server takes the series
        with running(paths) as srv2:
            assert inspect_server.server_state(paths).available and srv2.url
    finally:
        srv.request_stop()
        t.join(180)
        assert not t.is_alive()


def test_stopping_wait_is_bounded_once(series_est, monkeypatch, capsys):
    """N5: a server that does not finish stopping: the inspect command waits once, for STOP_WAIT_S, then fails
    naming the lock file (not a second wait, not a second server)."""
    paths = series_est
    monkeypatch.setattr(inspect_server, "STOP_WAIT_S", 1.0)

    def must_not_serve(*args, **kwargs):
        raise AssertionError("a server was started while another one holds the series")

    monkeypatch.setattr(inspect_server, "serve", must_not_serve)
    fh = runner._take_lock(paths.inspection_lock, {"url": None, "current": [], "prepared": None, "stopping": True})
    assert fh is not None
    try:
        assert inspect_server.server_state(paths).stopping
        capsys.readouterr()
        t0 = time.monotonic()
        assert cli.main(["inspect", str(paths.root), "--no-browser"]) == 3
        assert time.monotonic() - t0 < 1.9
        out = capsys.readouterr().out
        assert "[FAILED]" in out and str(paths.inspection_lock) in out
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
                texts += _texts(s.document)
        with tab(srv, S1) as s, tab(srv, S1) as second:
            page = srv.page_of(s.id)
            texts += _texts(s.document) + _texts(second.document)  # the second: "opening", then refused
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
            before = {id(p) for p in srv.pages()}
            t0 = time.perf_counter()
            with urllib.request.urlopen(inspect_server.session_url(srv.url, sid), timeout=600) as r:
                assert r.status == 200
            dt = time.perf_counter() - t0
            page, = [p for p in srv.pages() if id(p) not in before]
            assert _opened(page, sid)
            return page, dt

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
