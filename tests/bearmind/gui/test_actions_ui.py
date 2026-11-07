"""Monitor tab parts 5-6 (warnings, log) and the Actions menu: texts, confirmations, writes through actions."""
import re

import pytest
from PySide6.QtCore import QItemSelectionModel
from PySide6.QtWidgets import QLabel, QPushButton

from bearmind import monitor, registry
from bearmind.gui.logs import LogView
from bearmind.gui.monitor_tab import ACTIONS, MonitorTab
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, Check, StageRecord, State
from tests.bearmind.gui.test_monitor_tab import done, put, spawn, stop

S1, S2, S3, S4 = "LT1_J01_1D", "LT1_J01_2D", "LT1_J02_1D", "LT1_J02_2D"
CYRILLIC = "[Ѐ-ӿ]"


def _series(tmp_path, with_warnings=True):
    paths = SeriesPaths(tmp_path / "series")
    paths.registry_file("x").parent.mkdir(parents=True)
    base = {"crop": done(box="512x480@0,0")}
    shift = [Check("mc", "mc_max_shift_px", "warning", 18.0, "max shift 18 px (warn above 10)")]
    ok_mc = StageRecord(State.DONE, metrics={"max_shift_px": 3.0})
    bad_mc = StageRecord(State.DONE, metrics={"max_shift_px": 18.0}, checks=shift)
    failed = StageRecord(State.FAILED, error="boom")
    put(paths, S1, {**base, "mc": bad_mc if with_warnings else ok_mc})
    put(paths, S2, {**base, "mc": ok_mc, "thresholds": done(), "cnmf": failed if with_warnings else StageRecord()})
    put(paths, S3, {**base, "mc": ok_mc})
    put(paths, S4, base)
    return paths


@pytest.fixture
def series4(tmp_path):
    return _series(tmp_path)


@pytest.fixture
def series4_clean(tmp_path):
    return _series(tmp_path, with_warnings=False)


class Ui:
    """A tab with every dialog replaced; records what the window asked, showed and put in the status bar."""

    def __init__(self, paths, answer=True, stage=None, file=None):
        self.asked, self.shown, self.status = [], [], []
        self.stage, self.file = stage, file
        self.tab = MonitorTab(paths.root, confirm=lambda t: (self.asked.append(t), answer)[1],
                              choose_stage=lambda: self.stage, choose_file=lambda d: self.file or d,
                              message=lambda title, text: self.shown.append(text))
        self.tab.status_message.connect(self.status.append)

    def select(self, *sids):
        sel = self.tab.table.selectionModel()
        sel.clearSelection()
        for i, sid in enumerate(self.tab._sids):
            if sid in sids:
                sel.select(self.tab.model.index(i, 0), QItemSelectionModel.SelectionFlag.Select
                           | QItemSelectionModel.SelectionFlag.Rows)
        assert sorted(self.tab.selected_sids()) == sorted(sids)


def _retry_buttons(tab):
    return [b for b in tab.warnings_list.findChildren(QPushButton) if b.text() == "Retry"]


def _decisions(paths, sid):
    return [(d.what, d.value) for d in registry.load_record(paths, sid).decisions]


def test_panel_lists_errors_first_with_retry(qapp, series4):
    t = Ui(series4).tab
    texts = [lab.text() for lab in t.warnings_list.findChildren(QLabel)]
    assert "ERROR" in texts and "WARN" in texts and texts.index("ERROR") < texts.index("WARN")
    assert any(S2 in x and "CNMF-E" in x for x in texts) and any(S1 in x and "MC" in x for x in texts)
    assert len(_retry_buttons(t)) == 1  # only the failed stage has Retry


def test_panel_text_does_not_repeat_the_stage(qapp, series4):
    """R4: the bold "<sid> · CNMF-E" label is followed by what happened, not by the stage again."""
    texts = [lab.text() for lab in Ui(series4).tab.warnings_list.findChildren(QLabel)]
    assert "boom. Other sessions continue." in texts
    assert not [x for x in texts if x.startswith(("CNMF-E:", "MC:"))]


def test_already_passed_message_asks_for_nothing(qapp, series4):
    """M8: the window has no force path, so its text does not ask to confirm."""
    registry.update_record(series4, S1, lambda r: r.stages.update({"gate": done(gate="not_needed")}))
    ui = Ui(series4)
    ui.select(S1)
    ui.tab.run_action("inspection_not_needed")
    msg = next(x for x in ui.shown if S1 in x)
    assert "already passed" in msg and "confirm" not in msg.lower()


def test_log_poll_keeps_the_reading_position(qapp, tmp_path):
    """M9: new lines scroll the view only when it was at the end."""
    f = tmp_path / "a.log"
    f.write_text("".join(f"line {i}\n" for i in range(300)), encoding="utf-8")
    v = LogView()
    v.resize(400, 150)
    v.show()
    v.show_log(f)
    qapp.processEvents()
    bar = v.verticalScrollBar()
    assert bar.maximum() > 0 and bar.value() == bar.maximum()
    bar.setValue(10)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("new 1\nnew 2\n")
    v.poll()
    qapp.processEvents()
    assert bar.value() == 10
    bar.setValue(bar.maximum())
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("new 3\n")
    v.poll()
    qapp.processEvents()
    assert bar.value() == bar.maximum() and v.toPlainText().endswith("new 3")
    v.close()


def test_panel_header_counts_sessions(qapp, series4):
    assert Ui(series4).tab.warnings_header.text() == "Warnings — 2 sessions"


def test_panel_empty(qapp, series4_clean):
    t = Ui(series4_clean).tab
    assert "No warnings" in [lab.text() for lab in t.warnings_list.findChildren(QLabel)]
    assert not _retry_buttons(t)


def test_retry_button_resets_failed_stage(qapp, series4):
    ui = Ui(series4)
    _retry_buttons(ui.tab)[0].click()
    rec = registry.load_record(series4, S2)
    assert all(rec.stages[s].state is State.PENDING for s in STAGES[STAGES.index("cnmf"):])
    assert f"{S2}: reset from CNMF-E" in ui.status[-1] and "Start the batch" in ui.status[-1]
    assert not _retry_buttons(ui.tab) and ui.tab.warnings_header.text() == "Warnings — 1 session"


def test_retry_with_live_batch_does_not_say_start(qapp, series4):
    ui = Ui(series4)
    p = spawn(series4, S3, "mc", "running")
    try:
        ui.tab.refresh()
        _retry_buttons(ui.tab)[0].click()
        assert "Start the batch" not in ui.status[-1] and "reset from" in ui.status[-1]
    finally:
        stop(p)


def test_retry_does_not_ask_confirmation(qapp, series4):
    ui = Ui(series4, answer=False)
    _retry_buttons(ui.tab)[0].click()
    assert ui.asked == []
    assert registry.load_record(series4, S2).stages["cnmf"].state is State.PENDING


def test_panel_texts_are_english(qapp, series4):
    ui = Ui(series4)
    ui.tab.log_view.show_log(series4.series_log)
    t = ui.tab
    texts = [w.text() for w in t.findChildren(QLabel)] + [w.text() for w in t.findChildren(QPushButton)]
    texts += [a.text() for a in t.menu_actions]
    assert texts and not [x for x in texts if re.search(CYRILLIC, x)]


def test_actions_are_the_documented_set(qapp, series4):
    assert ACTIONS == ("exclude", "include", "accept_mismatch", "rerun_from", "retry_failed",
                       "inspection_not_needed", "inspection_not_needed_all", "export_csv")
    assert len(MonitorTab(series4.root).menu_actions) == len(ACTIONS)


def test_exclude_selected(qapp, series4):
    ui = Ui(series4)
    ui.select(S1)
    ui.tab.run_action("exclude")
    assert S1 in ui.asked[0]
    assert registry.load_record(series4, S1).excluded
    assert "Excluded" in ui.status[-1]


def test_confirm_no_writes_nothing(qapp, series4):
    ui = Ui(series4, answer=False)
    ui.select(S1)
    before = _decisions(series4, S1)
    for name in ("exclude", "include", "accept_mismatch", "inspection_not_needed", "retry_failed"):
        ui.tab.run_action(name)
    ui.stage = "cnmf"
    ui.tab.run_action("rerun_from")
    assert len(ui.asked) == 6 and _decisions(series4, S1) == before
    assert not registry.load_record(series4, S1).excluded


def test_no_selection_says_so_and_asks_nothing(qapp, series4):
    ui = Ui(series4)
    ui.tab.run_action("exclude")
    assert ui.asked == [] and "Select" in ui.status[-1]


def test_rerun_with_stage_choice(qapp, series4):
    ui = Ui(series4, stage="cnmf")
    ui.select(S1, S3)
    ui.tab.run_action("rerun_from")
    assert ui.asked == [f"Sessions {S1}, {S3} will be recomputed from CNMF-E. "
                        "Old results stay until the new ones are ready."]
    for sid in (S1, S3):
        assert all(registry.load_record(series4, sid).stages[s].state is State.PENDING
                   for s in STAGES[STAGES.index("cnmf"):])
    assert registry.load_record(series4, S1).stages["mc"].state is State.DONE


def test_rerun_cancelled_stage_choice(qapp, series4):
    ui = Ui(series4, stage=None)
    ui.select(S1)
    ui.tab.run_action("rerun_from")
    assert ui.asked == [] and _decisions(series4, S1) == []


def test_inspection_not_needed_all_waiting(qapp, series4):
    registry.update_record(series4, S3, lambda r: r.stages.update({"cnmf": done(n_components=50)}))
    ui = Ui(series4)
    ui.tab.run_action("inspection_not_needed_all")
    assert S3 in ui.asked[0]
    assert any(v == "not_needed" for _, v in _decisions(series4, S3))
    assert not any(v == "not_needed" for _, v in _decisions(series4, S1))


def test_failures_shown_in_english(qapp, series4):
    ui = Ui(series4)
    ui.select(S1)  # no failed stage
    ui.tab.run_action("retry_failed")
    assert any("has no failed stage" in s and S1 in s for s in ui.shown)
    assert not any(re.search(CYRILLIC, s) for s in ui.shown)


def test_retry_failed_without_selection_takes_all_failed(qapp, series4):
    ui = Ui(series4)
    ui.tab.run_action("retry_failed")
    assert S2 in ui.asked[0]
    assert registry.load_record(series4, S2).stages["cnmf"].state is State.PENDING


def test_warnings_in_message(qapp, series4):
    registry.update_record(series4, S4, lambda r: setattr(r, "excluded", True))
    ui = Ui(series4)
    ui.select(S4)
    ui.tab.run_action("inspection_not_needed")
    assert any(S4 in s and "excluded" in s for s in ui.shown)


def test_confirmed_set_is_the_written_set(qapp, series4):
    ui = Ui(series4)
    # a session starts failing between the confirmation and the write: it must not be reset
    def confirm(text):
        ui.asked.append(text)
        registry.update_record(series4, S3, lambda r: r.stages.update({"mc": StageRecord(State.FAILED, error="x")}))
        return True

    ui.tab._confirm = confirm
    ui.tab.run_action("retry_failed")
    assert S2 in ui.asked[0] and S3 not in ui.asked[0]
    assert registry.load_record(series4, S3).stages["mc"].state is State.FAILED
    assert registry.load_record(series4, S2).stages["cnmf"].state is State.PENDING


def test_confirmed_set_is_the_written_set_inspection_all(qapp, series4):
    registry.update_record(series4, S3, lambda r: r.stages.update({"cnmf": done(n_components=50)}))
    ui = Ui(series4)

    def confirm(text):
        ui.asked.append(text)
        registry.update_record(series4, S1, lambda r: r.stages.update({"cnmf": done(n_components=9)}))
        return True

    ui.tab._confirm = confirm
    ui.tab.run_action("inspection_not_needed_all")
    assert S3 in ui.asked[0] and S1 not in ui.asked[0]
    assert not any(v == "not_needed" for _, v in _decisions(series4, S1))


def test_confirmation_grammar(qapp, series4):
    ui = Ui(series4, answer=False)
    ui.select(S1)
    ui.tab.run_action("exclude")
    ui.select(S1, S3)
    ui.tab.run_action("exclude")
    assert ui.asked[0] == f"Session {S1} will be excluded: the runner skips it until you include it again."
    assert ui.asked[1] == (f"Sessions {S1}, {S3} will be excluded: the runner skips them until you include "
                           "them again.")
    ui.select(S1)
    ui.tab.run_action("accept_mismatch")
    assert "the session is used as it is" in ui.asked[2]


def test_retry_button_unknown_session_does_not_crash(qapp, series4):
    ui = Ui(series4)
    btn = _retry_buttons(ui.tab)[0]
    (series4.registry_file(S2)).unlink()
    btn.click()
    assert "not in the series" in ui.status[-1]


def test_refresh_repeat_with_other_error_logs_traceback(qapp, series4, monkeypatch, caplog):
    t = MonitorTab(series4.root, report=lambda *e: None)
    errs = iter([RuntimeError("a"), RuntimeError("a"), ValueError("b")])

    def boom(*a, **k):
        raise next(errs)

    monkeypatch.setattr(monitor, "series_view", boom)
    with caplog.at_level("WARNING", logger="bearmind"):
        t.refresh()
        t.refresh()
        t.refresh()
    assert any(r.exc_info and r.exc_info[0] is ValueError for r in caplog.records)
    assert not any(r.exc_info and r.exc_info[0] is RuntimeError for r in caplog.records)


def test_programming_error_in_refresh_fails_the_test(qapp, series4, monkeypatch):
    t = MonitorTab(series4.root)
    monkeypatch.setattr(monitor, "series_view", lambda *a, **k: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        t.refresh()


def test_export_csv(qapp, series4, tmp_path):
    dest = tmp_path / "out" / "s.csv"
    dest.parent.mkdir()
    ui = Ui(series4, file=dest)
    ui.tab.run_action("export_csv")
    assert dest.is_file() and "sid" in dest.read_text(encoding="utf-8").splitlines()[0]
    assert str(dest) in ui.status[-1]


def test_export_overwrite_is_not_asked_twice_and_refusal_writes_nothing(qapp, series4, tmp_path):
    dest = tmp_path / "s.csv"
    dest.write_text("old", encoding="utf-8")
    ui = Ui(series4, answer=False, file=dest)
    ui.tab.run_action("export_csv")  # the dialog accepted: no second question
    assert ui.asked == [] and dest.read_text(encoding="utf-8") != "old"
    dest.write_text("old", encoding="utf-8")
    t = MonitorTab(series4.root, choose_file=lambda d: None)
    t.run_action("export_csv")  # the dialog was refused
    assert dest.read_text(encoding="utf-8") == "old"


def test_export_default_name(qapp, series4):
    seen = []
    t = MonitorTab(series4.root, choose_file=lambda d: (seen.append(d), None)[1])
    t.run_action("export_csv")
    assert seen == [series4.root / f"{series4.root.name}_summary.csv"]


def test_log_tail_and_switch(qapp, series4):
    series4.logs_dir.mkdir(parents=True, exist_ok=True)
    series4.session_log(S1).write_text("\n".join(f"s1 line {i}" for i in range(600)) + "\n", encoding="utf-8")
    series4.session_log(S2).write_text("s2 only\n", encoding="utf-8")
    series4.series_log.write_text("series line\n", encoding="utf-8")
    ui = Ui(series4)
    t = ui.tab
    assert t.log_title.text() == "Log — series.log" and "series line" in t.log_view.toPlainText()
    ui.select(S1)
    assert t.log_title.text() == f"Log — {S1}"
    lines = t.log_view.toPlainText().splitlines()
    assert len(lines) == 500 and lines[0] == "s1 line 100" and lines[-1] == "s1 line 599"
    ui.select(S2)
    assert t.log_view.toPlainText().strip() == "s2 only"
    t.table.clearSelection()
    assert "series line" in t.log_view.toPlainText()


def test_log_missing_file_is_empty(qapp, tmp_path):
    v = LogView()
    v.show_log(tmp_path / "none.log")
    assert v.toPlainText() == ""


def test_log_poll_appends_only_new(qapp, tmp_path):
    f = tmp_path / "a.log"
    f.write_text("one\n", encoding="utf-8")
    v = LogView()
    v.show_log(f)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("two\nthree\n")
    v.poll()
    assert v.toPlainText().splitlines() == ["one", "two", "three"]


def test_log_poll_waits_for_the_end_of_a_line(qapp, tmp_path):
    f = tmp_path / "a.log"
    f.write_text("one\n", encoding="utf-8")
    v = LogView()
    v.show_log(f)
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("tw")
    v.poll()
    assert v.toPlainText().splitlines() == ["one"]
    with open(f, "a", encoding="utf-8") as fh:
        fh.write("o\n")
    v.poll()
    assert v.toPlainText().splitlines() == ["one", "two"]


def test_log_poll_crlf_has_no_empty_lines(qapp, tmp_path):
    f = tmp_path / "a.log"
    f.write_bytes(b"one\r\n")
    v = LogView()
    v.show_log(f)
    with open(f, "ab") as fh:
        fh.write(b"two\r\nthree\r\n")
    v.poll()
    with open(f, "ab") as fh:
        fh.write(b"four\r\n")
    v.poll()
    assert v.toPlainText() == "one\ntwo\nthree\nfour"
    assert "\r" not in v.toPlainText()


def test_log_truncated_rereads(qapp, tmp_path):
    f = tmp_path / "a.log"
    f.write_text("old1\nold2\nold3\n", encoding="utf-8")
    v = LogView()
    v.show_log(f)
    f.write_text("new\n", encoding="utf-8")
    v.poll()
    assert v.toPlainText().splitlines() == ["new"]


def test_log_does_not_read_whole_file(qapp, tmp_path, monkeypatch):
    f = tmp_path / "big.log"
    with open(f, "wb") as fh:
        fh.truncate(50 * 1024 * 1024)
        fh.seek(0, 2)  # truncate keeps the position
        fh.write(b"\nlast line\n")
    read = []
    real = open

    class Counting:  # BufferedReader does not accept attribute assignment, so wrap it
        def __init__(self, fh):
            self._fh = fh

        def read(self, n=-1):
            data = self._fh.read(n)
            read.append(len(data))
            return data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._fh.close()

        def __getattr__(self, name):
            return getattr(self._fh, name)

    def spy(file, mode="r", *a, **k):
        fh = real(file, mode, *a, **k)
        return Counting(fh) if str(file) == str(f) else fh

    monkeypatch.setattr("builtins.open", spy)
    v = LogView()
    v.show_log(f)
    assert v.toPlainText().splitlines()[-1] == "last line"
    assert read and sum(read) <= 1024 * 1024


def test_closing_window_does_not_touch_runner(qapp, series4):
    import psutil

    from bearmind.gui.app import MainWindow

    p = spawn(series4, S3, "mc", "running")
    try:
        w = MainWindow(series4.root)
        w.close()
        w.deleteLater()
        assert psutil.Process(p.pid).is_running() and p.poll() is None
        assert monitor.runner_state(series4).alive
    finally:
        stop(p)


def test_refresh_error_reported_once(qapp, series4, monkeypatch):
    reports, status = [], []
    t = MonitorTab(series4.root, report=lambda *e: reports.append(e))
    t.status_message.connect(status.append)
    real = monitor.series_view
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RuntimeError("disk gone")
        return real(*a, **k)

    monkeypatch.setattr(monitor, "series_view", flaky)
    t.refresh()
    assert len(reports) == 1 and not status
    t.refresh()
    assert len(reports) == 1 and len(status) == 1 and "disk gone" in status[0]
    t.refresh()  # succeeds: the next failure is reported again

    def boom(*a, **k):
        raise RuntimeError("again")

    monkeypatch.setattr(monitor, "series_view", boom)
    t.refresh()
    assert len(reports) == 2
