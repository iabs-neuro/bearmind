"""Thresholds tab: session list, MC video, gSig strip, corr/PNR maps with the red mask, sliders, counters, writes."""
import csv
import re
import statistics
import time

import numpy as np
import pytest
import yaml
from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import QAbstractButton, QLabel

from bearmind import actions, config, registry
from bearmind import threshold_model as tm
from bearmind.registry import State, update_record
from tests.bearmind.test_cli import S1, S2, S3, _set, make_current, series  # noqa: F401  (the series fixture)

CYRILLIC = re.compile("[А-Яа-яЁё]")
SUG = {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0, "method": "robust_bg"}
GSIGS = (3, 4, 5)
SHAPE = (40, 60)  # not square: the aspect ratio of the shown maps is checkable
# blob centres and their PNR peaks: at min_corr 0.85 every blob passes min_pnr 8, only the last two pass 20
BLOBS = ((10, 12, 10.0), (10, 45, 15.0), (29, 15, 22.0), (29, 45, 30.0))
CNMF_ASK = ("CNMF-E of {} is already done. With the new thresholds it will be recomputed, and manual inspection "
            "will be needed again. Write?")
CNMF_RUNNING_ASK = ("CNMF-E of {} is running with the old thresholds. It will run again with the new ones, and manual "
                    "inspection will be needed again. Write?")


def _gauss(shape, cy, cx, s):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2.0 * s ** 2))


def save_maps(path, gsigs=GSIGS, mirror=True, shape=SHAPE):
    """Maps as the thresholds stage writes them; the blob width grows with gSig."""
    rng = np.random.default_rng(0)
    arrays = {"gsig": np.array(gsigs), "mean": (60 + 20 * rng.random(shape)).astype(np.float32),
              "max": (120 + 20 * rng.random(shape)).astype(np.float32)}
    for g in gsigs:
        cn, pnr = 0.02 * rng.random(shape), rng.random(shape)
        for cy, cx, peak in BLOBS:
            b = _gauss(shape, cy * shape[0] / SHAPE[0], cx * shape[1] / SHAPE[1], g)
            cn, pnr = cn + 0.97 * b, pnr + peak * b
        arrays[f"cn_{g}"], arrays[f"pnr_{g}"] = cn.astype(np.float32), pnr.astype(np.float32)
        if mirror:
            m = _gauss(shape, shape[0] / 2, shape[1] / 2, g)
            arrays[f"cn_mirror_{g}"] = (0.02 * rng.random(shape) + 0.97 * m).astype(np.float32)
            arrays[f"pnr_mirror_{g}"] = (rng.random(shape) + 12 * m).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def save_video(make_avi, path, shape=SHAPE, n=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.stack([np.full(shape, 10 * i, np.uint8) for i in range(n)])
    return make_avi(path, frames)


def with_maps(paths, sid, make_avi, mirror=True, suggested=SUG, shape=SHAPE):
    """Maps and MC video of the session, with crop, MC and thresholds up to date by the runner's rule."""
    save_maps(paths.threshold_maps(sid), mirror=mirror, shape=shape)
    save_video(make_avi, paths.mc_video(sid), shape=shape)
    make_current(paths, sid, "crop", "mc", "thresholds")
    _set(paths, sid, "thresholds", State.DONE, suggested=suggested)
    _set(paths, sid, "cnmf", State.PENDING)


@pytest.fixture
def series_maps(series, make_avi):  # noqa: F811  (the series fixture imported above)
    """Two mice: S1 (maps with mirrors) and S2 (maps without mirrors) wait for thresholds with the robust_bg
    suggestion; S3 of the other mouse has no maps yet."""
    p = series.paths
    with_maps(p, S1, make_avi, mirror=True)
    with_maps(p, S2, make_avi, mirror=False)
    _set(p, S3, "thresholds", State.PENDING)
    series.make_avi = make_avi
    return series


class Dialogs:
    def __init__(self, answer=True, during=None):
        self.answer = answer
        self.during = during  # runs while the question is open: the modal loop lets the timer refresh meanwhile
        self.asked, self.messages = [], []

    def confirm(self, text, yes="Write", no="Cancel"):
        self.asked.append((text, yes, no))
        if self.during is not None:
            self.during()
        return self.answer

    def message(self, title, text):
        self.messages.append((title, text))


def _reraise(exc_type, exc, tb):
    raise exc


@pytest.fixture
def make_tab(qapp):
    from bearmind.gui.thresholds_tab import ThresholdsTab

    made = []

    def _make(paths, dlg=None, report=_reraise):
        dlg = dlg or Dialogs()
        tab = ThresholdsTab(paths.root, confirm=dlg.confirm, message=dlg.message, report=report)
        tab.dlg = dlg
        tab.said = []
        tab.status_message.connect(tab.said.append)
        made.append(tab)
        return tab

    yield _make
    for tab in made:  # the video worker is joined before the next test
        tab.shutdown()
        tab.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)  # no event loop runs in the tests to do it
    qapp.processEvents()


def _wait(qapp, cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        qapp.processEvents()
        assert time.monotonic() - t0 < timeout, "timeout"
        time.sleep(0.005)


def tree_rows(tab):
    """sid -> state text, and the mouse rows in order."""
    rows, mice = {}, []
    tree = tab.session_list
    for i in range(tree.topLevelItemCount()):
        m = tree.topLevelItem(i)
        mice.append(m.text(0))
        for j in range(m.childCount()):
            c = m.child(j)
            rows[c.data(0, Qt.ItemDataRole.UserRole)] = c.text(1)
    return rows, mice


def red_pixels(img):
    """Pixels of the red mask in the full-resolution overlay (the maps themselves are grey)."""
    a = np.array([[img.pixelColor(x, y).getRgb()[:3] for x in range(img.width())] for y in range(img.height())])
    return int(((a[..., 0] > 200) & (a[..., 1] < 60) & (a[..., 2] < 60)).sum())


def _cfg(paths, sid):
    return (yaml.safe_load(paths.session_config(sid).read_text(encoding="utf-8")) or {}).get("cnmf")


def _choices(paths):
    if not paths.threshold_choices_csv.exists():
        return []
    with open(paths.threshold_choices_csv, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _blobs(text):
    return int(re.match(r"blobs (\d+)", text).group(1))


# --- part 1: the session list ---

def test_session_list_states(qapp, series_maps, make_tab):
    p = series_maps.paths
    tab = make_tab(p)
    rows, mice = tree_rows(tab)
    assert mice == ["NOF_H01", "NOF_H02"]
    assert rows == {S1: "waiting", S2: "waiting", S3: "no maps: not computed yet"}
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    tab.refresh()
    assert tree_rows(tab)[0][S2] == "chosen"
    _set(p, S2, "cnmf", State.DONE)  # done with other thresholds than these: the batch runs it again
    tab.refresh()
    assert tree_rows(tab)[0][S2] == "chosen"
    make_current(p, S2, "cnmf")
    tab.refresh()
    assert tree_rows(tab)[0][S2] == "CNMF-E done"
    update_record(p, S1, lambda r: setattr(r, "excluded", True))
    tab.refresh()
    assert tree_rows(tab)[0] == {S1: "excluded", S2: "CNMF-E done", S3: "no maps: not computed yet"}


def test_opens_on_first_waiting(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    tab = make_tab(p)
    assert tab.current()[0] == S2
    assert tab.session_list.currentItem().data(0, Qt.ItemDataRole.UserRole) == S2
    # part 2: the MC video of the session opens when the tab is shown
    tab.show()
    _wait(qapp, lambda: tab.video.slider.isEnabled())
    assert tab.video.slider.maximum() == 9
    assert S2 in tab.video_title.text()


def test_video_open_only_while_visible(qapp, series_maps, make_tab):
    """A hidden tab holds no decoder thread and no file: Monitor-only use never opens an MC video, and the batch
    gets the file back when the person leaves the tab."""
    from bearmind.gui import video_view

    before = set(video_view._LIVE)
    tab = make_tab(series_maps.paths)
    qapp.processEvents()
    assert set(video_view._LIVE) == before and not tab.video.slider.isEnabled()
    tab.show()
    _wait(qapp, lambda: tab.video.slider.isEnabled())
    tab.video.seek(5)
    _wait(qapp, lambda: tab.video.shown_image() is not None and tab.video.current_index() == 5)
    tab.hide()
    assert set(video_view._LIVE) == before and not tab.video.slider.isEnabled()
    tab.show_session(S2)  # a session change while hidden opens nothing either
    assert set(video_view._LIVE) == before
    tab.show_session(S1)
    tab.show()
    _wait(qapp, lambda: tab.video.slider.isEnabled())
    assert tab.video.current_index() == 0  # another session was shown meanwhile: from the start
    tab.video.seek(7)
    tab.hide()
    tab.show()
    _wait(qapp, lambda: tab.video.shown_image() is not None and tab.video.current_index() == 7)  # back where it was


# --- parts 3-6: strip, maps, sliders, counters, start values ---

def test_initial_sliders_from_suggestion(qapp, series_maps, make_tab):
    tab = make_tab(series_maps.paths)
    assert tab.current() == (S1, 4, 0.85, 8.0)
    assert tab.min_corr_slider.value() == 85 and tab.min_corr_slider.maximum() == 100
    assert tab.min_pnr_slider.value() == 80
    assert tab.min_corr_spin.value() == 0.85 and tab.min_pnr_spin.value() == 8.0
    assert tab.source_label.text().startswith("Autotune suggestion (robust_bg)")
    assert sorted(tab.gsig_buttons) == list(GSIGS) and tab.gsig_buttons[4].isChecked()
    assert tab.gsig_buttons[4].text() == "gSig 4 · 4"
    assert tab.counters_label.text() == "blobs 4 · single pixels 0 % · blobs on mirrored maps 1"


def test_slider_updates_overlay_and_counters(qapp, series_maps, make_tab):
    tab = make_tab(series_maps.paths)
    before = tab.counters_label.text()
    red_before = red_pixels(tab.overlay_image("corr"))
    assert red_before == red_pixels(tab.overlay_image("pnr")) > 0
    # a drag: the red mask follows the slider, the numbers wait for the release
    tab.min_pnr_slider.setSliderDown(True)
    tab.min_pnr_slider.setValue(200)
    assert tab.min_pnr_spin.value() == 20.0 and tab.current()[3] == 20.0
    assert 0 < red_pixels(tab.overlay_image("corr")) < red_before
    assert tab.counters_label.text() == before
    assert tab.gsig_buttons[4].text() == "gSig 4 · 4"
    tab.min_pnr_slider.setSliderDown(False)  # release
    after = tab.counters_label.text()
    assert after != before and _blobs(after) <= _blobs(before)
    k = tm.counters(tm.load_maps(series_maps.paths.threshold_maps(S1)), 4, 0.85, 20.0, 2)
    assert 0 < k.n_blobs < 4
    assert after == (f"blobs {k.n_blobs} · single pixels {round(k.single_frac * 100)} % · "
                     f"blobs on mirrored maps {k.n_mirror_blobs}")
    assert tab.gsig_buttons[4].text() == f"gSig 4 · {k.n_blobs}"
    # the spin box moves the slider, and a change that is not a drag updates the numbers at once
    tab.min_pnr_spin.setValue(8.0)
    assert tab.min_pnr_slider.value() == 80 and tab.counters_label.text() == before
    # S2 has no mirrored maps: the last part is not shown
    tab.show_session(S2)
    assert tab.counters_label.text() == "blobs 4 · single pixels 0 %"


def test_gsig_click_switches_maps(qapp, series_maps, make_tab):
    tab = make_tab(series_maps.paths)
    red4 = red_pixels(tab.overlay_image("corr"))
    tab.gsig_buttons[5].click()
    assert tab.current() == (S1, 5, 0.85, 8.0)
    assert tab.gsig_buttons[5].isChecked() and not tab.gsig_buttons[4].isChecked()
    assert red_pixels(tab.overlay_image("corr")) > red4  # wider blobs at gSig 5
    assert "gSig 5" in tab.maps_title.text()
    tab.set_gsig(3)
    assert tab.current()[1] == 3 and tab.gsig_buttons[3].isChecked()
    assert red_pixels(tab.overlay_image("corr")) < red4


def test_gsig_listed_without_maps_falls_back(qapp, series_maps, make_tab):
    p = series_maps.paths
    with np.load(p.threshold_maps(S1)) as z:
        arrays = {k: z[k] for k in z.files if not k.endswith("_4")}  # the suggested gSig 4 lost its maps
    np.savez_compressed(p.threshold_maps(S1), **arrays)
    tab = make_tab(p)
    assert sorted(tab.gsig_buttons) == [3, 5]
    assert tab.current() == (S1, 5, 0.85, 8.0) and tab.gsig_buttons[5].isChecked()
    assert tab.source_label.text().endswith(" — gSig 4 has no map; showing 5")  # the swap is said, not hidden
    # Reset asks for the suggestion again: the same swap, and the screen and current() agree
    tab.set_gsig(3)
    tab.set_thresholds(0.7, 12.0)
    tab.reset_button.click()
    assert tab.current() == (S1, 5, 0.85, 8.0)
    assert tab.maps_title.text() == "Correlation and PNR, gSig 5"
    assert [g for g, b in tab.gsig_buttons.items() if b.isChecked()] == [5]
    assert tab.source_label.text().endswith(" — gSig 4 has no map; showing 5")


def test_overlay_is_fast_on_600_px_maps(qapp, series, make_avi, make_tab):  # noqa: F811
    p = series.paths
    with_maps(p, S1, make_avi, shape=(600, 600))
    tab = make_tab(p)
    tab.resize(1100, 700)
    tab.show()
    qapp.processEvents()
    times = []
    tab.min_pnr_slider.setSliderDown(True)
    for v in range(60, 260, 10):
        t = time.perf_counter()
        tab.min_pnr_slider.setValue(v)
        times.append(time.perf_counter() - t)
    tab.min_pnr_slider.setSliderDown(False)
    assert statistics.median(times) < 0.05  # the requirement: the mask follows a drag within 50 ms


def test_whole_image_visible(qapp, series_maps, make_tab):
    tab = make_tab(series_maps.paths)
    for size in ((1100, 700), (700, 450), (1600, 1000)):
        tab.resize(*size)
        tab.show()
        qapp.processEvents()
        for view in (tab.corr_view, tab.pnr_view):
            pm = view.pixmap()
            room = view.contentsRect()
            assert not pm.isNull()
            assert pm.width() <= room.width() and pm.height() <= room.height()
            assert abs(pm.width() / pm.height() - SHAPE[1] / SHAPE[0]) < 0.05
            assert pm.width() <= 4 * SHAPE[1] and pm.height() <= 4 * SHAPE[0]  # never upscaled above 4x


def test_refresh_keeps_the_shown_values(qapp, series_maps, make_tab):
    p = series_maps.paths
    tab = make_tab(p)
    tab.set_gsig(5)
    tab.set_thresholds(0.7, 12.0)
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    tab.refresh()
    assert tab.current() == (S1, 5, 0.7, 12.0)
    assert tree_rows(tab)[0][S2] == "chosen"


def test_new_maps_file_reloads_and_keeps_the_values(qapp, series_maps, make_tab):
    """The batch replaced the shown session's maps (here: without mirrored maps): the tab shows the new maps
    without a session change, and the person's slider values stay."""
    p = series_maps.paths
    tab = make_tab(p)
    tab.set_gsig(5)
    tab.set_thresholds(0.7, 12.0)
    assert "mirrored" in tab.counters_label.text()
    save_maps(p.threshold_maps(S1), mirror=False)
    tab.refresh()
    assert tab.current() == (S1, 5, 0.7, 12.0) and tab.gsig_buttons[5].isChecked()
    assert "mirrored" not in tab.counters_label.text()
    assert tab.source_label.text().startswith("Autotune suggestion (robust_bg)")


# --- part 7: buttons ---

def test_accept_and_next_writes_and_moves(qapp, series_maps, make_tab):
    p = series_maps.paths
    tab = make_tab(p)
    tab.set_gsig(5)
    tab.set_thresholds(0.8, 9.0)
    tab.accept_and_next()
    assert _cfg(p, S1) == {"gSig": 5, "min_corr": 0.8, "min_pnr": 9.0}
    (row,) = _choices(p)
    assert row["sid"] == S1 and row["source"] == "window" and row["who"] == actions.current_user()
    assert (row["gSig"], row["min_corr"], row["min_pnr"]) == ("5", "0.8", "9.0") and row["n_blobs"] == "4"
    assert row["n_mirror_blobs"] == "1"
    assert tab.current()[0] == S2 and tree_rows(tab)[0][S1] == "chosen"
    assert not tab.dlg.asked
    # S2 was the last waiting session: the tab stays on it and says what is still without thresholds
    tab.accept_and_next()
    assert tab.current()[0] == S2 and _cfg(p, S2) == {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0}
    assert "All sessions have thresholds." not in tab.said[-1] and S3 in tab.said[-1]
    update_record(p, S3, lambda r: setattr(r, "excluded", True))
    tab.refresh()
    tab.accept_and_next()
    assert tab.said[-1].endswith("All sessions have thresholds.")


def test_next_waiting_wraps_to_the_top(qapp, series_maps, make_tab):
    p = series_maps.paths
    with_maps(p, S3, series_maps.make_avi)
    tab = make_tab(p)
    tab.show_session(S2)
    tab.accept_and_next()
    assert tab.current()[0] == S3
    tab.accept_and_next()
    assert tab.current()[0] == S1  # wrapped to the top
    tab.next_button.click()
    assert tab.current()[0] == S2
    tab.prev_button.click()
    assert tab.current()[0] == S1 and not tab.prev_button.isEnabled()


def test_reset_to_suggestion(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 3, "min_corr": 0.7, "min_pnr": 6.0}})
    tab = make_tab(p)
    tab.show_session(S1)
    assert tab.current() == (S1, 3, 0.7, 6.0) and tab.source_label.text() == "Your earlier choice (session config)"
    tab.reset_button.click()
    assert tab.current() == (S1, 4, 0.85, 8.0)
    assert tab.source_label.text().startswith("Autotune suggestion (robust_bg)")
    with_maps(p, S2, series_maps.make_avi, suggested=None)
    tab.refresh()
    tab.show_session(S2)
    assert not tab.reset_button.isEnabled()
    assert tab.source_label.text() == "No suggestion — start values from the config"


def test_accept_for_mouse(qapp, series_maps, make_tab):
    p = series_maps.paths
    actions.set_thresholds(p, S2, 3, 0.7, 6.0)
    tab = make_tab(p, Dialogs(answer=False))
    assert tab.current()[0] == S1
    tab.set_thresholds(0.9, 10.0)
    tab.accept_for_mouse()
    ((text, yes, no),) = tab.dlg.asked
    assert text.splitlines() == ["Thresholds gSig 4, min_corr 0.90, min_pnr 10.0 for the sessions of NOF_H01.",
                                 "Will get these thresholds: NOF_H01_1D",
                                 "Keep their own: NOF_H01_2D"]
    assert (yes, no) == ("Write", "Cancel")
    assert not p.session_config(S1).exists() and len(_choices(p)) == 1  # Cancel writes nothing
    tab.dlg.answer = True
    tab.accept_for_mouse()
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.9, "min_pnr": 10.0}
    assert _cfg(p, S2) == {"gSig": 3, "min_corr": 0.7, "min_pnr": 6.0}  # its own choice stays
    assert [r["source"] for r in _choices(p)] == ["cli", "window_mouse"]
    assert not p.session_config(S3).exists()
    assert not tab.dlg.messages


def test_accept_for_mouse_names_excluded_and_cnmf_done(qapp, series_maps, make_tab):
    from tests.bearmind.test_runner import _record

    p = series_maps.paths
    registry.save_record(p, _record("NOF_H01_3D", decided=False, excluded=True))
    _set(p, S2, "cnmf", State.DONE)  # done, but its config lost the thresholds: the write recomputes it
    tab = make_tab(p, Dialogs(answer=False))
    tab.show_session(S1)
    tab.accept_for_mouse()
    ((text, _, _),) = tab.dlg.asked
    assert text.splitlines() == [
        "Thresholds gSig 4, min_corr 0.85, min_pnr 8.0 for the sessions of NOF_H01.",
        "Will get these thresholds: NOF_H01_1D, NOF_H01_2D",
        "Keep their own: none",
        "Excluded, left alone: NOF_H01_3D",
        "CNMF-E of NOF_H01_2D is already done: it will be recomputed, and manual inspection will be needed again."]
    assert not _choices(p)


def test_accept_for_mouse_says_when_the_sessions_changed(qapp, series_maps, make_tab):
    """The registry changed while the question was open: the message names what was really written."""
    p = series_maps.paths
    dlg = Dialogs(during=lambda: actions.set_thresholds(p, S2, 3, 0.7, 6.0))
    tab = make_tab(p, dlg)
    tab.show_session(S1)
    tab.accept_for_mouse()
    assert "Will get these thresholds: NOF_H01_1D, NOF_H01_2D" in dlg.asked[0][0].splitlines()
    assert dlg.messages == [("BEARMiND", "The sessions of NOF_H01 changed after the question. Written: NOF_H01_1D.")]
    assert _cfg(p, S2) == {"gSig": 3, "min_corr": 0.7, "min_pnr": 6.0}


def test_accept_mouse_tooltip(qapp, series_maps, make_tab):
    tab = make_tab(series_maps.paths)
    assert tab.accept_mouse_button.toolTip() == ("Write these thresholds to every session of this mouse that has "
                                                 "no thresholds yet and is not excluded")


# --- part 8 and the states that refuse a write ---

def test_cnmf_done_asks_before_write(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    _set(p, S1, "cnmf", State.DONE)
    tab = make_tab(p, Dialogs(answer=False))
    tab.show_session(S1)
    assert tab.accept_button.isEnabled() and tab.accept_mouse_button.isEnabled()
    tab.set_thresholds(0.9, 9.0)
    tab.accept_and_next()
    assert tab.dlg.asked == [(CNMF_ASK.format(S1), "Write", "Cancel")]
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0} and not _choices(p)
    tab.dlg.answer = True
    tab.accept_and_next()
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.9, "min_pnr": 9.0}


def test_cnmf_done_same_values_need_no_question(qapp, series_maps, make_tab):
    """Writing the thresholds CNMF-E already used changes nothing it reads, so the question would be false."""
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    _set(p, S1, "cnmf", State.DONE)
    tab = make_tab(p, Dialogs(answer=False))
    tab.show_session(S1)
    assert tab.current() == (S1, 4, 0.8, 8.0)
    tab.accept_and_next()
    assert not tab.dlg.asked and _choices(p)[-1]["sid"] == S1


def test_cnmf_running_asks_before_write(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    _set(p, S1, "cnmf", State.RUNNING)
    tab = make_tab(p, Dialogs(answer=False))
    tab.show_session(S1)
    assert tree_rows(tab)[0][S1] == "chosen" and tab.accept_button.isEnabled()
    tab.set_thresholds(0.9, 9.0)
    tab.accept_and_next()
    assert tab.dlg.asked == [(CNMF_RUNNING_ASK.format(S1), "Write", "Cancel")]
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0} and not _choices(p)


def test_cnmf_running_with_the_values_it_uses_needs_no_question(qapp, series_maps, make_tab):
    """The run's own parameters decide: the config changed after the run started does not."""
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 5, "min_corr": 0.7, "min_pnr": 6.0}})
    _set(p, S1, "cnmf", State.RUNNING)
    update_record(p, S1, lambda r: r.stages["cnmf"].params.update(gSig=[4, 4], min_corr=0.85, min_pnr=8.0))
    tab = make_tab(p, Dialogs(answer=False))
    tab.show_session(S1)
    tab.set_gsig(4)
    tab.set_thresholds(0.85, 8.0)
    tab.accept_and_next()
    assert not tab.dlg.asked and _cfg(p, S1) == {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0}


def test_question_open_while_the_maps_go_away(qapp, series_maps, make_tab):
    """The modal question runs a nested event loop, so the timer may clear the maps meanwhile: the write uses the
    values and the counters the person saw when asked."""
    p = series_maps.paths
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    _set(p, S1, "cnmf", State.DONE)
    tab = make_tab(p)

    def meanwhile():
        _set(p, S1, "thresholds", State.RUNNING)
        tab.refresh()

    tab.dlg.during = meanwhile
    tab.show_session(S1)
    tab.set_thresholds(0.9, 9.0)
    seen = tab.counters_label.text()
    tab.accept_and_next()
    assert tab.dlg.asked and _cfg(p, S1) == {"gSig": 4, "min_corr": 0.9, "min_pnr": 9.0}
    row = _choices(p)[-1]
    assert seen.startswith(f"blobs {row['n_blobs']} ·") and row["n_mirror_blobs"] != ""


@pytest.mark.parametrize("state, error, text", [
    (State.RUNNING, None, "The maps are being recomputed by the batch."),
    (State.PENDING, None, "The maps are out of date: the batch will recompute them."),
    (State.FAILED, "Traceback\n  File x\nMemoryError: cannot allocate\n",
     "The maps are out of date: the thresholds stage failed: MemoryError: cannot allocate. Retry it in Monitor."),
    (State.SKIPPED, "skipped: mc failed",
     "The maps are out of date (skipped: mc failed). Retry the failed stage in Monitor."),
])
def test_out_of_date_maps_reason(qapp, series_maps, make_tab, state, error, text):
    p = series_maps.paths
    _set(p, S1, "thresholds", state, error=error)
    tab = make_tab(p)
    tab.show_session(S1)
    assert tab.corr_view.text() == text and tab.pnr_view.text() == text
    assert not tab.accept_button.isEnabled() and tab.accept_button.toolTip() == text


def test_excluded_session_reason_says_the_batch_ignores_it(qapp, series_maps, make_tab):
    p = series_maps.paths
    _set(p, S1, "thresholds", State.PENDING)  # an old maps file, the stage reset
    for sid in (S1, S3):  # S3 has no maps file at all
        update_record(p, sid, lambda r: setattr(r, "excluded", True))
    tab = make_tab(p)
    tab.show_session(S1)
    assert tab.corr_view.text() == ("The maps are out of date, and the session is excluded: the batch does not "
                                    "process it.")
    tab.show_session(S3)
    assert tab.corr_view.text() == "No maps. The session is excluded: the batch does not process it."
    assert "recompute" not in tab.corr_view.text() and not tab.accept_button.isEnabled()


def test_refresh_guard_reports_once_then_says_it(qapp, series_maps, make_tab, monkeypatch):
    reported = []
    tab = make_tab(series_maps.paths, report=lambda t, e, tb: reported.append(e))

    def boom(*args, **kwargs):
        raise RuntimeError("registry busy\nmore")

    monkeypatch.setattr(tm, "threshold_items", boom)
    tab.refresh()
    tab.refresh()
    assert [str(e) for e in reported] == ["registry busy\nmore"]  # one dialog, not one per tick
    assert tab.said == ["Refresh failed: RuntimeError: registry busy"]


def test_video_workers_joined_at_exit_without_exec(qapp, make_avi, tmp_path):
    """A process that never runs exec() gets no aboutToQuit: the atexit guard joins the workers. Twice is harmless,
    and a worker whose C++ object is already gone is skipped."""
    import shiboken6

    from bearmind.gui import video_view
    from bearmind.gui.video_view import VideoView

    path = save_video(make_avi, tmp_path / "mc.avi")
    view = VideoView()
    view.set_video(path, 20)
    _wait(qapp, lambda: view.slider.isEnabled())
    assert video_view._LIVE
    video_view._join_at_exit()
    assert not video_view._LIVE
    video_view._join_at_exit()
    gone = video_view._Worker(path, 0)
    video_view._LIVE.add(gone)
    shiboken6.delete(gone)
    video_view._join_at_exit()
    assert not video_view._LIVE


def test_session_without_maps(qapp, series_maps, make_tab):
    p = series_maps.paths
    tab = make_tab(p)
    tab.show_session(S3)
    assert tab.current()[0] == S3
    assert "not computed yet" in tab.corr_view.text() and "not computed yet" in tab.pnr_view.text()
    assert not tab.accept_button.isEnabled() and not tab.accept_mouse_button.isEnabled()
    assert not tab.min_corr_slider.isEnabled() and not tab.gsig_buttons
    tab.accept_and_next()
    tab.accept_for_mouse()
    assert not p.session_config(S3).exists() and not tab.dlg.asked
    # an excluded session shows its maps, but takes no thresholds
    update_record(p, S1, lambda r: setattr(r, "excluded", True))
    tab.refresh()
    tab.show_session(S1)
    assert tab.corr_view.pixmap() is not None and not tab.corr_view.pixmap().isNull()
    assert not tab.accept_button.isEnabled() and not tab.accept_mouse_button.isEnabled()


def test_maps_recomputed_or_unreadable_show_the_reason(qapp, series_maps, make_tab):
    p = series_maps.paths
    _set(p, S1, "thresholds", State.RUNNING)
    p.threshold_maps(S2).write_bytes(b"not a maps file")
    tab = make_tab(p)
    tab.show_session(S1)
    assert "being recomputed" in tab.corr_view.text() and not tab.accept_button.isEnabled()
    tab.show_session(S2)
    assert tab.corr_view.text().startswith("Maps cannot be read") and not tab.accept_button.isEnabled()
    # the batch finishes the maps of the shown session: the tab shows them without a session change
    tab.show_session(S1)
    _set(p, S1, "thresholds", State.DONE)
    tab.refresh()
    assert not tab.corr_view.pixmap().isNull() and tab.accept_button.isEnabled()


def test_session_gone_before_write(qapp, series_maps, make_tab):
    p = series_maps.paths
    tab = make_tab(p)
    p.registry_file(S1).unlink()
    tab.accept_and_next()
    assert tab.dlg.messages and S1 in tab.dlg.messages[-1][1]
    assert not p.session_config(S1).exists()
    assert S1 not in tree_rows(tab)[0]


def test_timer_refreshes_only_when_visible(qapp, series_maps, make_tab, monkeypatch):
    tab = make_tab(series_maps.paths)
    assert tab.timer.isActive()
    assert tab.timer.interval() == config.load_defaults()["monitor"]["refresh_ms"]
    calls = []
    monkeypatch.setattr(tab, "refresh", lambda: calls.append(1))
    tab._tick()
    assert calls == []
    tab.show()
    qapp.processEvents()
    calls.clear()
    tab._tick()
    assert calls == [1]


def test_thresholds_tab_texts_are_english(qapp, series_maps, make_tab):
    p = series_maps.paths
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    _set(p, S2, "cnmf", State.DONE)
    tab = make_tab(p, Dialogs(answer=False))
    texts = [w.text() for w in tab.findChildren(QLabel)] + [b.text() for b in tab.findChildren(QAbstractButton)]
    texts += [w.toolTip() for w in tab.findChildren(QAbstractButton)]
    tree = tab.session_list
    texts += [tree.headerItem().text(c) for c in range(tree.columnCount())]
    for i in range(tree.topLevelItemCount()):
        m = tree.topLevelItem(i)
        texts += [m.text(0)] + [m.child(j).text(c) for j in range(m.childCount()) for c in range(2)]
        texts += [m.child(j).toolTip(0) for j in range(m.childCount())]
    tab.accept_for_mouse()
    tab.show_session(S2)
    tab.set_thresholds(0.9, 9.0)
    tab.accept_and_next()
    tab.show_session(S3)
    texts += [tab.corr_view.text(), tab.pnr_view.text()]
    texts += [t for t, _, _ in tab.dlg.asked] + [t for _, t in tab.dlg.messages] + tab.said
    assert len(tab.dlg.asked) == 2
    for t in texts:
        assert not CYRILLIC.search(t), t


# --- the main window ---

def test_window_thresholds_tab_replaces_placeholder(qapp, tmp_series):
    from bearmind.gui.app import MainWindow
    from bearmind.gui.thresholds_tab import ThresholdsTab

    w = MainWindow(tmp_series)
    tab = w.tabs.widget(2)
    assert isinstance(tab, ThresholdsTab) and w.tabs.tabText(2) == "Thresholds"
    assert "Scan" in tab.corr_view.text()  # an empty series says where the sessions come from
    w.monitor.thresholds_button.setEnabled(True)  # "Thresholds: N sessions ->", enabled once a session waits
    w.monitor.thresholds_button.click()
    assert w.tabs.currentIndex() == 2
    tab.status_message.emit("hello")
    assert w.statusBar().currentMessage() == "hello"
    w.open_series(tmp_series)  # a second open replaces the tab and stops the old one
    assert w.tabs.widget(2) is not tab and isinstance(w.tabs.widget(2), ThresholdsTab)
    assert not tab.timer.isActive()
    w.thresholds.shutdown()
    w.close()


def test_window_refreshes_monitor_after_a_write(qapp, series_maps):
    from bearmind.gui.app import MainWindow

    w = MainWindow(series_maps.paths.root, message=lambda title, text: None)
    assert w.monitor.thresholds_button.text() == "Thresholds: 2 sessions →"
    w.monitor.timer.stop()  # the count below comes from the write, not from the timer
    w.thresholds.accept_and_next()
    assert w.monitor.thresholds_button.text() == "Thresholds: 1 session →"
    # tests never emit aboutToQuit, which joins the video workers in the app: the tab is released here instead
    w.thresholds.shutdown()
    w.close()


def test_no_registry_sessions_is_an_empty_list(qapp, tmp_path, make_tab):
    from bearmind.paths import SeriesPaths

    p = SeriesPaths(tmp_path / "empty")
    p.root.mkdir()
    tab = make_tab(p)
    assert tab.session_list.topLevelItemCount() == 0 and tab.current()[0] == ""
    assert not tab.accept_button.isEnabled()
    assert registry.read_all(p) == ([], [])


# --- phase 4 final fix wave, A: one rule for the tab, Monitor and actions ---

def test_probe_cases_agree_in_tab_monitor_and_actions(qapp, series_maps, make_tab):
    """A6 (probe_states): (a) maps without a suggestion, CNMF-E pending; (b) a suggestion, CNMF-E FAILED and no
    thresholds; (c) maps DONE but out of date. The tab, Monitor and actions give one answer: (a) and (b) wait,
    (c) does not, and its maps are refused with the reason."""
    from bearmind import monitor

    p = series_maps.paths
    _set(p, S1, "thresholds", State.DONE, suggested=None)  # (a)
    _set(p, S2, "cnmf", State.FAILED, error="Traceback\nMemoryError: x")  # (b)
    with_maps(p, S3, series_maps.make_avi)  # (c): maps of S3, then a crop input of its mouse changes
    config.update_yaml(p.mouse_config("NOF_H02"), {"crop": {"fov_threshold_frac": 0.3}})
    tab = make_tab(p)
    assert tree_rows(tab)[0] == {S1: "waiting", S2: "waiting", S3: "maps out of date: the batch redoes crop first"}
    v = monitor.series_view(p)
    assert v.waiting["thresholds"] == 2
    cells = {r.sid: r.cells["thresholds"] for r in v.rows}
    assert cells[S1].text == cells[S2].text == "waits" and cells[S3].text == "out of date"
    assert actions.waiting_threshold_sids(p) == [S1, S2]
    assert [i.sid for i in tm.threshold_items(p) if i.waiting] == [S1, S2]
    tab.show_session(S3)
    note = "The maps are out of date: the batch redoes crop first."
    assert tab.corr_view.text() == note and tab.pnr_view.text() == note
    assert not tab.accept_button.isEnabled() and tab.accept_button.toolTip() == note
    assert not tab.accept_mouse_button.isEnabled()
    tab.accept_and_next()  # refused: nothing is written
    assert not p.session_config(S3).exists() and not _choices(p)
    config.update_yaml(p.session_config(S3), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    tab.refresh()
    assert tree_rows(tab)[0][S3] == "chosen · maps will be recomputed"


def test_window_count_drops_at_once_after_accept_while_the_registry_still_waits(qapp, series_maps):
    """A4: Monitor's "Thresholds: N" and its "waiting for you" follow the config at once after Accept; the
    registry says WAITING until the batch checks again (up to poll_s later)."""
    from bearmind.gui.app import MainWindow

    p = series_maps.paths
    for sid in (S1, S2):
        _set(p, sid, "cnmf", State.WAITING, error="waiting for thresholds")
    w = MainWindow(p.root, message=lambda title, text: None)
    assert w.monitor.thresholds_button.text() == "Thresholds: 2 sessions →"
    w.monitor.timer.stop()  # the count below comes from the write, not from the timer
    w.thresholds.accept_and_next()
    assert registry.load_record(p, S1).stages["cnmf"].state is State.WAITING  # the batch has not looked yet
    assert w.monitor.thresholds_button.text() == "Thresholds: 1 session →"
    row = next(i for i in range(w.monitor.model.rowCount()) if w.monitor.model.item(i, 0).text() == S1)
    assert w.monitor.model.item(row, 3).text() == "gSig 4"  # the Thresholds cell no longer says it waits
    w.thresholds.shutdown()
    w.close()


# --- phase 4 final fix wave, B: writes that never lie ---

NO_GATE_DONE = "CNMF-E of {} is already done. With the new thresholds it will be recomputed. Write?"
NO_GATE_RUNNING = "CNMF-E of {} is running with the old thresholds. It will run again with the new ones. Write?"
PENDING_TEXT = ("Thresholds written. The choice log is open in another program; the row was kept in "
                "threshold_choices.pending.csv.")


def test_broken_series_config_refuses_the_mouse_write(qapp, series_maps, make_tab):
    """B1 (review probe): with series.yaml broken by a hand edit, the shown session cannot be scheduled: Accept and
    Accept for all sessions of this mouse are refused with that reason, and S2 keeps its own thresholds."""
    p = series_maps.paths
    actions.set_thresholds(p, S2, 3, 0.7, 6.0)
    _set(p, S2, "cnmf", State.DONE)
    p.series_config.write_text(p.series_config.read_text(encoding="utf-8") + "\ngate: {enabled: false\n",
                               encoding="utf-8")
    tab = make_tab(p, Dialogs(answer=True))
    tab.show_session(S1)
    rows = tree_rows(tab)[0]
    assert rows[S1].startswith("cannot be scheduled: config: ConfigError: ") and "series.yaml" in rows[S1]
    why = tab.accept_mouse_button.toolTip()
    assert why.startswith("The session cannot be scheduled: config: ") and tab.corr_view.text() == why
    assert not tab.accept_button.isEnabled() and not tab.accept_mouse_button.isEnabled()
    tab.accept_mouse_button.click()
    tab.accept_for_mouse()
    assert not tab.dlg.asked and _cfg(p, S2) == {"gSig": 3, "min_corr": 0.7, "min_pnr": 6.0}


def test_mouse_question_lists_sessions_whose_config_cannot_be_read(qapp, series_maps, make_tab):
    """B1: a session of the mouse whose config cannot be read is named in the question and left alone."""
    p = series_maps.paths
    p.session_config(S2).parent.mkdir(parents=True, exist_ok=True)
    p.session_config(S2).write_text("cnmf: {gSig: 3, min_corr: 0.7\n", encoding="utf-8")
    before = p.session_config(S2).read_text(encoding="utf-8")
    tab = make_tab(p, Dialogs(answer=True))
    tab.show_session(S1)
    tab.accept_for_mouse()
    ((text, _, _),) = tab.dlg.asked
    assert text.splitlines() == [
        "Thresholds gSig 4, min_corr 0.85, min_pnr 8.0 for the sessions of NOF_H01.",
        "Will get these thresholds: NOF_H01_1D",
        "Keep their own: none",
        "Config cannot be read, left alone: NOF_H01_2D (_bearmind/config/sessions/NOF_H01_2D.yaml: while parsing "
        "a flow mapping (line 1))"]
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0}
    assert p.session_config(S2).read_text(encoding="utf-8") == before and not tab.dlg.messages


def test_unwritable_choice_log_says_so_and_moves_on(qapp, series_maps, make_tab):
    """B2 (review probe test_probe_choice_log_unwritable): the thresholds are written, the window says where the
    row went, emits decided and moves to the next waiting session, as after any write."""
    p = series_maps.paths
    p.threshold_choices_csv.mkdir(parents=True)
    reported, decided = [], []
    tab = make_tab(p, report=lambda t, e, tb: reported.append(e))
    tab.decided.connect(lambda: decided.append(1))
    tab.accept_button.click()  # through the guarded slot, as a click
    assert not reported and decided == [1]
    assert tab.dlg.messages == [("BEARMiND", PENDING_TEXT)]
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0}
    assert tab.current()[0] == S2 and tree_rows(tab)[0][S1] == "chosen"
    assert tab.said[-1].startswith(f"Thresholds of {S1} written")


def test_mouse_write_lists_the_sessions_it_could_not_write(qapp, series_maps, make_tab, monkeypatch):
    """B2: apply_to_mouse goes on after a failed session; the window lists the failures, and does not claim the
    registry changed."""
    from pathlib import Path

    p = series_maps.paths
    real = config.update_yaml

    def flaky(path, patch):
        if Path(path) == p.session_config(S2):
            raise PermissionError(13, "Permission denied", str(path))
        return real(path, patch)

    tab = make_tab(p, Dialogs(answer=True))
    tab.show_session(S1)
    monkeypatch.setattr(config, "update_yaml", flaky)
    tab.accept_for_mouse()
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.85, "min_pnr": 8.0}
    ((title, text),) = tab.dlg.messages
    assert text.startswith(f"Writing failed for {S2}: PermissionError: ") and "changed after" not in text


def test_gate_disabled_questions_do_not_promise_inspection(qapp, series_maps, make_tab):
    """B3: with the gate disabled in the session's resolved config, a rerun of CNMF-E needs no manual inspection, so
    the questions do not say it will."""
    p = series_maps.paths
    config.update_yaml(p.series_config, {"gate": {"enabled": False}})
    for sid in (S1, S2):
        config.update_yaml(p.session_config(sid), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    _set(p, S1, "cnmf", State.DONE)
    _set(p, S2, "cnmf", State.RUNNING)
    tab = make_tab(p, Dialogs(answer=False))
    for sid, text in ((S1, NO_GATE_DONE), (S2, NO_GATE_RUNNING)):
        tab.show_session(sid)
        tab.set_thresholds(0.9, 9.0)
        tab.accept_and_next()
        assert tab.dlg.asked[-1][0] == text.format(sid)
    config.update_yaml(p.session_config(S2), {"gate": {"enabled": True}})  # S2 inspects again, S1 does not
    config.update_yaml(p.session_config(S1), {"cnmf": {"gSig": None}})
    config.update_yaml(p.session_config(S2), {"cnmf": {"gSig": None}})
    _set(p, S2, "cnmf", State.DONE)
    tab.refresh()
    tab.show_session(S1)
    tab.accept_for_mouse()
    lines = tab.dlg.asked[-1][0].splitlines()
    assert f"CNMF-E of {S1} is already done: it will be recomputed." in lines
    assert (f"CNMF-E of {S2} is already done: it will be recomputed, and manual inspection will be needed again."
            in lines)


def test_spin_boxes_take_a_dot_whatever_the_system_locale(qapp, series_maps, make_tab):
    """B4 (check finding 1): on a Russian Windows the spin boxes took only a comma and wrote 12.0 for "12.5"."""
    from PySide6.QtCore import QLocale
    from PySide6.QtTest import QTest

    old = QLocale()
    QLocale.setDefault(QLocale(QLocale.Language.Russian, QLocale.Country.Russia))
    try:
        tab = make_tab(series_maps.paths)
        for spin, typed in ((tab.min_pnr_spin, "12.5"), (tab.min_corr_spin, "0.62")):
            spin.lineEdit().selectAll()
            QTest.keyClicks(spin, typed)
            QTest.keyClick(spin, Qt.Key.Key_Return)
        assert tab.min_pnr_spin.value() == 12.5 and tab.min_corr_spin.value() == 0.62
        assert tab.current()[2:] == (0.62, 12.5) and tab.min_pnr_spin.text() == "12.5"
        assert tab.min_corr_spin.locale() == QLocale.c() == tab.min_pnr_spin.locale()
    finally:
        QLocale.setDefault(old)


def test_untouched_suggestion_is_written_exactly(qapp, series_maps, make_tab):
    """B5: a start value between slider steps stays exact until the person moves that control; Accept then writes
    what the source label says and what CLI accept-suggested writes."""
    p = series_maps.paths
    sug = {"gSig": 4, "min_corr": 0.853, "min_pnr": 8.27, "method": "robust_bg"}
    _set(p, S1, "thresholds", State.DONE, suggested=sug)
    tab = make_tab(p)
    assert tab.current() == (S1, 4, 0.853, 8.27)
    assert tab.min_corr_slider.value() == 85 and tab.min_pnr_slider.value() == 83
    assert (tab.min_corr_spin.text(), tab.min_pnr_spin.text()) == ("0.853", "8.27")
    assert tab.source_label.text() == "Autotune suggestion (robust_bg): gSig 4, min_corr 0.853, min_pnr 8.27"
    tab.set_gsig(5)  # another control: the thresholds stay exact
    tab.set_gsig(4)
    assert tab.current()[2:] == (0.853, 8.27)
    tab.accept_and_next()
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.853, "min_pnr": 8.27}
    assert tab.said[-1].startswith(f"Thresholds of {S1} written: gSig 4, min_corr 0.853, min_pnr 8.27.")
    assert (_choices(p)[-1]["min_corr"], _choices(p)[-1]["min_pnr"]) == ("0.853", "8.27")
    actions.accept_suggested(p, [S1])  # the CLI's way writes the same
    assert _cfg(p, S1) == {"gSig": 4, "min_corr": 0.853, "min_pnr": 8.27}
    # moving a control takes its step value; the other control stays exact
    tab.show_session(S1)
    assert tab.current() == (S1, 4, 0.853, 8.27) and tab.source_label.text().startswith("Your earlier choice")
    tab.min_corr_slider.setValue(86)
    assert tab.current()[2:] == (0.86, 8.27) and tab.min_corr_spin.text() == "0.86"
    tab.min_pnr_spin.setValue(8.3)  # the same slider position, typed: the typed value counts
    assert tab.current()[2:] == (0.86, 8.3)


# --- phase 4 final fix wave, D: minors ---

def test_busy_registry_file_keeps_the_shown_session(qapp, series_maps, make_tab, monkeypatch):
    """D2: the batch replaces a registry file while the tab reads it (PermissionError after the retries): the
    last good record is kept, as in Monitor, so the session and the slider values stay."""
    p = series_maps.paths
    tab = make_tab(p)
    tab.set_gsig(5)
    tab.set_thresholds(0.7, 12.0)
    update_record(p, S1, lambda r: r.stages["thresholds"].metrics.update(note="rewritten"))  # a new file version
    from pathlib import Path

    from bearmind import fileio

    real, held = fileio.read_text, p.registry_file(S1)

    def busy(path, *a, **k):
        if Path(path) == held:
            raise PermissionError(13, "The process cannot access the file", str(path))
        return real(path, *a, **k)

    monkeypatch.setattr(fileio, "read_text", busy)
    tab.refresh()
    assert tab.current() == (S1, 5, 0.7, 12.0) and S1 in tree_rows(tab)[0]
    assert tab.session_list.currentItem().data(0, Qt.ItemDataRole.UserRole) == S1


def test_maps_reload_sets_the_video_again(qapp, series_maps, make_tab):
    """D3: new maps of the shown session (MC or crop redone) reload the MC video too, with the contrast of the new
    mean image, so the movie and the maps match; the position is kept."""
    p = series_maps.paths
    tab = make_tab(p)
    tab.show()
    _wait(qapp, lambda: tab.video.slider.isEnabled())
    tab.video.seek(4)
    _wait(qapp, lambda: tab.video.current_index() == 4 and tab.video.shown_image() is not None)
    before = tab._video_args[2]
    with np.load(p.threshold_maps(S1)) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["mean"] = arrays["mean"] * 2.0
    np.savez_compressed(p.threshold_maps(S1), **arrays)
    tab.refresh()
    after = tab._video_args[2]
    assert after != before and after[1] == pytest.approx(2 * before[1], rel=1e-3)
    assert tab.video._contrast == after
    _wait(qapp, lambda: tab.video.slider.isEnabled() and tab.video.current_index() == 4)


def test_earlier_choice_names_its_config(qapp, series_maps, make_tab):
    """D5: "Your earlier choice" says which config the values come from."""
    p = series_maps.paths
    config.update_yaml(p.mouse_config("NOF_H01"), {"cnmf": {"gSig": 3, "min_corr": 0.7, "min_pnr": 6.0}})
    tab = make_tab(p)
    tab.show_session(S1)
    assert tab.source_label.text() == "Your earlier choice (mouse config)"
    config.update_yaml(p.session_config(S1), {"cnmf": {"min_pnr": 7.0}})
    tab.refresh()
    tab.show_session(S2)
    tab.show_session(S1)
    assert tab.source_label.text() == "Your earlier choice (session and mouse config)"


def test_mouse_write_logs_the_counters_of_the_shown_session(qapp, series_maps, make_tab):
    """D6: the person saw the counters of the shown session only: its row carries them, the others none."""
    p = series_maps.paths
    tab = make_tab(p, Dialogs(answer=True))
    tab.show_session(S1)
    seen = tab.counters_label.text()
    tab.accept_for_mouse()
    rows = {r["sid"]: r for r in _choices(p)}
    assert seen.startswith(f"blobs {rows[S1]['n_blobs']} ·") and rows[S1]["n_mirror_blobs"] != ""
    assert (rows[S2]["n_blobs"], rows[S2]["single_frac"], rows[S2]["n_mirror_blobs"]) == ("", "", "")


def test_monitor_button_opens_the_first_waiting_session(qapp, series_maps):
    """D7: Monitor's "Thresholds: N sessions →" opens the tab on the first session that waits."""
    from bearmind.gui.app import MainWindow

    w = MainWindow(series_maps.paths.root, message=lambda title, text: None)
    w.thresholds.show_session(S2)
    w.monitor.thresholds_button.click()
    assert w.tabs.currentIndex() == 2 and w.thresholds.current()[0] == S1
    w.thresholds.shutdown()
    w.close()


def test_video_is_about_as_large_as_one_map(qapp, series_maps, make_tab):
    """D8 (check finding 5): the MC video got ~200 px of a 1400 px window; it has a minimum width and the
    splitter gives it about the room of one map view."""
    from bearmind.gui.thresholds_tab import VIDEO_MIN_W

    tab = make_tab(series_maps.paths)
    assert VIDEO_MIN_W >= 300 and tab.video.minimumWidth() == VIDEO_MIN_W
    for size in ((1400, 860), (1100, 700)):
        tab.resize(*size)
        tab.show()
        qapp.processEvents()
        v, m = tab.video.width(), tab.corr_view.width()
        assert v >= VIDEO_MIN_W and abs(v - m) <= 0.3 * m, (size, v, m)
