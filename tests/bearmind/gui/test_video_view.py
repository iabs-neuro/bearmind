import time

import numpy as np

from bearmind.gui.video_view import VideoView


def _wait(qapp, cond, timeout=10.0):
    t0 = time.monotonic()
    while not cond():
        qapp.processEvents()
        assert time.monotonic() - t0 < timeout, "timeout"
        time.sleep(0.005)


def _movie(make_avi, tmp_path, n=40):
    # frame i is the constant 5*i, so every frame is identifiable by its pixels
    frames = np.stack([np.full((12, 16), 5 * i, np.uint8) for i in range(n)])
    return make_avi(tmp_path / "mc.avi", frames), n


def test_shows_requested_frame(qapp, make_avi, tmp_path):
    p, n = _movie(make_avi, tmp_path)
    v = VideoView()
    v.set_video(p, 20)
    v.seek(7)
    _wait(qapp, lambda: v.shown_image() is not None and v.current_index() == 7)
    img = v.shown_image()
    assert (img.width(), img.height()) == (16, 12)
    assert img.pixelColor(3, 3).red() == 35
    assert v.slider.maximum() == n - 1
    v.close()


def test_contrast_stretches(qapp, make_avi, tmp_path):
    p, _ = _movie(make_avi, tmp_path)
    v = VideoView()
    v.set_video(p, 20, contrast=(20.0, 120.0))
    v.seek(10)  # value 50 -> (50-20)/100*255 = 76
    _wait(qapp, lambda: v.shown_image() is not None)
    assert abs(v.shown_image().pixelColor(0, 0).red() - 76) <= 1
    v.close()


def test_rapid_seek_shows_last_request(qapp, make_avi, tmp_path):
    p, n = _movie(make_avi, tmp_path)
    v = VideoView()
    shown = []
    v.frame_shown.connect(shown.append)
    v.set_video(p, 20)
    _wait(qapp, lambda: v.slider.isEnabled())
    for i in range(50):
        v.seek(i % n)
    last = 49 % n
    _wait(qapp, lambda: shown and shown[-1] == last)
    assert v.current_index() == last
    v.close()


def test_missing_video_message(qapp, tmp_path):
    p = tmp_path / "nope.avi"
    v = VideoView()
    v.set_video(p, 20)
    _wait(qapp, lambda: v.image_label.text() != "")
    assert v.image_label.text() == f"MC video not found: {p}"
    assert not v.slider.isEnabled()
    v.close()


def test_play_advances(qapp, make_avi, tmp_path):
    p, n = _movie(make_avi, tmp_path)
    v = VideoView()
    v.set_video(p, 20)
    _wait(qapp, lambda: v.slider.isEnabled())
    v.play_button.click()
    assert v.play_button.text() == "❚❚" and v.play_button.toolTip() == "Pause"
    _wait(qapp, lambda: v.current_index() >= 3)
    v.play_button.click()
    assert v.play_button.text() == "▶" and v.play_button.toolTip() == "Play"
    i = v.current_index()
    qapp.processEvents()
    assert v.current_index() == i
    v.close()


def test_position_label_format(qapp, make_avi, tmp_path):
    p, n = _movie(make_avi, tmp_path)
    v = VideoView()
    v.set_video(p, 20)
    _wait(qapp, lambda: v.slider.isEnabled())
    v.seek(30)
    assert v.position_label.text() == f"frame 31 / {n} · 00:01.5"
    v.close()


def test_file_released_on_session_change(qapp, make_avi, tmp_path):
    p, _ = _movie(make_avi, tmp_path)
    v = VideoView()
    v.set_video(p, 20)
    v.seek(3)
    _wait(qapp, lambda: v.shown_image() is not None)
    v.set_video(None, 20)
    assert v._worker is None
    p.rename(tmp_path / "moved.avi")  # fails on Windows while a handle is open
    v.close()
