from types import SimpleNamespace

import cv2
import numpy as np
from scipy import sparse

from bearmind import artifact_review, bad_frames, timestamps


def test_frame_repair_preserves_time_and_applies_the_selected_policy(monkeypatch):
    frames = [np.full((2, 2), v, dtype=np.uint8) for v in (0, 10, 250, 250, 40, 50)]
    assert [int(f[0, 0]) for f in bad_frames.filter_frames(frames, interpolate=[(2, 3)])] == [0, 10, 20, 30, 40, 50]
    assert [int(f[0, 0]) for f in bad_frames.filter_frames(frames, drop=[(2, 3)])] == [0, 10, 40, 50]
    axis = np.array([0,.1,.2,.3,.4,.9])
    plan = bad_frames.repair_plan([2, 3], [], "auto", .5, axis, 10, 6)
    assert plan == [{"first": 2, "last": 3, "duration_s": .2, "action": "interpolate"}]
    assert bad_frames.repair_plan([2, 3, 4], [], "auto", .5, axis, 10, 6)[0]["action"] == "report"
    rec = SimpleNamespace(n_frames=6, stages={"mc": SimpleNamespace(metrics={"dropped_ranges": [[0, 0], [2, 3]],
                                                                              "n_frames": 3})})
    raw = timestamps.Timestamps(np.arange(6), 1000 * axis, timestamps.BONSAI)
    monkeypatch.setattr(timestamps, "session_timestamps", lambda *_: raw)
    assert timestamps.analysis_frame_count(rec) == 3
    assert np.allclose(timestamps.session_time_s(None, rec), [.1, .4, .9])


def test_stripe_meter_detects_a_displaced_band():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (128, 128)).astype(np.uint8), (0, 0), 3)
    meter = bad_frames.StripeMeter()
    for i in range(400):
        frame = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
        if i == 250:
            frame[40:60] = np.roll(frame[40:60], 37, axis=1)
        meter.add(frame)
    assert meter.suspects() == [250]


def test_wave_candidates_require_nearby_components_and_edge_mask_honors_concavity():
    traces = np.zeros((4, 100))
    traces[:, 20:28] = 10
    est = SimpleNamespace(C=traces, neurons_sn=np.ones(4))
    centres = np.array([[0, 0], [1, 1], [2, 2], [50, 50]])
    result = artifact_review.wave_candidates(est, np.arange(4), centres, np.arange(100) / 20, 20,
                                             k_mad=5., min_nearby=3)
    assert result["events"][0]["components"] == [0, 1, 2]
    assert result["events"][0]["duration_s"] == .4 and result["severe"]
    lit = np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    spatial = sparse.csc_matrix((np.ones(2), ([1, 4], [0, 1])), shape=(9, 2))
    assert artifact_review.edge_candidates(spatial, [20, 21], lit) == [
        {"component": 21, "outside_lit": 1.0, "outside_activity": None}]
