r"""Finished stages of the LT1 series must keep their stored params_hash after any code change (global constraint of
the LT1 fixes plan: no finished stage of the running batch goes out of date).

Source of the snapshot: H:\Projects\LT1\CalciumData\_bearmind (9 sessions: registry records and configs),
copied 2026-11-17. The realdata test runs the same check on the live series (BEARMIND_LT1_SERIES), read-only.
"""
import pytest

from bearmind import config, registry, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, State
from bearmind.timestamps import read_timestamps

from tests.bearmind.lt1_snapshot import SNAPSHOT, SNAPSHOT_SIDS, copy_series, live_series_root, stale_finished, write_machine


def test_snapshot_finished_stages_stay_current(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    recs, broken = registry.read_all(paths)
    assert broken == [] and sorted(r.sid for r in recs) == sorted(SNAPSHOT_SIDS)
    assert sum(r.stages[s].state is State.DONE for r in recs for s in STAGES) >= 30  # not vacuous
    assert stale_finished(paths) == []
    view = runner.StageView(paths)
    assert all(view.current(r, "mc") for r in recs)  # the placeholder outputs make "current" judge by hashes


def test_snapshot_check_sees_a_changed_input(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    config.update_yaml(paths.mouse_config("LT1_J03"), {"mc": {"max_shifts": [30, 30]}})
    assert stale_finished(paths) == [("LT1_J03_2D", "cnmf"), ("LT1_J03_2D", "mc"), ("LT1_J03_2D", "thresholds")]


def test_fps_change_keeps_mc_and_maps_current(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")
    registry.update_record(paths, "LT1_J03_2D", lambda r: setattr(r, "fps", r.fps + 0.01))
    rec = registry.load_record(paths, "LT1_J03_2D")
    view = runner.StageView(paths)
    assert view.current(rec, "mc") and view.current(rec, "thresholds")
    assert not view.current(rec, "cnmf")  # fr is an input of CNMF-E


def test_legacy_mc_hash_needs_its_own_fps(tmp_path):
    paths = copy_series(SNAPSHOT, tmp_path / "lt1")

    def other_fps(r):
        r.stages["mc"].params["fps"] = 12.5

    registry.update_record(paths, "LT1_J03_2D", other_fps)
    assert ("LT1_J03_2D", "mc") in stale_finished(paths)


@pytest.mark.realdata
def test_live_series_finished_stages_stay_current():
    """Reads the live series in place (nothing is copied or written): the gate, autoinspect and export hashes take
    stats of files that a copy of the service folder would not have."""
    root = live_series_root()
    if root is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    paths = SeriesPaths(root)
    write_machine()  # the test's own BEARMIND_HOME, not the real machine file
    recs, broken = registry.read_all(paths)
    assert recs and broken == [], f"no registry under {root}\_bearmind, or unreadable records: {broken}"
    assert stale_finished(paths) == []


@pytest.mark.realdata
def test_live_timestamp_files_lose_a_row_only_in_j05_1d():
    root = live_series_root()
    if root is None:
        pytest.skip("BEARMIND_LT1_SERIES is not set")
    paths = SeriesPaths(root)
    recs, _ = registry.read_all(paths)
    dropped = []
    for r in recs:
        for p in r.parts:
            f = paths.root / p.timestamps_file
            ts = read_timestamps(f)
            rows = [ln for ln in f.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            assert len(ts.raw) + (ts.dropped_row is not None) == len(rows), r.sid
            if ts.dropped_row is not None:
                dropped.append(r.sid)
    assert len(recs) == 84 and dropped == ["LT1_J05_1D"]
