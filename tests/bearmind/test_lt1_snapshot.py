r"""Finished stages of the LT1 series must keep their stored params_hash after any code change (global constraint of
the LT1 fixes plan: no finished stage of the running batch goes out of date).

Source of the snapshot: H:\Projects\LT1\CalciumData\_bearmind (9 sessions: registry records and configs),
copied 2026-11-17. The realdata test runs the same check on the live series (BEARMIND_LT1_SERIES), read-only.
"""
import pytest

from bearmind import config, registry, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, State

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
