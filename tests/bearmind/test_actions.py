"""Actions module: codes, unknown sessions raise before any write, same effects as the CLI commands."""
import re

import pytest
import yaml

from bearmind import actions
from bearmind.registry import STAGES, State, load_record, update_record
from tests.bearmind.test_cli import S1, S2, S3, _set, series  # noqa: F401  (the series fixture)


def _decisions(paths, sid):
    return [(d.what, d.value, d.who) for d in load_record(paths, sid).decisions]


def test_set_excluded_and_back(series):
    res = actions.set_excluded(series.paths, [S1], True, "ann")
    assert res.ok == [S1] and not res.failed
    assert load_record(series.paths, S1).excluded
    assert ("exclude", "true", "ann") in _decisions(series.paths, S1)
    actions.set_excluded(series.paths, [S1], False, "ann")
    assert not load_record(series.paths, S1).excluded
    assert ("exclude", "false", "ann") in _decisions(series.paths, S1)


def test_set_excluded_include_without_fps_warns(series):
    update_record(series.paths, S1, lambda r: setattr(r, "fps", None))
    res = actions.set_excluded(series.paths, [S1], False, "ann")
    assert res.warnings == [(S1, "no_fps_stays_excluded")]


def test_unknown_sid_writes_nothing(series):
    before = _decisions(series.paths, S1)
    with pytest.raises(actions.UnknownSession) as e:
        actions.set_excluded(series.paths, [S1, "NOPE"], True, "ann")
    assert e.value.unknown == ["NOPE"]
    assert not load_record(series.paths, S1).excluded
    with pytest.raises(actions.UnknownSession):
        actions.rerun_from(series.paths, [S1, "NOPE"], "mc", "ann")
    assert _decisions(series.paths, S1) == before


def test_rerun_unknown_stage_writes_nothing(series):
    before = _decisions(series.paths, S1)
    with pytest.raises(ValueError):
        actions.rerun_from(series.paths, [S1], "nope", "ann")
    assert _decisions(series.paths, S1) == before


def _wait_gate(series, sid):
    _set(series.paths, sid, "cnmf", State.DONE)
    _set(series.paths, sid, "gate", State.WAITING)
    update_record(series.paths, sid, lambda r: r.decisions.clear())


def test_gate_not_needed_all_waiting(series):
    _wait_gate(series, S1)
    res = actions.gate_not_needed(series.paths, None, "ann")
    assert res.ok == [S1] and not res.failed
    assert ("manual_inspection", "not_needed", "ann") in _decisions(series.paths, S1)
    assert not any(d[2] == "ann" for d in _decisions(series.paths, S2))
    assert not any(d[2] == "ann" for d in _decisions(series.paths, S3))


def test_gate_not_needed_already_passed(series):
    _set(series.paths, S1, "gate", State.DONE)
    before = _decisions(series.paths, S1)
    res = actions.gate_not_needed(series.paths, [S1], "ann")
    assert list(res.failed) == [S1] and res.failed[S1][0] == "already_passed"
    assert _decisions(series.paths, S1) == before
    res = actions.gate_not_needed(series.paths, [S1], "ann", force=True)
    assert res.ok == [S1]
    assert _decisions(series.paths, S1) != before


def test_gate_cleaned_without_file(series):
    before = _decisions(series.paths, S1)
    res = actions.gate_cleaned(series.paths, S1, "ann")
    assert res.failed == {S1: ("manual_result_missing", str(series.paths.manual_result(S1)))}
    assert not res.ok and _decisions(series.paths, S1) == before


def test_gate_cleaned_with_file(series):
    f = series.paths.manual_result(S1)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"x")
    res = actions.gate_cleaned(series.paths, S1, "ann")
    assert res.ok == [S1]
    assert ("manual_inspection", "cleaned", "ann") in _decisions(series.paths, S1)


def test_rerun_from(series):
    for st in STAGES:
        _set(series.paths, S1, st, State.DONE)
    res = actions.rerun_from(series.paths, [S1], "cnmf", "ann")
    assert res.ok == [S1]
    states = load_record(series.paths, S1).stages
    assert states["mc"].state is State.DONE and states["cnmf"].state is State.PENDING
    assert states["export"].state is State.PENDING


def test_retry_failed(series):
    for st in STAGES:
        _set(series.paths, S1, st, State.DONE)
    _set(series.paths, S1, "cnmf", State.FAILED, error="boom")
    res = actions.retry_failed(series.paths, [S1, S2], "ann")
    assert res.ok == [S1] and res.failed[S2][0] == "no_failed_stage"
    states = load_record(series.paths, S1).stages
    assert states["mc"].state is State.DONE
    assert all(states[s].state is State.PENDING for s in STAGES[STAGES.index("cnmf"):])


def test_retry_failed_none_means_all(series):
    _set(series.paths, S1, "cnmf", State.FAILED, error="boom")
    _set(series.paths, S2, "cnmf", State.FAILED, error="boom")
    update_record(series.paths, S2, lambda r: setattr(r, "excluded", True))
    res = actions.retry_failed(series.paths, None, "ann")
    assert res.ok == [S1] and not res.failed
    assert load_record(series.paths, S2).stages["cnmf"].state is State.FAILED


@pytest.mark.parametrize("args", [(2, 92.0, 8.0), (2, float("nan"), 8.0), (2, 0.8, 0), (0, 0.8, 8.0)])
def test_set_thresholds_invalid(series, args):
    with pytest.raises(ValueError):
        actions.set_thresholds(series.paths, S1, *args)
    assert not series.paths.session_config(S1).exists()


def test_set_thresholds_message_and_write(series):
    with pytest.raises(ValueError, match=r"--min-corr must be a number above 0 and at most 1, got 92.0"):
        actions.set_thresholds(series.paths, S1, 2, 92.0, 8.0)
    res = actions.set_thresholds(series.paths, S1, 3.0, 0.8, 8.0)
    assert res.ok == [S1]
    cfg = yaml.safe_load(series.paths.session_config(S1).read_text(encoding="utf-8"))
    assert cfg["cnmf"] == {"gSig": 3, "min_corr": 0.8, "min_pnr": 8.0}
    with pytest.raises(actions.UnknownSession):
        actions.set_thresholds(series.paths, "NOPE", 3, 0.8, 8.0)


def test_accept_suggested_mixed(series):
    _set(series.paths, S1, "thresholds", State.DONE, suggested={"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0})
    _set(series.paths, S2, "thresholds", State.DONE)
    res = actions.accept_suggested(series.paths, [S1, S2])
    assert res.ok == [S1] and res.failed == {S2: ("no_suggestion", "")}
    assert series.paths.session_config(S1).exists() and not series.paths.session_config(S2).exists()


def test_accept_suggested_invalid(series):
    _set(series.paths, S1, "thresholds", State.DONE, suggested={"gSig": 4, "min_corr": 7, "min_pnr": 8.0})
    res = actions.accept_suggested(series.paths, [S1])
    assert res.failed[S1][0] == "invalid_suggestion" and "min-corr" in res.failed[S1][1]
    assert not series.paths.session_config(S1).exists()


def test_accept_mismatch_warns_without_error(series):
    res = actions.accept_mismatch(series.paths, [S1], "ann")
    assert res.ok == [S1] and res.warnings == [(S1, "no_mismatch_error")]
    assert ("accept_mismatch", "true", "ann") in _decisions(series.paths, S1)


def test_failure_text_english_ascii_free_of_cyrillic():
    for code in ("no_suggestion", "invalid_suggestion", "manual_result_missing", "no_failed_stage",
                 "already_passed", "something_new"):
        text = actions.failure_text(code, "detail")
        assert text and not re.search("[Ѐ-ӿ]", text)


def test_export_summary(series, tmp_path):
    dest = actions.export_summary(series.paths, tmp_path / "out.csv")
    assert dest.is_file()
    assert "sid" in dest.read_text(encoding="utf-8").splitlines()[0].split(",")


def test_export_summary_equals_status_csv(series, capsys):
    from tests.bearmind.test_cli import _status

    _status(series, capsys)
    from_status = series.paths.summary_csv.read_bytes()
    dest = actions.export_summary(series.paths, series.root / "copy.csv")
    assert dest.read_bytes() == from_status
