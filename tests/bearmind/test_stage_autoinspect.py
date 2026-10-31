import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ae_launch
from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import SessionRecord, save_record
from bearmind.stages import autoinspect

SID = "NOF_H01_1D"


def _setup(tmp_path, fps=19.76):
    paths = SeriesPaths(tmp_path / "series")
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=10, fps=fps,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    save_record(paths, rec)
    return paths, rec, config.load_defaults()


def _processed(paths):
    return paths.inspection_dir(SID) / f"{SID}_processed.pickle"


def _write_processed(paths):
    """What run_auto_inspection does for a session with components: save the processed pickle."""
    _processed(paths).parent.mkdir(parents=True, exist_ok=True)
    _processed(paths).write_bytes(pickle.dumps(SimpleNamespace(a=1)))


def test_passes_production_parameters(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    calls = []

    def fake(estimates_path, fps=None, **kw):
        calls.append((estimates_path, fps, kw))
        _write_processed(paths)
        return {"summary": {"n_initial": 100, "n_final": 60}}

    monkeypatch.setattr(ae_launch, "run_auto_inspection", fake)
    res = autoinspect.run(paths, rec, cfg)
    (est_path, fps, kw), = calls
    assert est_path == str(paths.estimates(SID))
    assert fps == 20 and isinstance(fps, int)
    assert kw["pxlthr_distance_boundary"] == 15 and kw["brain"] == "hybrid"
    assert kw["session_name"] == SID
    assert kw["artifacts_dir"] == str(paths.inspection_dir(SID))
    assert Path(kw["ml_model_path"]).is_absolute() and Path(kw["ml_model_path"]).exists()
    assert "model_path" not in kw
    assert res.outputs == {"processed": f"5_Inspection/{SID}/{SID}_processed.pickle"}
    assert res.metrics == {"n_initial": 100, "n_final": 60, "deleted_frac": 0.4, "autoinspection_fps": 20}
    assert res.anomalies == []


def test_zero_components_saves_estimates_with_warning(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    est = SimpleNamespace(idx_components=[], metrics_df=pd.DataFrame())
    # the real function returns early for 0 components and saves nothing
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: {"estimates": est, "summary": {"n_initial": 0, "n_final": 0}})
    res = autoinspect.run(paths, rec, cfg)
    assert res.metrics["deleted_frac"] == 0.0
    assert res.outputs == {"processed": paths.rel(_processed(paths))}
    back = pickle.loads(_processed(paths).read_bytes())
    assert back.idx_components == [] and back.metrics_df.empty
    assert not list(paths.inspection_dir(SID).glob("*.partial"))
    assert [(a.code, a.level) for a in res.anomalies] == [("no_components", "warning")]


def test_zero_components_replaces_stale_processed(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    _write_processed(paths)  # from an earlier run on other estimates
    est = SimpleNamespace(idx_components=[], metrics_df=pd.DataFrame())
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: {"estimates": est, "summary": {"n_initial": 0, "n_final": 0}})
    res = autoinspect.run(paths, rec, cfg)
    assert pickle.loads(_processed(paths).read_bytes()).idx_components == []
    assert [a.code for a in res.anomalies] == ["no_components"]


def test_missing_processed_estimates_raise(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: {"summary": {"n_initial": 5, "n_final": 3}})
    with pytest.raises(FileNotFoundError, match="autoinspection produced no processed estimates"):
        autoinspect.run(paths, rec, cfg)


def test_uses_manual_result_when_cleaned(tmp_path, monkeypatch):
    from bearmind.stages import gate
    paths, rec, cfg = _setup(tmp_path)
    paths.manual_result(SID).parent.mkdir(parents=True)
    paths.manual_result(SID).write_bytes(b"x")
    gate.decide(paths, SID, "cleaned", "viktor")
    from bearmind.registry import load_record
    seen = []
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda p, fps=None, **k: seen.append(p) or _write_processed(paths)
                        or {"summary": {"n_initial": 1, "n_final": 1}})
    autoinspect.run(paths, load_record(paths, SID), cfg)
    assert seen == [str(paths.manual_result(SID))]


def test_artifacts_dir_writes_directly_into_folder(tmp_path):
    target = tmp_path / "out" / SID
    df = pd.DataFrame({"delete": [0, 1]})
    folder = ae_launch._save_inspection_artifacts(
        session_name=SID, decision_df=df, match_mtx=np.zeros((2, 2)), FCD=np.zeros((2, 2)),
        FBD=np.zeros((2, 2)), edge_info=None, est_processed=SimpleNamespace(a=1),
        artifacts_path=str(tmp_path / "ignored"), artifacts_dir=str(target))
    assert Path(folder) == target
    assert (target / f"{SID}_processed.pickle").exists()
    assert (target / f"{SID}_metrics_with_decisions.csv").exists()
    assert not (tmp_path / "ignored").exists()


def test_default_keeps_subfolder(tmp_path):
    folder = ae_launch._save_inspection_artifacts(
        session_name=SID, decision_df=pd.DataFrame({"delete": [0]}), match_mtx=np.zeros((1, 1)),
        FCD=np.zeros((1, 1)), FBD=np.zeros((1, 1)), edge_info=None, artifacts_path=str(tmp_path))
    assert Path(folder) == tmp_path / f"inspection_artifacts_{SID}"
