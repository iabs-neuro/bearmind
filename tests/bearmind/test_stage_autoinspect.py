from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

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


def test_passes_production_parameters(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    calls = []

    def fake(estimates_path, fps=None, **kw):
        calls.append((estimates_path, fps, kw))
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


def test_zero_components_gives_zero_deleted_frac(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: {"summary": {"n_initial": 0, "n_final": 0}})
    assert autoinspect.run(paths, rec, cfg).metrics["deleted_frac"] == 0.0


def test_uses_manual_result_when_cleaned(tmp_path, monkeypatch):
    from bearmind.stages import gate
    paths, rec, cfg = _setup(tmp_path)
    paths.manual_result(SID).parent.mkdir(parents=True)
    paths.manual_result(SID).write_bytes(b"x")
    gate.decide(paths, SID, "cleaned", "viktor")
    from bearmind.registry import load_record
    seen = []
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda p, fps=None, **k: seen.append(p) or {"summary": {"n_initial": 1, "n_final": 1}})
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
