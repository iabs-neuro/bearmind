import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import ae_launch
import estimates_compression
from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import SessionRecord, save_record
from bearmind.stages import autoinspect

SID = "NOF_H01_1D"


def _input_est(n=3, n_good=None):
    """A CNMF-E output as the stage gets it: per-component arrays in the order of A's columns."""
    rng = np.random.default_rng(0)
    good = np.arange(n if n_good is None else n_good)
    return SimpleNamespace(
        A=sp.csc_matrix(rng.random((16, n))), C=rng.random((n, 20)), YrA=rng.random((n, 20)),
        S=rng.random((n, 20)), bl=rng.random(n), c1=rng.random(n), neurons_sn=rng.random(n),
        g=rng.random((n, 1)), SNR_comp=np.arange(n, dtype=float) + 1, r_values=np.linspace(0.5, 1, n),
        idx_components=good, idx_components_bad=np.setdiff1d(np.arange(n), good), dims=(4, 4))


def _write_input(path, est):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(est, fh)


def _setup(tmp_path, fps=19.76):
    paths = SeriesPaths(tmp_path / "series")
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=10, fps=fps,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    save_record(paths, rec)
    _write_input(paths.estimates(SID), _input_est())
    return paths, rec, config.load_defaults()


def _processed_df(n_ok, n_merged, n_deleted):
    """metrics_df of processed estimates: kept rows, merge products (P = NaN), rows marked delete."""
    n = n_ok + n_merged + n_deleted
    return pd.DataFrame({
        "component_idx": np.arange(n),
        "decision": ["ok"] * n_ok + ["from_merge"] * n_merged + ["delete"] * n_deleted,
        "delete": [0] * (n_ok + n_merged) + [1] * n_deleted,
        "ml_keep_probability": [0.9] * n_ok + [np.nan] * n_merged + [0.3] * n_deleted,
    })


def _result(n_initial, n_deleted, n_final, df):
    return {"estimates": SimpleNamespace(metrics_df=df),
            "summary": {"n_initial": n_initial, "n_deleted": n_deleted, "n_final": n_final}}


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
        return _result(100, 40, 60, _processed_df(55, 5, 0))

    monkeypatch.setattr(ae_launch, "run_auto_inspection", fake)
    res = autoinspect.run(paths, rec, cfg)
    (est_path, fps, kw), = calls
    assert est_path == str(paths.estimates(SID))
    assert fps == 20 and isinstance(fps, int)
    assert kw["pxlthr_distance_boundary"] == 15 and kw["brain"] == "hybrid"
    assert kw["session_name"] == SID
    assert kw["seed"] == 0
    assert kw["artifacts_dir"] == str(paths.inspection_dir(SID))
    assert Path(kw["ml_model_path"]).is_absolute() and Path(kw["ml_model_path"]).exists()
    assert "model_path" not in kw
    assert res.outputs == {"processed": f"5_Inspection/{SID}/{SID}_processed.pickle"}
    assert res.metrics == {"n_initial": 100, "n_after_merge": 60, "n_deleted": 40, "n_merge_groups": 5,
                           "n_kept": 60, "deleted_frac": 0.4, "autoinspection_fps": 20}
    assert res.anomalies == []


def test_zero_components_saves_estimates_with_warning(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    est = SimpleNamespace(idx_components=[], metrics_df=pd.DataFrame())
    # the real function returns early for 0 components and saves nothing
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: {"estimates": est,
                                         "summary": {"n_initial": 0, "n_deleted": 0, "n_final": 0}})
    res = autoinspect.run(paths, rec, cfg)
    assert res.metrics == {"n_initial": 0, "n_after_merge": 0, "n_deleted": 0, "n_merge_groups": 0, "n_kept": 0,
                           "deleted_frac": 0.0, "autoinspection_fps": 20}
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
                        lambda *a, **k: {"estimates": est,
                                         "summary": {"n_initial": 0, "n_deleted": 0, "n_final": 0}})
    res = autoinspect.run(paths, rec, cfg)
    assert pickle.loads(_processed(paths).read_bytes()).idx_components == []
    assert [a.code for a in res.anomalies] == ["no_components"]


def test_missing_processed_estimates_raise(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: {"summary": {"n_initial": 5, "n_deleted": 2, "n_final": 3}})
    with pytest.raises(FileNotFoundError, match="autoinspection produced no processed estimates"):
        autoinspect.run(paths, rec, cfg)


def test_uses_manual_result_when_cleaned(tmp_path, monkeypatch):
    from bearmind.stages import gate
    paths, rec, cfg = _setup(tmp_path)
    _write_input(paths.manual_result(SID), _input_est())
    gate.decide(paths, SID, "cleaned", "viktor")
    from bearmind.registry import load_record
    seen = []
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda p, fps=None, **k: seen.append(p) or _write_processed(paths)
                        or _result(1, 0, 1, _processed_df(1, 0, 0)))
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


@pytest.mark.parametrize("sid, n_initial, n_deleted, n_after_merge, n_kept, n_groups", [
    ("NOF_H01_1D", 224, 62, 194, 132, 18),  # series_short, from its processed pickles (ML-filter findings)
    ("NOF_H03_3D", 218, 117, 211, 94, 1),
])
def test_metrics_say_what_autoinspection_decided(tmp_path, monkeypatch, sid, n_initial, n_deleted,
                                                 n_after_merge, n_kept, n_groups):
    """Autoinspection marks components delete and keeps them; merges shrink the count. The metrics count
    each, and deleted_frac is the marked share, not the merge shrinkage."""
    paths, rec, cfg = _setup(tmp_path)
    df = _processed_df(n_kept - n_groups, n_groups, n_after_merge - n_kept)
    monkeypatch.setattr(ae_launch, "run_auto_inspection", lambda *a, **k: _write_processed(paths) or
                        _result(n_initial, n_deleted, n_after_merge, df))
    m = autoinspect.run(paths, rec, cfg).metrics
    assert (m["n_initial"], m["n_deleted"], m["n_after_merge"], m["n_kept"], m["n_merge_groups"]) == (
        n_initial, n_deleted, n_after_merge, n_kept, n_groups)
    assert m["deleted_frac"] == pytest.approx(n_deleted / n_initial)
    assert "n_final" not in m


def _refused(tmp_path, monkeypatch, est, match):
    paths, rec, cfg = _setup(tmp_path)
    _write_input(paths.estimates(SID), est)
    monkeypatch.setattr(ae_launch, "run_auto_inspection",
                        lambda *a, **k: pytest.fail("autoinspection ran on misaligned input"))
    with pytest.raises(ValueError, match=match) as ei:
        autoinspect.run(paths, rec, cfg)
    assert str(paths.estimates(SID)) in str(ei.value)


def test_refuses_compressed_estimates(tmp_path, monkeypatch):
    """estimates_compression keeps arr[:N] of SNR_comp / r_values: lengths match A, the order does not."""
    est, _, _ = estimates_compression.compress_estimates_ultra_lightweight(_input_est(n=5, n_good=3))
    assert est.A.shape[1] == len(est.SNR_comp) == len(est.r_values) == 3  # the length check alone passes
    _refused(tmp_path, monkeypatch, est, "idx_components_bad_original")


@pytest.mark.parametrize("attr", ["SNR_comp", "r_values"])
def test_refuses_per_component_arrays_of_another_length(tmp_path, monkeypatch, attr):
    est = _input_est(n=4)
    setattr(est, attr, getattr(est, attr)[:3])  # what CaImAn's manual_merge leaves after merging two
    _refused(tmp_path, monkeypatch, est, rf"len\({attr}\) = 3 but A has 4 components")


def test_refuses_missing_snr(tmp_path, monkeypatch):
    est = _input_est()
    del est.SNR_comp
    _refused(tmp_path, monkeypatch, est, "no SNR_comp")


def test_seed_is_an_autoinspect_input(tmp_path):
    from bearmind import runner
    paths, rec, cfg = _setup(tmp_path)
    p0 = runner.stage_params("autoinspect", rec, cfg, paths)
    cfg1 = copy.deepcopy(cfg); cfg1["autoinspect"]["seed"] = 1
    p1 = runner.stage_params("autoinspect", rec, cfg1, paths)
    assert p0["autoinspect"]["seed"] == 0
    assert config.params_hash(p0) != config.params_hash(p1)
