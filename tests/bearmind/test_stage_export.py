import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

import export_estimates_data as old
from bearmind import config, provenance
from bearmind.paths import SeriesPaths
from bearmind.registry import (Decision, RawPart, SessionRecord, StageRecord, load_record,
                               save_record)
from bearmind.stages import export

SID = "NOF_H01_1D"
H, W, T, N = 6, 8, 30, 5


def _est(n_frames=T):
    rng = np.random.default_rng(1)
    A = sp.lil_matrix((H * W, N), dtype=np.float32)
    for k in range(N):
        A[(k + 1) * 3 % (H * W), k] = 1.0
        A[(k + 1) * 3 % (H * W) + 1, k] = 0.5
    return SimpleNamespace(
        C=rng.random((N, n_frames)).astype(np.float32), A=A.tocsc(),
        # after a merge CaImAn leaves SNR_comp in the old order: it must not be read for the exported rows
        SNR_comp=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        idx_components=np.array([0, 1, 2, 3, 4]), idx_components_bad=np.array([]),
        imax=np.zeros((W, H)),  # shape[::-1] == (H, W) like A reshape in export_filters_mat
        cnmf_dict={"fr": 19.76, "gSig": (4, 4)},
        autoinspection_config={"ml_threshold": 0.5},
        # 0: an SNR duplicate (marked delete, P above threshold), 4: a merge product (P = NaN)
        metrics_df=pd.DataFrame({"component_idx": [0, 1, 2, 3, 4],
                                 "ml_keep_probability": [0.9, 0.4, 0.6, 0.1, np.nan],
                                 "delete": [1, 1, 0, 1, 0],
                                 "decision": ["delete", "delete", "ok", "delete", "from_merge"],
                                 "caiman_snr": [5.0, 1.5, 7.0, 0.5, 9.0]}),
    )


def _setup(tmp_path, layout="bonsai", n_frames=T, est_frames=None, model=True):
    paths = SeriesPaths(tmp_path / "series")
    if layout == "bonsai":
        ts = paths.timestamps_root / f"{SID}_Mini_TS.csv"
        ts.parent.mkdir(parents=True)
        ts.write_text("".join(f"{i * 500000}\n" for i in range(n_frames + 3)), encoding="utf-8")
        parts = [RawPart("1_Raw/x", ["a.avi"], n_frames, (H, W), paths.rel(ts), "bonsai_ticks",
                         None, None)]
    else:
        parts = []
        for k, hms in enumerate(["10_00_00", "11_00_00"]):
            d = paths.raw_root / "J" / hms
            d.mkdir(parents=True)
            f = d / "timeStamps.csv"
            f.write_text("Frame Number,Time Stamp (ms),Buffer Index\n"
                         + "".join(f"{i},{i * 50},0\n" for i in range(15)), encoding="utf-8")
            parts.append(RawPart(paths.rel(d), ["0.avi"], 15, (H, W), paths.rel(f), "ucla_ms",
                                 1728989165178.0 + k * 3600_000, None))
    proc = paths.inspection_dir(SID) / f"{SID}_processed.pickle"
    proc.parent.mkdir(parents=True)
    with open(proc, "wb") as fh:
        pickle.dump(_est(est_frames or n_frames), fh)
    rec = SessionRecord(
        sid=SID, mouse="NOF_H01", mode=layout, parts=parts, n_frames=n_frames, fps=19.76,
        fps_info={"source": "timestamps", "n_outliers": 1, "max_gap_ms": 60.0}, anomalies=[],
        decisions=[Decision("thresholds", "4", "viktor", "2026-10-31T00:00:00")], excluded=False,
        stages={"autoinspect": StageRecord(outputs={"processed": paths.rel(proc)},
                                           metrics={"autoinspection_fps": 20},
                                           params_hash="abc123",
                                           params={"autoinspect": {"n_iter": 3}, "fps": 20}),
                "mc": StageRecord()})
    save_record(paths, rec)
    cfg = config.load_defaults()
    if model:
        m = tmp_path / "model.pkl"
        m.write_bytes(b"model-bytes")
        cfg["autoinspect"]["model_path"] = str(m)
    return paths, rec, cfg


def test_export_matches_old_structure(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    res = export.run(paths, rec, cfg)
    # reference: old export() on the same object with the same ML filter and the registry's fps
    est = pickle.loads((paths.root / rec.stages["autoinspect"].outputs["processed"]).read_bytes())
    idx = np.array([0, 2])
    ref_npz, ref_json = old.export(old.extract_data(est, idx), old.build_metadata(
        est, rec.fps, SID, component_indices=idx), SID, tmp_path / "ref")
    with np.load(paths.export_npz(SID)) as a, np.load(ref_npz) as b:
        assert set(a.files) == set(b.files)
        assert a["C"].shape == (2, T)
        assert all(np.array_equal(a[k], b[k]) for k in a.files)  # content as the old export
        assert float(a["fps"]) == 19.76  # the exact rate from the timestamps, not the rounded 20
    meta = json.loads(paths.export_json(SID).read_text())
    ref = json.loads(ref_json.read_text())
    assert set(ref) | {"autoinspection_fps", "provenance", "timestamps_file"} <= set(meta)
    assert meta["fps"] == 19.76
    assert meta["autoinspection_fps"] == 20
    assert meta["timestamps_file"] == f"{SID}_timestamps.csv"
    assert meta["feedback_applied"] is False
    assert meta["autoinspection_stats"]["n_after_ml_filter"] == 2
    prov = meta["provenance"]
    assert prov["mode"] == "bonsai" and prov["fps"]["value"] == 19.76
    assert prov["model"]["path"] == cfg["autoinspect"]["model_path"] and len(prov["model"]["sha256"]) == 64
    assert prov["stage_params"] == {"autoinspect": {"autoinspect": {"n_iter": 3}, "fps": 20}}
    assert prov["decisions"][0]["what"] == "thresholds"
    assert prov["inputs"] == ["1_Raw/x"]
    assert set(prov["packages"]) == {"caiman", "numpy", "driada", "interpret", "python"}
    rows = paths.export_timestamps(SID).read_text().split()
    assert len(rows) == T and rows[0] == "0" and rows[-1] == str((T - 1) * 500000)
    mat = scipy.io.loadmat(paths.export_filters(SID))
    assert mat["A"].shape == (2, H, W)
    ref_mat = scipy.io.loadmat(old.export_filters_mat(est, idx, tmp_path / "ref" / "f.mat"))
    assert np.array_equal(mat["A"], ref_mat["A"])
    rule = {"component_rule": "ml_probability", "n_exported": 2, "n_snr_duplicates_exported": 1,
            "n_merge_products_dropped": 1}
    assert {k: meta[k] for k in rule} == rule  # json fields only added
    # median SNR of the exported rows (0, 2) from metrics_df, not SNR_comp[[0, 2]] = 2.0
    assert res.metrics == {"n_neurons": 2, "median_snr": 6.0, **rule}
    assert set(res.outputs) == {"npz", "json", "timestamps", "filters"}
    assert res.outputs["npz"] == paths.rel(paths.export_npz(SID))
    assert not list((paths.root / "6_Export").rglob("*.partial*"))
    assert not list((paths.root / "6_Export").rglob(".partial_*"))


def test_stitched_ucla_timestamps(tmp_path):
    paths, rec, cfg = _setup(tmp_path, layout="legacy", n_frames=30)
    export.run(paths, rec, cfg)
    lines = paths.export_timestamps(SID).read_text().splitlines()
    assert lines[0] == "Frame Number,Time Stamp (ms),Buffer Index"
    assert len(lines) == 31
    assert lines[1].split(",")[1] == "0" and lines[-1].split(",")[1] == str(3600_000 + 14 * 50)


def test_part_with_fewer_rows_than_frames_is_rejected(tmp_path):
    import pytest
    paths, rec, cfg = _setup(tmp_path, layout="legacy", n_frames=30)
    rec.parts[0].n_frames = 16  # registry from before per-part capping: part A has 15 rows
    with pytest.raises(ValueError, match="rescan"):
        export.run(paths, rec, cfg)
    assert not paths.export_json(SID).exists()


def test_frame_mismatch_warns_and_writes_min(tmp_path):
    paths, rec, cfg = _setup(tmp_path, n_frames=30, est_frames=25)
    res = export.run(paths, rec, cfg)
    assert [a.code for a in res.anomalies] == ["export_frames_mismatch"]
    assert res.anomalies[0].level == "warning"
    assert len(paths.export_timestamps(SID).read_text().split()) == 25


def test_provenance_git_and_missing_package(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    p = provenance.collect(paths, rec, cfg)
    assert p["code"]["git_commit"] is None or len(p["code"]["git_commit"]) == 40
    monkeypatch.setattr(provenance.subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("no git")))
    p = provenance.collect(paths, rec, cfg)
    assert p["code"] == {"git_commit": None, "git_dirty": None}


def test_no_filters_removes_stale_file_and_unlists_output(tmp_path, monkeypatch):
    paths, rec, cfg = _setup(tmp_path)
    paths.export_filters(SID).parent.mkdir(parents=True)
    paths.export_filters(SID).write_bytes(b"stale")
    monkeypatch.setattr(export.old, "export_filters_mat", lambda *a, **k: None)
    res = export.run(paths, rec, cfg)
    assert "filters" not in res.outputs and not paths.export_filters(SID).exists()
    assert "no_filters" in [a.code for a in res.anomalies]


def test_short_timestamps_warn(tmp_path):
    paths, rec, cfg = _setup(tmp_path, n_frames=30)
    paths.timestamps_root.joinpath(f"{SID}_Mini_TS.csv").write_text(
        "".join(f"{i * 500000}\n" for i in range(20)), encoding="utf-8")
    res = export.run(paths, rec, cfg)
    assert "export_frames_mismatch" in [a.code for a in res.anomalies]
    assert len(paths.export_timestamps(SID).read_text().split()) == 20


def test_replace_failure_leaves_no_json(tmp_path, monkeypatch):
    import pytest
    paths, rec, cfg = _setup(tmp_path)
    export.run(paths, rec, cfg)
    assert paths.export_json(SID).exists()
    real = export.os.replace

    def flaky(src, dst):
        if str(dst).endswith("_data.npz"):
            raise OSError("boom")
        return real(src, dst)

    monkeypatch.setattr(export.os, "replace", flaky)
    with pytest.raises(OSError):
        export.run(paths, rec, cfg)
    assert not paths.export_json(SID).exists()
    assert not (paths.root / "6_Export" / "npz" / f".partial_{SID}").exists()


def _counts(est):
    idx, info = export._ml_filter(est)
    return export.rule_counts(est, idx, info)


def test_rule_counts_on_series_short_shape():
    """Counts of the ML-probability rule on rows shaped like NOF_H03_3D (ML-filter findings): 93 kept with
    P >= 0.72, 1 merge product, 1 SNR duplicate exported, the rest marked delete below the threshold."""
    n_ok, n_dup, n_low = 93, 1, 116
    n = n_ok + 1 + n_dup + n_low
    df = pd.DataFrame({
        "component_idx": np.arange(n),
        "ml_keep_probability": [0.9] * n_ok + [np.nan] + [0.8] * n_dup + [0.2] * n_low,
        "delete": [0] * (n_ok + 1) + [1] * (n_dup + n_low),
        "decision": ["ok"] * n_ok + ["from_merge"] + ["delete"] * (n_dup + n_low),
        "caiman_snr": np.r_[np.full(n_ok, 5.0), 99.0, 7.0, np.full(n_low, 1.0)],
    })
    est = SimpleNamespace(idx_components=np.arange(n), metrics_df=df, autoinspection_config={"ml_threshold": 0.72})
    assert _counts(est) == {"component_rule": "ml_probability", "n_exported": 94, "n_snr_duplicates_exported": 1,
                            "n_merge_products_dropped": 1, "median_snr": 5.0}


def test_rule_counts_without_ml_or_components():
    no_ml = SimpleNamespace(idx_components=np.array([0, 1]), metrics_df=None)
    assert _counts(no_ml) == {"component_rule": "idx_components", "n_exported": 2,
                              "n_snr_duplicates_exported": None, "n_merge_products_dropped": None,
                              "median_snr": None}
    empty = SimpleNamespace(idx_components=np.array([], dtype=int), metrics_df=pd.DataFrame())
    assert _counts(empty) == {"component_rule": "idx_components", "n_exported": 0,
                              "n_snr_duplicates_exported": 0, "n_merge_products_dropped": 0, "median_snr": None}
    nan_snr = SimpleNamespace(idx_components=np.array([0]), autoinspection_config={"ml_threshold": 0.5},
                              metrics_df=pd.DataFrame({"component_idx": [0], "ml_keep_probability": [0.9],
                                                       "delete": [0], "decision": ["ok"],
                                                       "caiman_snr": [np.nan]}))
    assert np.isnan(_counts(nan_snr)["median_snr"])  # the check warns "not a number"
    no_idx = SimpleNamespace(idx_components=np.array([0, 1]), metrics_df=pd.DataFrame({"caiman_snr": [1.0, 2.0]}))
    assert _counts(no_idx) == {"component_rule": "idx_components", "n_exported": 2,
                               "n_snr_duplicates_exported": None, "n_merge_products_dropped": None,
                               "median_snr": None}


def test_export_metadata_carries_checks_and_anomalies(tmp_path):
    """I7 (spec 5.11): every check result and anomaly of the session goes to the export json."""
    from bearmind.registry import Anomaly, Check, State

    paths, rec, cfg = _setup(tmp_path, n_frames=30, est_frames=25)
    rec.stages["mc"] = StageRecord(state=State.DONE, checks=[
        Check("mc", "mc_max_shift_px", "warning", 12.0, "max shift 12.0 px (warn > 10, error > 30)"),
        Check("mc", "mc_template_corr", "ok", 0.9, "template corr 0.9 (warn < 0.7)")])
    rec.anomalies = [Anomaly("dropped_frames", "warning", "3 gaps in timestamps", data={"n_gaps": 3}),
                     Anomaly("fov_not_found", "warning", "no field of view", data={"stage": "crop"})]
    prov = provenance.collect(paths, rec, cfg)
    assert prov["checks"] == {"mc": [
        {"name": "mc_max_shift_px", "value": 12.0, "level": "warning",
         "message": "max shift 12.0 px (warn > 10, error > 30)"},
        {"name": "mc_template_corr", "value": 0.9, "level": "ok", "message": "template corr 0.9 (warn < 0.7)"}]}
    assert prov["anomalies"] == [
        {"code": "dropped_frames", "level": "warning", "stage": None, "message": "3 gaps in timestamps",
         "auto_fixed": False, "data": {"n_gaps": 3}},
        {"code": "fov_not_found", "level": "warning", "stage": "crop", "message": "no field of view",
         "auto_fixed": False, "data": {"stage": "crop"}}]
    export.run(paths, rec, cfg)
    meta = json.loads(paths.export_json(SID).read_text())["provenance"]
    assert meta["checks"]["mc"] == prov["checks"]["mc"]
    # the export's own checks and anomalies are in its json too (2 neurons: below the error threshold 5)
    assert {c["name"]: c["level"] for c in meta["checks"]["export"]} == {"export_n_neurons": "error",
                                                                         "export_median_snr": "ok"}
    assert [(a["code"], a["stage"]) for a in meta["anomalies"]] == [
        ("dropped_frames", None), ("fov_not_found", "crop"), ("export_frames_mismatch", "export")]


def _no_components_est():
    """What autoinspection leaves for a session without accepted components: an empty metrics_df."""
    est = _est()
    est.idx_components, est.idx_components_bad = np.array([], dtype=int), np.arange(N)
    est.metrics_df = pd.DataFrame()
    est.asp_cache, est.reconstructions = {}, {}
    return est


def _check_empty_export(paths, res, n_components):
    assert res.metrics["n_neurons"] == 0 and res.metrics["n_exported"] == 0
    a = [x for x in res.anomalies if x.code == "no_components"]
    assert len(a) == 1 and a[0].level == "warning"
    assert a[0].data == {"n_components": n_components, "n_exported": 0}
    meta = json.loads(paths.export_json(SID).read_text())
    assert meta["n_exported"] == 0 and meta["component_rule"] in ("ml_probability", "idx_components")
    assert [x["code"] for x in meta["provenance"]["anomalies"]] == ["no_components"]
    with np.load(paths.export_npz(SID)) as d:
        assert d["C"].shape == (0, T) and d["component_indices"].shape == (0,)
        return set(d.files)


def test_session_without_components_exports_empty(tmp_path):
    """9: zero components finish the export (old build_metadata raised KeyError 'component_idx')."""
    paths, rec, cfg = _setup(tmp_path)
    proc = paths.root / rec.stages["autoinspect"].outputs["processed"]
    with open(proc, "wb") as fh:
        pickle.dump(_no_components_est(), fh)
    res = export.run(paths, rec, cfg)
    keys = _check_empty_export(paths, res, 0)
    assert keys == {"C", "component_indices", "asp", "reconstructions", "fps"}  # keys of an inspected session
    mat = scipy.io.loadmat(paths.export_filters(SID))
    assert mat["A"].shape == (0, H, W) and mat["A"].dtype == np.uint8
    assert set(res.outputs) == {"npz", "json", "timestamps", "filters"}


def test_components_none_of_which_pass_the_rule(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    proc = paths.root / rec.stages["autoinspect"].outputs["processed"]
    est = _est()
    est.autoinspection_config = {"ml_threshold": 0.99}
    with open(proc, "wb") as fh:
        pickle.dump(est, fh)
    res = export.run(paths, rec, cfg)
    assert _check_empty_export(paths, res, N) == {"C", "component_indices", "fps"}  # as with components
    assert scipy.io.loadmat(paths.export_filters(SID))["A"].shape == (0, H, W)
