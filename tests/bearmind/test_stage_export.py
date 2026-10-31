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
H, W, T, N = 6, 8, 30, 4


def _est(n_frames=T):
    rng = np.random.default_rng(1)
    A = sp.lil_matrix((H * W, N), dtype=np.float32)
    for k in range(N):
        A[(k + 1) * 3 % (H * W), k] = 1.0
        A[(k + 1) * 3 % (H * W) + 1, k] = 0.5
    return SimpleNamespace(
        C=rng.random((N, n_frames)).astype(np.float32), A=A.tocsc(),
        SNR_comp=np.array([1.0, 2.0, 3.0, 4.0]),
        idx_components=np.array([0, 1, 2, 3]), idx_components_bad=np.array([]),
        imax=np.zeros((W, H)),  # shape[::-1] == (H, W) like A reshape in export_filters_mat
        cnmf_dict={"fr": 19.76, "gSig": (4, 4)},
        autoinspection_config={"ml_threshold": 0.5},
        metrics_df=pd.DataFrame({"component_idx": [0, 1, 2, 3],
                                 "ml_keep_probability": [0.9, 0.4, 0.6, 0.1]}),
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
                                           params_hash="abc123"),
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
    # reference: old export() on the same object with the same ML filter
    est = pickle.loads((paths.root / rec.stages["autoinspect"].outputs["processed"]).read_bytes())
    idx = np.array([0, 2])
    ref_npz, ref_json = old.export(old.extract_data(est, idx), old.build_metadata(
        est, 20.0, SID, component_indices=idx), SID, tmp_path / "ref")
    with np.load(paths.export_npz(SID)) as a, np.load(ref_npz) as b:
        assert set(a.files) == set(b.files)
        assert a["C"].shape == (2, T)
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
    assert prov["stage_params"] == {"autoinspect": "abc123"}
    assert prov["decisions"][0]["what"] == "thresholds"
    assert prov["inputs"] == ["1_Raw/x"]
    assert set(prov["packages"]) == {"caiman", "numpy", "driada", "interpret", "python"}
    rows = paths.export_timestamps(SID).read_text().split()
    assert len(rows) == T and rows[0] == "0" and rows[-1] == str((T - 1) * 500000)
    mat = scipy.io.loadmat(paths.export_filters(SID))
    assert mat["A"].shape == (2, H, W)
    assert res.metrics == {"n_neurons": 2, "median_snr": 2.0}
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
