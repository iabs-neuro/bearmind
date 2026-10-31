import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_bm_qc import synthetic_movie  # noqa: E402

from bearmind import config  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import SessionRecord  # noqa: E402
from bearmind.stages import thresholds  # noqa: E402
from bearmind.video import write_ffv1  # noqa: E402

SID = "NOF_H01_1D"


def test_thresholds_stage(tmp_path):
    mov, _ = synthetic_movie(T=300)
    paths = SeriesPaths(tmp_path / "series")
    mc = paths.mc_video(SID)
    mc.parent.mkdir(parents=True)
    write_ffv1([np.clip(np.round(mov), 0, 255).astype(np.uint8)], mc, 20.0)
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=300, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    cfg = config.load_defaults()
    cfg["machine"] = {"n_cpus": 2, "temp_dir": str(tmp_path / "tmp")}
    cfg["thresholds"]["gsig_range"] = [3, 4]

    res = thresholds.run(paths, rec, cfg)

    out = paths.threshold_maps(SID)
    assert res.outputs == {"threshold_maps": paths.rel(out)}
    assert not list(out.parent.glob("*.partial*"))
    d = np.load(out)
    shape = mov.shape[1:]
    assert list(d["gsig"]) == [3, 4]
    for g in (3, 4):
        for k in ("cn", "pnr", "cn_mirror", "pnr_mirror"):
            a = d[f"{k}_{g}"]
            assert a.shape == shape and a.dtype == np.float32
    for k in ("mean", "max"):
        assert d[k].shape == shape and d[k].dtype == np.float32
    s = res.metrics["suggested"]
    assert s["gSig"] == 4
    assert 0 < s["min_corr"] <= 1
    assert s["min_pnr"] > 0
    assert s["method"] in ("mirror_fdr", "robust_bg")
