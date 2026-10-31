"""Single export stage: npz + json + timestamps csv + filters mat, written atomically."""
from __future__ import annotations

import dataclasses
import os
import shutil

import numpy as np

import export_estimates_data as old
from bearmind import provenance
from bearmind.registry import Anomaly
from bearmind.stages import StageResult
from bearmind.timestamps import read_timestamps, stitch, write_timestamps


def _ml_filter(est):
    """Same rule as batch_export.export_one, without feedback."""
    idx = np.array(list(est.idx_components), dtype=int)
    n_before = len(idx)
    df = getattr(est, "metrics_df", None)
    if df is None or "ml_keep_probability" not in df.columns:
        return idx, {"ml_filtered": False, "threshold_used": None, "n_before": n_before,
                     "n_after": None}
    threshold = old.DEFAULT_ML_THRESHOLD
    if getattr(est, "autoinspection_config", None):
        threshold = est.autoinspection_config.get("ml_threshold", threshold)
    ok = set(df.loc[df["ml_keep_probability"] >= threshold, "component_idx"].tolist())
    idx = np.array([i for i in idx if i in ok], dtype=int)
    return idx, {"ml_filtered": True, "threshold_used": threshold, "n_before": n_before,
                 "n_after": len(idx)}


def _truncate(ts, n):
    return dataclasses.replace(ts, raw=ts.raw[:n], t_ms=ts.t_ms[:n],
                               buffer=None if ts.buffer is None else ts.buffer[:n])


def _stitched_timestamps(paths, rec):
    parts = []
    for i, p in enumerate(rec.parts):
        ts = read_timestamps(paths.root / p.timestamps_file, start_epoch_ms=p.start_ms)
        if i < len(rec.parts) - 1 and len(ts.raw) < p.n_frames:
            # the video of this part is longer than its timestamps: every later frame would get a
            # timestamp from the wrong position (registries scanned before parts were cut per part)
            raise ValueError(f"{p.folder}: {p.n_frames} frames but {len(ts.raw)} timestamp rows; "
                             f"rescan the series")
        parts.append(_truncate(ts, p.n_frames))
    return stitch(parts)


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    est = old.load_estimates(paths.root / rec.stages["autoinspect"].outputs["processed"])
    idx, ml_info = _ml_filter(est)
    n_frames = int(est.C.shape[1])
    anomalies = []
    ts = _stitched_timestamps(paths, rec)
    n_ts = min(n_frames, rec.n_frames, len(ts.raw))
    if n_frames != rec.n_frames or n_ts < n_frames:
        anomalies.append(Anomaly(
            "export_frames_mismatch", "warning",
            f"estimates have {n_frames} frames, registry has {rec.n_frames}, "
            f"timestamps have {len(ts.raw)}; timestamps written for {n_ts}",
            data={"estimates": n_frames, "registry": rec.n_frames, "timestamps": len(ts.raw)}))

    data = old.extract_data(est, idx)
    meta = old.build_metadata(est, rec.fps, sid, False, None, ml_info, idx)
    meta["fps"] = rec.fps
    meta["autoinspection_fps"] = rec.stages["autoinspect"].metrics.get("autoinspection_fps")
    meta["provenance"] = provenance.collect(paths, rec, cfg)
    meta["timestamps_file"] = paths.export_timestamps(sid).name

    targets = {"npz": paths.export_npz(sid), "json": paths.export_json(sid),
               "timestamps": paths.export_timestamps(sid), "filters": paths.export_filters(sid)}
    for t in targets.values():
        t.parent.mkdir(parents=True, exist_ok=True)
    tmp = targets["npz"].parent / f".partial_{sid}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        tmp.mkdir(parents=True)
        npz, js = old.export(data, meta, sid, tmp)
        write_timestamps(ts, np.arange(n_ts), tmp / targets["timestamps"].name)
        mat = old.export_filters_mat(est, idx, tmp / targets["filters"].name)
        staged = {"timestamps": tmp / targets["timestamps"].name, "npz": npz, "json": js}
        if mat is not None:
            staged["filters"] = mat
        else:
            targets["filters"].unlink(missing_ok=True)
            anomalies.append(Anomaly("no_filters", "warning",
                                     "estimates have no A matrix or imax; no filters file written"))
        # json goes last: it marks the export as complete
        try:
            for k in ("timestamps", "filters", "npz", "json"):
                if k in staged:
                    os.replace(staged[k], targets[k])
        except BaseException:
            targets["json"].unlink(missing_ok=True)
            raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    snr = getattr(est, "SNR_comp", None)
    median_snr = float(np.median(np.asarray(snr)[idx])) if snr is not None and len(idx) else None
    return StageResult(
        outputs={k: paths.rel(v) for k, v in targets.items() if k in staged},
        metrics={"n_neurons": int(len(idx)), "median_snr": median_snr},
        anomalies=anomalies)
