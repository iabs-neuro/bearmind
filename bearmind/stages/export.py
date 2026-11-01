"""Single export stage: npz + json + timestamps csv + filters mat, written atomically."""
from __future__ import annotations

import dataclasses
import os
import shutil

import numpy as np
from scipy.io import savemat

import export_estimates_data as old
from bearmind import checks, provenance
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


def rule_counts(est, idx, ml_info) -> dict:
    """Which rule chose the exported components and what it did with autoinspection's decisions.

    The rule is the old one (ml_keep_probability >= threshold; 'idx_components' when there are no ML
    probabilities). Autoinspection only marks components: an SNR duplicate keeps its P above the threshold
    and is exported although marked delete; a merge product has P = NaN and is never exported. The median
    SNR comes from metrics_df: after a merge, est.SNR_comp is not in the order of the components. None:
    not known (no metrics_df or no such column).
    """
    df = getattr(est, "metrics_df", None)
    exported = {int(i) for i in idx}
    out = {"component_rule": "ml_probability" if ml_info["ml_filtered"] else "idx_components",
           "n_exported": len(exported), "n_snr_duplicates_exported": None, "n_merge_products_dropped": None,
           "median_snr": None}
    if df is None:
        return out
    if df.empty:
        return {**out, "n_snr_duplicates_exported": 0, "n_merge_products_dropped": 0}
    if "component_idx" not in df.columns:  # rows cannot be matched to components (only without ML)
        return out
    rows = df[df["component_idx"].isin(exported)]
    if "delete" in df.columns:
        out["n_snr_duplicates_exported"] = int((rows["delete"] == 1).sum())
    if "decision" in df.columns:
        out["n_merge_products_dropped"] = int(((df["decision"] == "from_merge")
                                               & ~df["component_idx"].isin(exported)).sum())
    if "caiman_snr" in df.columns and len(rows):
        snr = rows["caiman_snr"].to_numpy(dtype=float)
        # all NaN: NaN, which the export_median_snr check reports as "not a number"
        out["median_snr"] = float(np.nanmedian(snr)) if np.isfinite(snr).any() else float("nan")
    return out


class _WithMetrics:
    """The estimates with another metrics_df; every other attribute is the estimates' own."""

    def __init__(self, est, df):
        self._est, self.metrics_df = est, df

    def __getattr__(self, name):
        return getattr(self._est, name)


def _metadata_input(est):
    """The estimates for old.build_metadata, which selects metrics_df rows by 'component_idx' and raises
    KeyError without that column: autoinspection leaves an empty metrics_df for a session without
    components. The column is added empty, so no row is listed for an exported neuron it cannot be matched
    to (the old function prints that warning to the session log)."""
    df = getattr(est, "metrics_df", None)
    if df is None or "component_idx" in df.columns:
        return est
    return _WithMetrics(est, df.assign(component_idx=np.nan))


def _empty_data(est, data) -> dict:
    """No neuron exported: the npz keeps the keys of an export with neurons, as arrays of length 0."""
    n_frames = int(est.C.shape[1])
    for key, attr in (("asp", "asp_cache"), ("reconstructions", "reconstructions")):
        if key not in data and hasattr(est, attr):  # the old extract_data adds them only for listed neurons
            data[key] = np.zeros((0, n_frames), np.float32)
    return data


def _filters(est, idx, path):
    """old.export_filters_mat; for no neuron an empty (0, h, w) uint8 stack, the layout of a non-empty one
    (the old function saves an empty float array there)."""
    if len(idx) or getattr(est, "A", None) is None or getattr(est, "imax", None) is None:
        return old.export_filters_mat(est, idx, path)
    savemat(path, {"A": np.zeros((0, *est.imax.shape[::-1]), np.uint8)})
    return path


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


def _provenance(paths, rec, cfg, metrics, anomalies) -> dict:
    """provenance.collect plus this stage's own check results and anomalies, which reach the registry only
    after the stage returns (the runner evaluates the same checks then)."""
    prov = provenance.collect(paths, rec, cfg)
    own = checks.evaluate("export", metrics, [metrics], cfg)
    if own:
        prov["checks"]["export"] = provenance.check_list(own)
    prov["anomalies"] += provenance.anomaly_list(
        [dataclasses.replace(a, data={**a.data, "stage": "export"}) for a in anomalies])
    return prov


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    est = old.load_estimates(paths.root / rec.stages["autoinspect"].outputs["processed"])
    idx, ml_info = _ml_filter(est)
    counts = rule_counts(est, idx, ml_info)
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

    if not len(idx):
        n = int(len(est.idx_components))
        why = (f"none of {n} components passes the export rule ({counts['component_rule']})" if n
               else "the estimates have no components")
        anomalies.append(Anomaly("no_components", "warning", f"no neurons exported: {why}",
                                 data={"n_components": n, "n_exported": 0}))

    data = old.extract_data(est, idx)
    if not len(idx):
        data = _empty_data(est, data)
    meta = old.build_metadata(_metadata_input(est), rec.fps, sid, False, None, ml_info, idx)
    meta["fps"] = rec.fps
    meta["autoinspection_fps"] = rec.stages["autoinspect"].metrics.get("autoinspection_fps")
    meta["timestamps_file"] = paths.export_timestamps(sid).name
    meta.update({k: v for k, v in counts.items() if k != "median_snr"})
    metrics = {"n_neurons": int(len(idx)), **counts}

    targets = {"npz": paths.export_npz(sid), "json": paths.export_json(sid),
               "timestamps": paths.export_timestamps(sid), "filters": paths.export_filters(sid)}
    for t in targets.values():
        t.parent.mkdir(parents=True, exist_ok=True)
    tmp = targets["npz"].parent / f".partial_{sid}"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        tmp.mkdir(parents=True)
        write_timestamps(ts, np.arange(n_ts), tmp / targets["timestamps"].name)
        mat = _filters(est, idx, tmp / targets["filters"].name)
        if mat is None:
            anomalies.append(Anomaly("no_filters", "warning",
                                     "estimates have no A matrix or imax; no filters file written"))
        # the json goes last into the staging folder: it carries this stage's own checks and anomalies too
        meta["provenance"] = _provenance(paths, rec, cfg, metrics, anomalies)
        npz, js = old.export(data, meta, sid, tmp)
        staged = {"timestamps": tmp / targets["timestamps"].name, "npz": npz, "json": js}
        if mat is not None:
            staged["filters"] = mat
        else:
            targets["filters"].unlink(missing_ok=True)
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

    return StageResult(
        outputs={k: paths.rel(v) for k, v in targets.items() if k in staged},
        metrics=metrics,
        anomalies=anomalies)
