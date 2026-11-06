"""Autoinspection stage: the unchanged ae_launch.run_auto_inspection with production parameters from config."""
import os
import pickle
from pathlib import Path

import ae_launch
import bearmind
from bearmind.registry import Anomaly
from bearmind.stages import StageResult, gate

# model_path in config is relative to the repo root, where production_models/ lives.
_REPO_ROOT = Path(bearmind.__file__).resolve().parent.parent


def check_aligned(path: Path) -> None:
    """Refuse estimates whose per-component arrays are not in the order of A's columns.

    The model features caiman_snr and caiman_r_score are read by position from est.SNR_comp and
    est.r_values (auto_inspector.estimates_to_metrics). CaImAn's manual_merge does not update them, and
    estimates_compression kept arr[:N] instead of the arrays of the kept components; on such input every
    component would be scored with another one's values. Both are fixed since main's 6c73765 / fbd1ad3
    (2026-10-31, 11-01), but a file compressed before that cannot be told apart from a newer one (both carry
    idx_components_bad_original), so every compressed input is still refused.
    """
    with open(path, "rb") as f:
        est = pickle.load(f)
    n = est.A.shape[1]
    problems = []
    if hasattr(est, "idx_components_bad_original"):
        problems.append("compressed by estimates_compression (has idx_components_bad_original)")
    for attr in ("SNR_comp", "r_values"):
        values = getattr(est, attr, None)
        if values is None:
            problems.append(f"no {attr}")
        elif len(values) != n:
            problems.append(f"len({attr}) = {len(values)} but A has {n} components")
    if problems:
        raise ValueError(f"{path}: per-component arrays not aligned with the components, autoinspection "
                         f"refused: {'; '.join(problems)}")


def _count(df, column, value) -> int:
    if df is None or column not in df.columns:
        return 0
    return int((df[column] == value).sum())


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    params = dict(cfg["autoinspect"])
    model = _REPO_ROOT / params.pop("model_path")
    out_dir = paths.inspection_dir(sid)
    # Integer fps on purpose: the ML model features and the paper data were built with integer fps.
    fps = round(rec.fps)
    processed = out_dir / f"{sid}_processed.pickle"
    source = gate.input_estimates(paths, rec)
    check_aligned(source)  # its copy of the estimates is freed on return, before the file is loaded again
    # a pickle from an earlier run would hide that this run saved nothing
    processed.unlink(missing_ok=True)
    result = ae_launch.run_auto_inspection(
        str(source), fps=fps, session_name=sid,
        artifacts_dir=str(out_dir), ml_model_path=str(model), **params)
    anomalies = []
    if not processed.exists():
        # run_auto_inspection returns early without saving when there are no components
        if result.get("estimates") is None:
            raise FileNotFoundError(f"autoinspection produced no processed estimates: {processed}")
        out_dir.mkdir(parents=True, exist_ok=True)
        part = processed.with_name(processed.name + ".partial")
        try:
            with open(part, "wb") as f:
                pickle.dump(result["estimates"], f)  # as auto_inspector.save_processed_estimates
            os.replace(part, processed)
        finally:
            part.unlink(missing_ok=True)
        anomalies.append(Anomaly("no_components", "warning",
                                 "no components to inspect; estimates saved without autoinspection"))
    # Autoinspection marks components delete and keeps them in the estimates (the export rule decides
    # what leaves); merges replace their sources by one 'from_merge' row each.
    s = result["summary"]
    df = result["estimates"].metrics_df
    n0, n_deleted = int(s["n_initial"]), int(s["n_deleted"])
    return StageResult(
        outputs={"processed": paths.rel(processed)},
        metrics={"n_initial": n0,
                 "n_after_merge": int(s["n_final"]),
                 "n_deleted": n_deleted,
                 "n_merge_groups": _count(df, "decision", "from_merge"),
                 "n_kept": _count(df, "delete", 0),
                 "deleted_frac": n_deleted / n0 if n0 > 0 else 0.0,
                 "autoinspection_fps": fps},
        anomalies=anomalies)
