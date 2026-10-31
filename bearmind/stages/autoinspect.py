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


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    params = dict(cfg["autoinspect"])
    model = _REPO_ROOT / params.pop("model_path")
    out_dir = paths.inspection_dir(sid)
    # Integer fps on purpose: the ML model features and the paper data were built with integer fps.
    fps = round(rec.fps)
    processed = out_dir / f"{sid}_processed.pickle"
    # a pickle from an earlier run would hide that this run saved nothing
    processed.unlink(missing_ok=True)
    result = ae_launch.run_auto_inspection(
        str(gate.input_estimates(paths, rec)), fps=fps, session_name=sid,
        artifacts_dir=str(out_dir), ml_model_path=str(model), **params)
    s = result["summary"]
    n0, n1 = s["n_initial"], s["n_final"]
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
    return StageResult(
        outputs={"processed": paths.rel(processed)},
        metrics={"n_initial": n0, "n_final": n1,
                 "deleted_frac": 1 - n1 / n0 if n0 > 0 else 0.0,
                 "autoinspection_fps": fps},
        anomalies=anomalies)
