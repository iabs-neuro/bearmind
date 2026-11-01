"""Equivalence of the batch pipeline with the old code (Task 18): MC, CNMF-E, ML probabilities, export schema.

Data:
- BEARMIND_TEST_SERIES: finished series (as for test_e2e_real.py). The checks use one short session (at most
  MAX_CLIP_FRAMES frames, all stages done; BEARMIND_EQUIV_SESSION picks one by sid). CNMF-E is never run
  on a full session here.
- BEARMIND_EQUIV_ML: folders or files (os.pathsep) with *_processed.pickle files made by the old code.
- BEARMIND_EQUIV_AI: one *_processed.pickle of the old env (CaImAn 1.11.3 / NumPy 1.26), and
  BEARMIND_EQUIV_AI_INPUT: the estimates it was made from. Autoinspection is run again on that input here
  (read only; artifacts go to the cache) and its metrics are compared with the old ones row by row.
Without them the tests are skipped.

The old MC run and both CNMF-E runs take minutes; equivalence_runs.py runs them in fresh processes once and
caches the results (BEARMIND_EQUIV_CACHE, default <system temp>/bearmind_equiv). The comparisons themselves
run on every invocation.
"""
import json
import os
import pickle
import shutil
import sys
from pathlib import Path, PureWindowsPath

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import equivalence_runs as er  # noqa: E402

from bearmind import config  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import State, all_records  # noqa: E402

SERIES = [Path(p) for p in os.environ.get("BEARMIND_TEST_SERIES", "").split(os.pathsep) if p.strip()]
ML_ROOTS = [Path(p) for p in os.environ.get("BEARMIND_EQUIV_ML", "").split(os.pathsep) if p.strip()]
AI_OLD = os.environ.get("BEARMIND_EQUIV_AI") or None
AI_INPUT = os.environ.get("BEARMIND_EQUIV_AI_INPUT") or None
WANT_SID = os.environ.get("BEARMIND_EQUIV_SESSION") or None
MAX_CLIP_FRAMES = 3000  # author's rule: CNMF-E equivalence on a short clip only
STAGES_DONE = ("crop", "mc", "cnmf", "autoinspect", "export")
MIN_MATCHED = 0.98

pytestmark = [pytest.mark.realdata, pytest.mark.slow]


@pytest.fixture(scope="module")
def clip():
    if not SERIES:
        pytest.skip("BEARMIND_TEST_SERIES is not set")
    for root in SERIES:
        paths = SeriesPaths(root.resolve())
        if not paths.registry_file("x").parent.is_dir():
            continue
        for r in sorted(all_records(paths), key=lambda r: r.sid):
            if r.excluded or (WANT_SID and r.sid != WANT_SID) or r.n_frames > MAX_CLIP_FRAMES:
                continue
            if all(r.stages.get(s) is not None and r.stages[s].state is State.DONE for s in STAGES_DONE):
                return paths, r
    pytest.skip(f"no finished session of <= {MAX_CLIP_FRAMES} frames in BEARMIND_TEST_SERIES")


def test_mc_matches_old_domotioncorrection(clip):
    paths, rec = clip
    cfg = config.resolve(paths, sid=rec.sid)
    st = rec.stages["mc"]
    assert st.params.get("mc") == cfg["mc"], "the series MC output was made with other parameters; rerun mc"
    out = er.ensure_mc(paths.root, rec.sid)
    meta = er.read_meta(out)
    assert meta["crop"] == st.metrics["crop"]

    old = np.load(out / "old_shifts.npy")
    new = np.load(paths.root / st.outputs["mc_shifts"])
    assert old.shape == new.shape == (rec.n_frames, 2)
    print(f"\nMC shifts max |old - new| = {np.max(np.abs(old - new)):.3g} px over {len(old)} frames")
    assert np.allclose(old, new, atol=1e-6)

    hist = er.frame_diff_hist(out / "old_MC.tif", paths.root / st.outputs["mc_video"])
    print(f"MC frames, new - old grey levels: {hist}")
    # the stage rounds, DoMotionCorrection truncates the float frames to uint8
    assert max(abs(k) for k in hist) <= 1


def test_cnmf_matches_old_docnmf(clip):
    paths, rec = clip
    n = max(1, (os.cpu_count() or 2) - 1)  # what both would pick on this machine; pinned on both sides
    pool = er.stage_pool(paths.root, rec.sid, n)
    if pool != n:
        pytest.skip(f"the RAM budget gives the CNMF stage a pool of {pool}, not {n}: no equal-pool comparison")
    outs = er.ensure_cnmf(paths.root, rec.sid, n)
    meta = {side: er.read_meta(out) for side, out in outs.items()}
    assert meta["old"]["pool"] == meta["new"]["pool"] == [n]
    assert meta["old"]["blas"] == meta["new"]["blas"] == er.BLAS_ENV
    assert meta["old"]["params"] == meta["new"]["params"]
    assert meta["old"]["params"]["fr"] == rec.fps  # exact session FPS, never fps / 2
    assert meta["old"]["memmap_identical"], "DoCNMF and the stage did not get the same CNMF-E input"

    m = er.match_components(er.load_ac(outs["old"] / "old.npz"), er.load_ac(outs["new"] / "new.npz"))
    print("\nCNMF-E old vs new: " + json.dumps({k: v for k, v in m.items() if not k.startswith("unmatched")}))
    assert m["frac_old"] >= MIN_MATCHED and m["frac_new"] >= MIN_MATCHED, m


def _processed_pickles(roots):
    files = set()
    for r in roots:
        if r.is_file():
            files.add(r)
        elif r.is_dir():
            files.update(r.rglob("*_processed.pickle"))
    return sorted(files)


@pytest.mark.skipif(not ML_ROOTS, reason="BEARMIND_EQUIV_ML is not set")
def test_ml_probabilities_match_old_env():
    import auto_inspector

    model = er.REPO / config.load_defaults()["autoinspect"]["model_path"]
    worst, n_files, n_rows, other_model = 0.0, 0, 0, []
    bad = {}
    for f in _processed_pickles(ML_ROOTS):
        with open(f, "rb") as fh:  # read only; nothing is written next to source data
            est = pickle.load(fh)
        df = getattr(est, "metrics_df", None)
        if df is None or "ml_keep_probability" not in df.columns:
            continue
        used = (getattr(est, "autoinspection_config", None) or {}).get("ml_model_path")
        if not used or PureWindowsPath(str(used)).name != model.name:
            other_model.append(f.name)
            continue
        scored = df[df["ml_keep_probability"].notna()]
        if scored.empty:
            continue
        _, info = auto_inspector._apply_ml_brain(scored, model)
        diff = np.abs(np.asarray(info["ml_keep_probability"]) - scored["ml_keep_probability"].to_numpy(float))
        worst = max(worst, float(diff.max()))
        n_files += 1
        n_rows += len(scored)
        if not np.allclose(info["ml_keep_probability"], scored["ml_keep_probability"], atol=1e-6):
            bad[f.name] = float(diff.max())
    if not n_files:
        pytest.skip(f"no processed pickle scored with {model.name} under BEARMIND_EQUIV_ML")
    print(f"\nML: {n_files} files, {n_rows} scored rows, max |diff| = {worst:.3g}; "
          f"{len(other_model)} files of another model skipped")
    assert not bad, bad


# Model features that do not reproduce across the environments, for causes measured in Task 19 on
# RFC_F01_1D (.superpowers/sdd/2026-10-31-bearmind-batch-core/task-19-report.md):
# - contour shape: CaImAn 1.13 get_contours gives pixels of equal value one cumulative energy
#   (caiman.base.rois.norm_nrg); 1.11.3 broke such ties by sort order, so the contour at cthr moves. On
#   the same input, 1.11.3 contours give the old areas exactly and 1.13.2 contours the new ones.
CONTOUR_FEATURES = ("area", "circularity", "aspect_ratio", "convexity", "max_edge", "footprint_compactness")
# - events and the reconstruction from them: driada's Neuron adds unseeded noise of 1e-8 to every trace
#   (calcium_preprocessing(seed=None)); two runs in one environment differ as much as old vs new, and are
#   bit-identical when that noise is seeded. With the noise seeded the old and the new environment still
#   differ (events_per_min in 14 of 156 components): event detection is sensitive to the last digits.
EVENT_FEATURES = ("events_per_min", "events_fraction", "t_rise", "t_off", "event_snr", "peak_amplitude_cv",
                  "event_r2_score", "r2_score", "nrmse", "nmae", "snr_recon")


@pytest.mark.skipif(not (AI_OLD and AI_INPUT), reason="BEARMIND_EQUIV_AI / BEARMIND_EQUIV_AI_INPUT not set")
def test_autoinspection_features_against_old_env():
    """Autoinspection recomputed in this env from the input of an old-env run, row by row against the old run.

    Every model feature outside CONTOUR_FEATURES and EVENT_FEATURES must be equal (rtol 1e-6, atol 1e-9).
    Those two groups, the probabilities and the decisions are printed, not asserted (see above). Rows are
    the components before merging (the old run's <session>_metrics_with_decisions.csv); without that file,
    the processed metrics_df, whose rows correspond only when the merges are the same.
    """
    import pandas as pd

    old_file, inp = Path(AI_OLD), Path(AI_INPUT)
    out = er.ensure_autoinspect(inp, old_file)
    meta = er.read_meta(out)
    old_est = er.load_pickle(old_file)
    model = er.load_pickle(er.model_file(old_est.autoinspection_config["ml_model_path"]))
    features = list(model.feature_names_in_)
    csv = sorted(old_file.parent.glob("*_metrics_with_decisions.csv"))
    if len(csv) == 1:
        level, old, new = "before merging", pd.read_csv(csv[0]), pd.read_pickle(out / "decision_df.pkl")
    else:
        level, old, new = "processed", old_est.metrics_df, pd.read_pickle(out / "processed_df.pkl")
    r = er.compare_metrics(old, new, features)
    print(f"\nautoinspection in this env: {meta['wall_s']:.0f} s, peak RSS {meta['peak_rss_gb']:.1f} GB, "
          f"summary {meta['summary']}")
    print(f"{level}: rows {r['rows_old']} / {r['rows_new']}, same rows {r['same_rows']}, missing {r['missing']}")
    assert r["same_rows"] and not r["missing"], r
    for name, group in (("contour", CONTOUR_FEATURES), ("events", EVENT_FEATURES),
                        ("other", [x for x in features if x not in CONTOUR_FEATURES + EVENT_FEATURES])):
        print(f"  {name}: " + ", ".join(f"{k} {r['features'][k]['max_abs_diff']:.3g}" for k in group))
    print(f"  ml_keep_probability {r['ml_keep_probability']}; rows that differ in delete "
          f"{r.get('delete_n_differ')}, merge {r.get('merge_n_differ')}, decision {r.get('decision_n_differ')}")
    unexplained = {k: v for k, v in r["features"].items()
                   if k not in CONTOUR_FEATURES + EVENT_FEATURES and not v["allclose"]}
    assert not unexplained, unexplained


def test_export_schema_matches_old_export(clip, tmp_path):
    import scipy.io

    import batch_export

    paths, rec = clip
    src = paths.root / rec.stages["autoinspect"].outputs["processed"]
    copy = tmp_path / src.name  # export_one writes next to its input: keep the series untouched
    shutil.copyfile(src, copy)
    batch_export.export_one(copy)

    new = {k: paths.root / v for k, v in rec.stages["export"].outputs.items()}
    old_npz, old_json = tmp_path / f"{rec.sid}_data.npz", tmp_path / f"{rec.sid}_metadata.json"
    old_mat = tmp_path / f"{rec.sid}_filters.mat"
    with np.load(old_npz, allow_pickle=True) as a, np.load(new["npz"], allow_pickle=True) as b:
        assert sorted(a.files) == sorted(b.files)
    assert ({n for n, _, _ in scipy.io.whosmat(str(old_mat))}
            == {n for n, _, _ in scipy.io.whosmat(str(new["filters"]))})

    def missing(o, n, where=""):
        out = []
        for k, v in o.items():
            if k not in n:
                out.append(where + k)
            elif isinstance(v, dict) and isinstance(n[k], dict):
                out += missing(v, n[k], f"{where}{k}.")
        return out

    old_meta = json.loads(old_json.read_text(encoding="utf-8"))
    new_meta = json.loads(new["json"].read_text(encoding="utf-8"))
    assert not missing(old_meta, new_meta)
