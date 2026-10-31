"""Equivalence of the batch pipeline with the old code (Task 18): MC, CNMF-E, ML probabilities, export schema.

Data:
- BEARMIND_TEST_SERIES: finished series (as for test_e2e_real.py). The checks use one short session (at most
  MAX_CLIP_FRAMES frames, all stages done; BEARMIND_EQUIV_SESSION picks one by sid). CNMF-E is never run
  on a full session here.
- BEARMIND_EQUIV_ML: folders or files (os.pathsep) with *_processed.pickle files made by the old code.
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
