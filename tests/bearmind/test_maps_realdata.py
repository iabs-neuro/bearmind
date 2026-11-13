"""Threshold maps recomputed on real LT1 sessions against the maps the code before the strip cut wrote.

The stored maps (H:\\Projects\\_bearmind_test\\lt1_J03_J06, read only) were made with 19.12 GB strips. Here the
budget is smaller (the batch may run on the same machine), so the strips differ, and so does the float32 rounding
of the spatial filter: OpenCV filters the CNMF-E kernel by DFT tiles sized from the strip. In the lit field this is
rounding (and the odd sample crossing the 2-sigma threshold of the correlation image). In the dark area outside
the motion-corrected field (zero in every frame, or in all but a few) the filtered data are rounding noise, and
PNR and correlation there are ratios of that noise: they change arbitrarily with the strips, as they did before
between machines with different RAM. The dark area is valid_mask's dark rule (mean under a quarter of the median
mean), without its dilation; the suggestion never looks there and must not change.

Outputs go to pytest's tmp_path: run with --basetemp in a scratch folder, and -s to see differences and timings.
"""
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from bearmind import config, registry
from bearmind.paths import SeriesPaths
from bearmind.stages import thresholds

ROOT = Path(os.environ.get("BEARMIND_MAPS_SERIES", r"H:\Projects\_bearmind_test\lt1_J03_J06"))
SIDS = ["LT1_J03_2D", "LT1_J06_2D"]
# 4 workers with 3.5 GB strips each
MACHINE = {"n_cpus": 5, "ram_gb": 35, "cnmf_ram_budget_gb": 16}

pytestmark = [pytest.mark.realdata, pytest.mark.slow,
              pytest.mark.skipif(not (ROOT / "3_MC").is_dir(), reason=f"{ROOT} not found")]


class _ScratchPaths(SeriesPaths):
    """Outputs under the scratch root; the MC movie read from the stored series."""

    def mc_video(self, sid):
        return ROOT / "3_MC" / f"{sid}_MC.avi"


def _diff(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = np.abs(a - b)
    rel = np.divide(d, np.abs(b), out=np.zeros_like(d), where=b != 0)
    return f"max_abs {d.max():.3g}  max_rel {rel.max():.3g}  differing {np.count_nonzero(d)}/{d.size}"


@pytest.mark.parametrize("sid", SIDS)
def test_maps_match_stored(sid, tmp_path):
    stored_paths = SeriesPaths(ROOT)
    rec = registry.load_record(stored_paths, sid)
    old = rec.stages["thresholds"]
    cfg = config.resolve(stored_paths, sid=sid)
    cfg["machine"] = {**MACHINE, "temp_dir": str(tmp_path / "tmp")}
    paths = _ScratchPaths(tmp_path / "series")

    t0 = datetime.now()
    res = thresholds.run(paths, rec, cfg)
    new_s = (datetime.now() - t0).total_seconds()
    old_s = (datetime.fromisoformat(old.finished) - datetime.fromisoformat(old.started)).total_seconds()

    assert res.metrics["suggested"] == old.metrics["suggested"]
    with np.load(stored_paths.threshold_maps(sid)) as ref, np.load(paths.threshold_maps(sid)) as new:
        assert sorted(new.files) == sorted(ref.files)
        for k in ("gsig", "mean", "max"):
            assert np.array_equal(new[k], ref[k]), k
        lit = ref["mean"] >= 0.25 * np.median(ref["mean"])
        print(f"\n{sid}: old {old_s:.0f} s ({old.metrics['map_workers']} workers, 19.12 GB strips), "
              f"new {new_s:.0f} s ({res.metrics['map_workers']} workers, "
              f"{thresholds.mem_budget_gb(cfg):.1f} GB strips); suggestion {res.metrics['suggested']}; "
              f"{np.count_nonzero(~lit)} of {lit.size} pixels dark, {np.count_nonzero(ref['max'] == 0)} zero in every "
              f"frame")
        for k in sorted(f for f in ref.files if f.startswith(("cn", "pnr"))):
            a, b = new[k], ref[k]
            assert a.dtype == np.float32 and a.shape == b.shape
            print(f"  {k:15s} lit: {_diff(a[lit], b[lit])}"
                  + (f" | dark: {_diff(a[~lit], b[~lit])}" if (~lit).any() else ""))
            np.testing.assert_allclose(a[lit], b[lit], rtol=1e-4, atol=1e-3, err_msg=k)
