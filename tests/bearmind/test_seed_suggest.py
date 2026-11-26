"""The threshold suggestion lives in bearmind.seed_suggest (no CaImAn); bm_qc re-exports it unchanged."""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bearmind import seed_suggest
from bearmind.seed_suggest import _robust, halo_mask

REPO = Path(__file__).resolve().parents[2]
NAMES = ("count_seeds", "_robust", "halo_mask", "threshold_seeds", "evaluate_thresholds", "suggest_init_thresholds")


def test_bm_qc_reexports_the_suggestion():
    import bm_qc
    from bearmind import seed_suggest
    for name in NAMES:
        assert getattr(bm_qc, name) is getattr(seed_suggest, name), name


def test_suggestion_module_loads_no_caiman():
    code = "import sys, bearmind.seed_suggest; print('caiman' in sys.modules)"
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"


def test_broad_correlations_still_mask_a_strong_transient():
    shape = (40, 40)
    cn = np.random.default_rng(151).uniform(.65, .95, shape)
    pnr = np.full(shape, 5.)
    cn[17, 21], pnr[17, 21] = .995, 40.
    valid = np.ones(shape, dtype=bool)
    med, spread = _robust(cn[valid])
    assert med + 3 * spread > 1

    halo = halo_mask({"cn": cn, "pnr": pnr}, valid, excl_radius=2)

    assert halo[17, 21] and halo[17, 22]
    assert not halo[17, 26] and not halo[4, 4]


def test_background_uses_the_mirrored_maps_outside_halos(monkeypatch):
    shape = (24, 24)
    halo = np.zeros(shape, dtype=bool)
    halo[:18, :] = True
    maps = dict(gSig=4, cn=np.full(shape, .94), pnr=np.full(shape, 60.),
                cn_neg=np.where(halo, .98, .82), pnr_neg=np.where(halo, 50., 13.))
    monkeypatch.setattr(seed_suggest, "halo_mask", lambda *_: halo)

    options = dict(border_pix=0, corr_grid=[.9], pnr_grid=[15.])
    suggestions, _ = seed_suggest.suggest_init_thresholds(maps, **options)
    assert suggestions["robust_bg"]["min_corr"] == .82
    assert suggestions["robust_bg"]["min_pnr"] == 13.

    halo[:] = True
    suggestions, grid = seed_suggest.suggest_init_thresholds(maps, **options)
    assert suggestions["robust_bg"]["min_corr"] == .94
    assert suggestions["robust_bg"]["min_pnr"] == 60.
    assert "fdr" not in suggestions["robust_bg"]
    assert suggestions["mirror_fdr"]["min_corr"] is None and grid is None

    del maps["cn_neg"], maps["pnr_neg"]
    suggestions, _ = seed_suggest.suggest_init_thresholds(maps, **options)
    assert suggestions["robust_bg"]["min_corr"] == .94
    assert suggestions["robust_bg"]["min_pnr"] == 60.


def test_unusable_maps_do_not_return_thresholds():
    cn = np.full((24, 24), np.nan)
    with pytest.raises(ValueError, match="no finite pixels"):
        seed_suggest.suggest_init_thresholds(dict(gSig=4, cn=cn, pnr=np.ones_like(cn)))
