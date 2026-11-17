"""The threshold suggestion lives in bearmind.seed_suggest (no CaImAn); bm_qc re-exports it unchanged."""
import os
import subprocess
import sys
from pathlib import Path

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
