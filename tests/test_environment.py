from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_versions():
    import caiman, numpy
    assert caiman.__version__.startswith("1.13.2")
    assert int(numpy.__version__.split(".")[0]) >= 2


def test_linalg_does_not_crash():
    import numpy as np, scipy.linalg as sl
    a = np.random.default_rng(0).random((80, 80))
    assert sl.inv(a).shape == (80, 80)
    assert np.linalg.lstsq(a, a[:, 0], rcond=None)[0].shape == (80,)


def test_openblas_coretype_applied():
    """The root conftest sets OPENBLAS_CORETYPE before numpy loads (the Zen 4 workaround, and the kernels the
    golden record was made with). A pytest plugin that imports numpy before any conftest (zarr's) undoes it
    silently outside an activated env; pytest.ini blocks that plugin."""
    import os
    import pytest
    from threadpoolctl import threadpool_info
    blas = [d for d in threadpool_info() if d.get("internal_api") == "openblas"]
    if not blas:
        pytest.skip("numpy does not use OpenBLAS here")
    want = os.environ["OPENBLAS_CORETYPE"].lower()
    assert [d["architecture"].lower() for d in blas] == [want] * len(blas)


def test_ffv1_codec_available():
    import av
    assert "ffv1" in av.codecs_available and "libx264" in av.codecs_available


def test_production_model_loads():
    import joblib, numpy as np
    m = joblib.load(REPO / "production_models" / "ebm_v9_iter8.pkl")
    assert m.predict_proba(np.zeros((2, len(m.feature_names_in_)))).shape == (2, 2)
