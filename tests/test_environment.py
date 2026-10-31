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


def test_ffv1_codec_available():
    import av
    assert "ffv1" in av.codecs_available and "libx264" in av.codecs_available


def test_production_model_loads():
    import joblib, numpy as np
    m = joblib.load(REPO / "production_models" / "ebm_v9_iter8.pkl")
    assert m.predict_proba(np.zeros((2, len(m.feature_names_in_)))).shape == (2, 2)
