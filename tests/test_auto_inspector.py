"""auto_inspector footprint features under NumPy 2."""
import torch  # noqa: F401  (torch shm.dll must load before driada/caiman here)

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse

import auto_inspector


def test_eccentricity_is_real():
    """NumPy 2 np.linalg.eigvals returns complex128 even for real eigenvalues; the complex column crashed the EBM
    (Unsupported pandas dtype: complex128) on the first real sessions. Values must stay what NumPy 1 gave."""
    dims = (40, 50)
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    blob = np.exp(-((yy - 20) ** 2 / (2 * 1.0 ** 2) + (xx - 25) ** 2 / (2 * 2.0 ** 2)))  # sigma_x 2, sigma_y 1
    round_ = np.exp(-((yy - 20) ** 2 + (xx - 25) ** 2) / (2 * 1.5 ** 2))
    A = scipy.sparse.csc_matrix(np.stack([blob.ravel(order="F"), round_.ravel(order="F")], axis=1))
    ecc = auto_inspector.compute_eccentricity(A, dims)
    assert ecc.dtype == np.float64
    assert np.allclose(ecc, [2.0, 1.0], atol=1e-3)


def _trace(seed, n=2000):
    """Calcium-like: sparse spikes, exp decay 1 s at 20 fps, noise, scaled to 0..1."""
    rng = np.random.default_rng(seed)
    spikes = (rng.random(n) < 0.01).astype(float) * rng.uniform(0.5, 1.5, n)
    kernel = np.exp(-np.arange(40) / 20.0)
    ca = np.convolve(spikes, kernel)[:n] + rng.normal(0, 0.03, n)
    return (ca - ca.min()) / (ca.max() - ca.min())


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (np.isnan(a) and np.isnan(b))
    return a == b


def test_every_neuron_gets_the_seed(monkeypatch):
    seen = []

    class Recording(auto_inspector.Neuron):
        def __init__(self, *a, **k):
            seen.append(k.get("seed", "missing"))
            super().__init__(*a, **k)

    monkeypatch.setattr(auto_inspector, "Neuron", Recording)
    # threads, not loky processes: the patched class must be the one that runs
    monkeypatch.setattr(auto_inspector, "get_parallelization_backend", lambda: "threading")
    traces = np.stack([_trace(1), _trace(2), np.random.default_rng(3).random(2000)])
    auto_inspector.get_multineuron_metrics(traces, fps=20, include_heavy=True, event_method="wavelet",
                                           n_iter=3, hybrid_kinetics=True, seed=7)
    assert seen and set(seen) == {7}


def test_event_metrics_are_reproducible():
    kw = dict(fps=20, include_heavy=True, event_method="wavelet", n_iter=3, hybrid_kinetics=True, seed=0)
    a = auto_inspector.get_single_neuron_metrics(_trace(1), **kw)
    b = auto_inspector.get_single_neuron_metrics(_trace(1), **kw)
    assert a.keys() == b.keys()
    for k in a:
        assert _same(a[k], b[k]), k


def test_seed_defaults_to_zero():
    import inspect
    import ae_launch
    for f in (auto_inspector.get_neuron_with_spikes, auto_inspector.get_single_neuron_metrics,
              auto_inspector.get_multineuron_metrics, auto_inspector.estimates_to_metrics,
              auto_inspector.transform_metrics_df_indices, ae_launch.run_auto_inspection):
        assert inspect.signature(f).parameters["seed"].default == 0, f.__name__


def test_run_auto_inspection_forwards_the_seed(tmp_path, monkeypatch):
    import ae_launch

    class Stop(Exception):
        pass

    seen = {}

    def fake(est, fps, **kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(ae_launch, "estimates_to_metrics", fake)
    p = tmp_path / "NOF_H01_1D_estimates.pickle"
    p.write_bytes(pickle.dumps(SimpleNamespace(C=np.zeros((1, 10)), idx_components=np.array([0]))))
    with pytest.raises(Stop):
        ae_launch.run_auto_inspection(str(p), fps=20, brain="thresholds", seed=5)
    assert seen["seed"] == 5


SHORT = Path(r"H:\Projects\_bearmind_test\series_short\4_Estimates\NOF_H03_3D_estimates.pickle")


@pytest.mark.slow
@pytest.mark.realdata
@pytest.mark.skipif(not SHORT.exists(), reason="series_short not on this machine")
def test_autoinspection_twice_gives_identical_decisions():
    """Production parameters (bearmind defaults.yaml autoinspect), fps 30 = round(29.784); nothing saved."""
    import ae_launch
    from bearmind import config
    repo = Path(__file__).resolve().parent.parent
    params = dict(config.load_defaults()["autoinspect"])
    params.pop("model_path")
    kw = dict(params, ml_model_path=str(repo / "production_models" / "ebm_v9_iter8.pkl"), save_artifacts=False)
    r1 = ae_launch.run_auto_inspection(str(SHORT), fps=30, **kw)
    r2 = ae_launch.run_auto_inspection(str(SHORT), fps=30, **kw)
    pd.testing.assert_frame_equal(r1["estimates"].metrics_df, r2["estimates"].metrics_df)
    assert np.array_equal(r1["estimates"].idx_components, r2["estimates"].idx_components)
    assert r1["estimates"].autoinspection_config["seed"] == 0
