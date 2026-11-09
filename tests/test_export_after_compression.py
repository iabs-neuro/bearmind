"""
Every export output of a compressed processed file describes the same components, one row each.

Autoinspection compresses the processed file: the bad components leave C, A, metrics_df, asp_cache and
reconstructions, and the rest are renumbered 0..n-1. The export then picks components with
export_estimates_data.approved_components and writes C, ASP and reconstructions (npz), metrics_df (json) and the
filters (mat).
If one structure keeps the old numbers, the files stop describing the same neurons: rows are missing or
belong to another neuron, and a missing asp_cache or reconstructions entry becomes an all-zero row.

A synthetic session, so the check runs on every machine (the real-session tests skip without output/).
It is exported the way the export stage does (tests/export_helpers.export_processed).
"""
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse
from scipy.io import loadmat

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import export_estimates_data as old
from estimates_compression import compress_estimates_ultra_lightweight
from export_helpers import export_processed

SESSION = 'SYN_S01_1D'
N_FRAMES = 300
D1, D2 = 40, 30  # imax shape
# component -> (ml_keep_probability, decision, delete); the threshold is 0.72
COMPONENTS = {
    0: (0.95, 'delete', 1),       # bad: compression removes it
    1: (0.20, 'delete', 1),       # bad
    2: (0.85, 'delete', 1),       # bad
    3: (0.95, 'keep', 0),         # exported
    4: (0.10, 'keep', 0),         # below the threshold
    5: (np.nan, 'from_merge', 0),  # merge product: no probability, exported
    6: (0.90, 'keep', 1),         # SNR duplicate left in idx_components for the GUI: not exported
    7: (0.99, 'delete', 1),       # bad
    8: (0.80, 'keep', 0),         # exported
    9: (0.72, 'keep', 0),         # exactly the threshold: exported
    10: (0.50, 'keep', 0),        # below the threshold
    11: (0.99, 'keep', 0),        # exported
}
BAD = [0, 1, 2, 7]
GOOD = [i for i in COMPONENTS if i not in BAD]
EXPORTED = [3, 5, 8, 9, 11]  # original numbers, in export order
# blob centre (row, col) of each component's footprint, all distinct and away from the border
CENTRES = {i: (6 + 3 * (i % 4) * 2, 6 + (i // 4) * 7) for i in COMPONENTS}


def _footprint(centre):
    r, c = np.mgrid[0:D1, 0:D2]
    blob = np.exp(-((r - centre[0]) ** 2 + (c - centre[1]) ** 2) / (2 * 2.0 ** 2))
    blob[blob < 1e-3] = 0
    # CaImAn stores a footprint as a column of A in Fortran order
    return blob.flatten(order='F')


def _synthetic_estimates():
    rng = np.random.default_rng(0)
    n = len(COMPONENTS)
    est = SimpleNamespace()
    est.name = f'{SESSION}01-01-2026 00-00-00_estimates.pickle'
    est.idx_components = np.array(GOOD)
    est.idx_components_bad = np.array(BAD)
    # each trace is offset by its component number, so a row can be traced back to its component
    est.C = rng.random((n, N_FRAMES)) + np.arange(n)[:, None]
    est.S = rng.random((n, N_FRAMES))
    est.YrA = rng.random((n, N_FRAMES))
    est.A = scipy.sparse.csc_matrix(np.column_stack([_footprint(CENTRES[i]) for i in range(n)]))
    est.imax = rng.random((D1, D2))
    est.metrics_df = pd.DataFrame({
        'component_idx': list(COMPONENTS),
        'ml_keep_probability': [v[0] for v in COMPONENTS.values()],
        'decision': [v[1] for v in COMPONENTS.values()],
        'delete': [v[2] for v in COMPONENTS.values()],
    })
    # autoinspection fills them for the kept components only; positive, so a zero row means a lost entry
    est.asp_cache = {i: rng.random(N_FRAMES) + 0.1 for i in GOOD}
    est.reconstructions = {i: rng.random(N_FRAMES) + 0.1 for i in GOOD}
    est.autoinspection_config = {'ml_threshold': 0.72, 'merge_grouping': old.MERGE_GROUPING}
    return est


@pytest.fixture(scope='module')
def original():
    est = _synthetic_estimates()
    return {'C': est.C.astype(np.float32), 'asp': {i: a.astype(np.float32) for i, a in est.asp_cache.items()},
            'recon': {i: r.astype(np.float32) for i, r in est.reconstructions.items()}}


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('export_after_compression')
    est, _, _ = compress_estimates_ultra_lightweight(_synthetic_estimates(), remove_bad_components=True)
    processed = tmp / f'{SESSION}_processed.pickle'
    with open(processed, 'wb') as f:
        pickle.dump(est, f)
    out = tmp / 'out'
    with pytest.MonkeyPatch.context() as mp:
        # the frame rate is not checked here, and the synthetic session has no row in fps_data.csv
        mp.setattr(old, 'get_fps', lambda name: 20.0)
        name = export_processed(processed, out)
    npz = np.load(out / f'{name}_data.npz')
    with open(out / f'{name}_metadata.json') as f:
        meta = json.load(f)
    mat = loadmat(out / f'{name}_filters.mat')
    return {'indices': npz['component_indices'], 'C': npz['C'], 'asp': npz['asp'], 'recon': npz['reconstructions'],
            'meta_indices': meta['metrics_df']['component_idx'],
            'meta_prob': meta['metrics_df']['ml_keep_probability'], 'A': mat['A']}


def test_every_output_has_one_row_per_exported_component(exported):
    n = len(exported['indices'])
    assert n == len(EXPORTED)
    assert exported['C'].shape[0] == n
    assert exported['asp'].shape[0] == n
    assert exported['recon'].shape[0] == n
    assert len(exported['meta_indices']) == n
    assert exported['A'].shape[0] == n


def test_no_zero_asp_rows(exported):
    zero = np.flatnonzero(np.all(exported['asp'] == 0, axis=1))
    assert not len(zero), f'all-zero ASP rows (lost asp_cache entries): {zero.tolist()}'
    zero = np.flatnonzero(np.all(exported['recon'] == 0, axis=1))
    assert not len(zero), f'all-zero reconstruction rows (lost reconstructions entries): {zero.tolist()}'


def test_rows_belong_to_the_approved_components(exported, original):
    # compression renumbers the kept components 0..n-1 in their old order
    new_number = {old_i: new_i for new_i, old_i in enumerate(GOOD)}
    assert exported['indices'].tolist() == [new_number[i] for i in EXPORTED]
    assert exported['meta_indices'] == exported['indices'].tolist()
    for row, i in enumerate(EXPORTED):
        np.testing.assert_array_equal(exported['C'][row], original['C'][i])
        np.testing.assert_array_equal(exported['asp'][row], original['asp'][i])
        np.testing.assert_array_equal(exported['recon'][row], original['recon'][i])
        prob = COMPONENTS[i][0]
        assert exported['meta_prob'][row] == (None if np.isnan(prob) else prob)
        # export_filters_mat writes each filter as (width, height): A's Fortran-ordered column in C order
        peak = np.unravel_index(np.argmax(exported['A'][row]), exported['A'][row].shape)
        assert peak == CENTRES[i][::-1], f'filter row {row} is not component {i}'
