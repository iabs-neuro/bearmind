"""
Automatic merging must keep per-component quality values with their component.

CaImAn's Estimates.manual_merge reorders A, C, YrA, S, bl, c1, neurons_sn and g
into "untouched components, then merged ones", but leaves SNR_comp, r_values,
cnn_preds and lam in the old order and length. Every component after the first
merged source then reads its SNR and r_value from another component.

Runs on a real CaImAn session from data/ (gitignored, so the test skips on a
machine without it).
"""
# auto_inspector imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_inspector import implement_decision

SESSION = Path(__file__).parent.parent / 'data' / 'WAW_H02_1D30-03-2026 19-27-57_estimates.pickle'

QUALITY = ['SNR_comp', 'r_values']


@pytest.fixture(scope='module')
def original():
    if not SESSION.exists():
        pytest.skip(f'real session not available: {SESSION.name}')
    with open(SESSION, 'rb') as f:
        est = pickle.load(f)
    return getattr(est, 'estimates', est)


@pytest.fixture(scope='module')
def merged(original):
    good = [int(i) for i in original.idx_components]
    # Two accepted components early in the list, so that most components move
    group = [good[3], good[10]]
    df = pd.DataFrame({'component_idx': np.arange(original.C.shape[0]),
                       'decision': 'ok', 'merge': 0})
    df.loc[df['component_idx'].isin(group), 'merge'] = 1
    est, mapping = implement_decision(original, df, return_index_mapping=True)
    return est, mapping, group


@pytest.mark.parametrize('attr', QUALITY)
def test_length_matches_components(merged, attr):
    est, _, _ = merged
    assert len(getattr(est, attr)) == est.C.shape[0]


@pytest.mark.parametrize('attr', QUALITY)
def test_untouched_components_keep_their_values(original, merged, attr):
    est, mapping, _ = merged
    old = np.asarray(getattr(original, attr), dtype=float)
    new = np.asarray(getattr(est, attr), dtype=float)
    wrong = [o for o, n in mapping['old_to_new'].items()
             if not np.isclose(new[n], old[o], equal_nan=True)]
    assert not wrong, f'{attr}: {len(wrong)} components carry another value, e.g. {wrong[:5]}'


@pytest.mark.parametrize('attr', QUALITY)
def test_merged_component_gets_mean_of_sources(original, merged, attr):
    est, mapping, group = merged
    old = np.asarray(getattr(original, attr), dtype=float)[group]
    new_idx = mapping['merged_groups'][0]['new_idx']
    expected = np.mean(old[np.isfinite(old)])
    assert np.isclose(getattr(est, attr)[new_idx], expected)
