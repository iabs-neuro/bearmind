"""
Per-component values must stay with their component when compression removes
bad components.

Runs on a real CaImAn session from data/ (gitignored, so the test skips on a
machine without it). Lengths alone cannot catch misalignment: after taking the
first N entries instead of the good ones, every array still has N entries.
"""
import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from estimates_compression import compress_estimates_ultra_lightweight

SESSION = Path(__file__).parent.parent / 'data' / 'WAW_H02_1D30-03-2026 19-27-57_estimates.pickle'

PER_COMPONENT = ['bl', 'c1', 'neurons_sn', 'SNR_comp', 'r_values', 'cnn_preds', 'lam', 'g']


@pytest.fixture(scope='module')
def original():
    if not SESSION.exists():
        pytest.skip(f'real session not available: {SESSION.name}')
    with open(SESSION, 'rb') as f:
        est = pickle.load(f)
    est = getattr(est, 'estimates', est)
    good = np.asarray(est.idx_components)
    # A session whose good components are exactly 0..N-1 cannot tell position from ID
    assert np.any(good != np.arange(len(good))), 'session does not exercise reindexing'
    return est


@pytest.fixture(scope='module')
def compressed(original):
    est, _, _ = compress_estimates_ultra_lightweight(copy.deepcopy(original), remove_bad_components=True)
    return est


@pytest.mark.parametrize('attr', PER_COMPONENT)
def test_values_follow_good_components(original, compressed, attr):
    full = getattr(original, attr, None)
    if not isinstance(full, (np.ndarray, list)) or len(full) != original.C.shape[0]:
        pytest.skip(f'{attr} is not a per-component array in this session')
    good = np.asarray(original.idx_components)
    got = getattr(compressed, attr)
    assert len(got) == len(good), f'{attr}: {len(got)} entries for {len(good)} components'
    for new_pos, old_id in enumerate(good):
        # Float32 conversion during compression is allowed; a value from another component is not
        np.testing.assert_allclose(np.asarray(got[new_pos], dtype=float),
                                   np.asarray(full[old_id], dtype=float),
                                   rtol=1e-6, err_msg=f'{attr} of component {old_id}')
