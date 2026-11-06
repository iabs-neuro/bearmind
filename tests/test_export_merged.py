"""
Export must keep components produced by automatic merging.

Merged components are never scored by the ML model (their ml_keep_probability is
NaN), so a plain `probability >= threshold` filter silently drops every one of
them from the NPZ, the metadata JSON and the filters .mat used for matching.

Runs the real export script on a real autoinspected session from output/
(gitignored, so the test skips on a machine without it).
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from export_estimates_data import DEFAULT_ML_THRESHOLD, extract_session_name

SESSION = (REPO / 'output' / 'inspection_artifacts_WAW_H02_1D'
           / 'WAW_H02_1D30-03-2026 19-27-57_processed.pickle')


@pytest.fixture(scope='module')
def est():
    if not SESSION.exists():
        pytest.skip(f'real session not available: {SESSION.name}')
    with open(SESSION, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def expected(est):
    df = est.metrics_df
    cfg = getattr(est, 'autoinspection_config', None) or {}
    thr = cfg.get('ml_threshold', DEFAULT_ML_THRESHOLD)
    accepted = set(int(i) for i in est.idx_components)
    merged = set(int(i) for i in df.loc[df['decision'] == 'from_merge', 'component_idx']) & accepted
    assert merged, 'session does not exercise merging'
    scored = set(int(i) for i in df.loc[df['ml_keep_probability'] >= thr, 'component_idx']) & accepted
    return {'merged': merged, 'scored': scored, 'all': merged | scored}


@pytest.fixture(scope='module')
def exported(est, tmp_path_factory):
    out = tmp_path_factory.mktemp('export')
    subprocess.run([sys.executable, str(REPO / 'export_estimates_data.py'), str(SESSION), '-o', str(out)],
                   check=True, cwd=REPO)
    name = extract_session_name(est)
    npz = np.load(out / f'{name}_data.npz')
    with open(out / f'{name}_metadata.json') as f:
        meta = json.load(f)
    mat = loadmat(out / f'{name}_filters.mat')
    return {'indices': set(int(i) for i in npz['component_indices']),
            'n_C': npz['C'].shape[0],
            'meta_indices': set(int(i) for i in meta['metrics_df']['component_idx']),
            'n_mat': mat['A'].shape[0]}


def test_merged_components_are_exported(expected, exported):
    lost = expected['merged'] - exported['indices']
    assert not lost, f'{len(lost)} of {len(expected["merged"])} merged components dropped: {sorted(lost)}'


def test_threshold_still_applies_to_scored_components(expected, exported):
    assert exported['indices'] == expected['all']


def test_all_outputs_describe_same_components(exported):
    assert exported['n_C'] == len(exported['indices'])
    assert exported['meta_indices'] == exported['indices']
    assert exported['n_mat'] == len(exported['indices'])
