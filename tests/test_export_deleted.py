"""
Export must skip neurons that autoinspection marked as deleted.

Deleted neurons stay in idx_components on purpose (the ExamineCells GUI shows
them with their deletion reason); their status lives in metrics_df['delete'].
A filter on ML probability alone lets some of them through: the duplicate rule
deletes the lower-SNR neuron of a close pair whatever its probability, and an
older session may have been inspected with a higher threshold than the export's.

Exports a real autoinspected session from output/ as the export stage does
(tests/export_helpers.export_processed; gitignored, so the test skips on a
machine without it).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from export_estimates_data import extract_session_name
from export_helpers import export_processed

SESSION = REPO / 'output' / 'inspection_artifacts_LNOF_J12_1D' / 'LNOF_J12_1D_processed.pickle'


@pytest.fixture(scope='module')
def est():
    if not SESSION.exists():
        pytest.skip(f'real session not available: {SESSION.name}')
    with open(SESSION, 'rb') as f:
        return pickle.load(f)


@pytest.fixture(scope='module')
def exported(est, tmp_path_factory):
    out = tmp_path_factory.mktemp('export')
    export_processed(SESSION, out)
    npz = np.load(out / f'{extract_session_name(est)}_data.npz')
    return set(int(i) for i in npz['component_indices'])


def test_no_deleted_neuron_is_exported(est, exported):
    df = est.metrics_df
    deleted = set(int(i) for i in df.loc[df['delete'] == 1, 'component_idx'])
    leaked = exported & deleted
    assert not leaked, f'{len(leaked)} deleted neurons exported: {sorted(leaked)}'


def test_kept_neurons_are_still_exported(est, exported):
    # Guards against the fix dropping everything: this session keeps hundreds
    assert len(exported) > 300
