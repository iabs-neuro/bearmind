"""
Validation against manual labels must count only the neurons autoinspection kept.

implement_decision leaves deleted neurons in idx_components (the GUI shows them
with their deletion reason). Taking all of idx_components as "the automatic
result" therefore scores the pipeline as if it had deleted nothing.

Runs on a real session pair from data/raw_compressed and data/final_compressed
(gitignored, so the test skips on a machine without them).
"""
# capcan_validation imports auto_inspector, which imports torch indirectly; on
# this environment torch's shm.dll fails to load (WinError 127) unless torch is
# imported first.
import torch  # noqa: F401

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import capcan_validation as cv

# Loky worker processes import torch on their own and hit the same shm.dll
# failure; threads share this process, where torch is already loaded.
config.CONFIG['PARALLELIZATION_BACKEND'] = 'threading'

# A small session, so the test runs in about a minute
SESSION_PREFIX = 'NOF_H03_1D'


@pytest.fixture(scope='module')
def result():
    found = [s for s in cv.mapping if s.startswith(SESSION_PREFIX)]
    if not found:
        pytest.skip(f'session pair not available: {SESSION_PREFIX}')
    init_file, gt_file = cv.mapping[found[0]]
    return cv.validate_single_session(found[0], init_file, gt_file, verbose=False)


def test_deleted_neurons_are_not_counted_as_kept(result):
    assert result['n_deleted'] > 0, 'session does not exercise deletion'
    # Merges can only lower the count further
    assert result['n_auto'] <= result['n_initial'] - result['n_deleted'], (
        f"{result['n_auto']} counted as kept, but {result['n_deleted']} of "
        f"{result['n_initial']} were deleted")
