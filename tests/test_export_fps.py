"""
Export must carry the frame rate from fps_data.csv exactly, in the metadata JSON and
in the NPZ next to the traces.

A rounded rate shifts the time axis: 46157 frames at 19.76 fps last 2336 s, at 20 fps
2308 s. The NPZ is often used on its own, so it needs the rate too.

Runs the real export script on a real autoinspected session from output/
(gitignored, so the test skips on a machine without it).
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

SESSION = (REPO / 'output' / 'inspection_artifacts_WAW_H02_1D'
           / 'WAW_H02_1D30-03-2026 19-27-57_processed.pickle')
SESSION_NAME = 'WAW_H02_1D'


@pytest.fixture(scope='module')
def table_fps():
    df = pd.read_csv(REPO / 'fps_data.csv', sep=';')
    fps = float(df.loc[df['Filename'] == SESSION_NAME, 'FPS'].values[0])
    assert fps != round(fps), 'session does not exercise rounding'
    return fps


@pytest.fixture(scope='module')
def exported(tmp_path_factory):
    if not SESSION.exists():
        pytest.skip(f'real session not available: {SESSION.name}')
    out = tmp_path_factory.mktemp('export')
    subprocess.run([sys.executable, str(REPO / 'export_estimates_data.py'), str(SESSION), '-o', str(out)],
                   check=True, cwd=REPO)
    npz = np.load(out / f'{SESSION_NAME}_data.npz')
    with open(out / f'{SESSION_NAME}_metadata.json') as f:
        meta = json.load(f)
    return npz, meta


def test_metadata_fps_is_exact(exported, table_fps):
    _, meta = exported
    assert meta['fps'] == pytest.approx(table_fps, abs=1e-9)


def test_npz_carries_same_fps(exported, table_fps):
    npz, _ = exported
    assert 'fps' in npz.files, 'NPZ has no frame rate'
    assert float(npz['fps']) == pytest.approx(table_fps, abs=1e-9)
