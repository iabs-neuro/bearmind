"""bm_spike_detection draws to notebook output itself; importing it does not switch Bokeh's output."""
# bm_spike_detection imports bm_examinator, which imports torch indirectly; on this environment torch's
# shm.dll fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import bm_spike_detection


def test_import_has_no_notebook_side_effect():
    code = ("import torch\n"
            "import bokeh.io, bokeh.io.output, bokeh.plotting\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('output_notebook called at import')\n"
            "bokeh.io.output_notebook = bokeh.io.output.output_notebook = boom\n"
            "bokeh.plotting.output_notebook = boom\n"
            "import bm_spike_detection\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]


def _csv(path, rows):
    np.savetxt(path, rows, delimiter=",", header="time_s,0,1", comments="")
    return str(path)


@pytest.mark.parametrize("which", ["FitEvents", "DrawSpEvents", "DrawSpEvents2"])
def test_drawing_turns_on_notebook_output(tmp_path, monkeypatch, which):
    t = np.arange(60) / 20.0
    traces = np.stack([t, np.exp(-((t - 1) ** 2) / 0.05), np.exp(-((t - 2) ** 2) / 0.05)], axis=1)
    spikes = np.stack([t, (np.arange(60) == 20) * 1.0, (np.arange(60) == 40) * 1.0], axis=1)
    tr = _csv(tmp_path / "NOF_H01_1D_traces.csv", traces)
    sp = _csv(tmp_path / "NOF_H01_1D_spikes.csv", spikes)
    calls = []
    bpl = bm_spike_detection.bpl
    monkeypatch.setattr(bpl, "output_notebook", lambda *a, **k: calls.append("output_notebook"))
    monkeypatch.setattr(bpl, "output_file", lambda *a, **k: calls.append("output_file"))
    monkeypatch.setattr(bpl, "show", lambda *a, **k: calls.append("show"))
    if which == "FitEvents":
        bm_spike_detection.FitEvents(tr, {"sigma": 1, "thr": 3, "draw_details": False,
                                          "est_ton": 0.05, "est_toff": 0.2})
    elif which == "DrawSpEvents":
        bm_spike_detection.DrawSpEvents(tr, sp)
    else:
        bm_spike_detection.DrawSpEvents2(tr, sp, sp)
    assert calls == ["output_notebook", "output_file", "show"]
