"""Synthetic sessions and a click driver for the tests of the inspection window (bm_examinator)."""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import pickle
from pathlib import Path

import numpy as np
from bokeh.document.document import Document
from bokeh.events import ButtonClick
from bokeh.models import Button, ColumnDataSource
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams
from scipy.sparse import csc_matrix


def make_estimates(folder, n=8, T=300) -> Path:
    """n round footprints in rows of four, T frames; enough for delete, merge, zoom and revert.

    The default, 8 footprints on 48x48 and 300 frames, is the session the callback tests were written on;
    a larger n widens the field and keeps the same footprints and random draws for the first eight."""
    rng = np.random.default_rng(0)
    dims = (48, 20 * (-(-n // 4)) + 8)
    yy, xx = np.mgrid[0:dims[0], 0:dims[1]]
    A = np.zeros((dims[0] * dims[1], n))
    C = np.zeros((n, T))
    for k in range(n):
        cy, cx = 8 + 10 * (k % 4), 10 + 20 * (k // 4)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
        A[:, k] = g.ravel(order="F")
        spikes = (rng.random(T) < 0.03).astype(float)
        C[k] = np.convolve(spikes, np.exp(-np.arange(60) / 5.0))[:T] + 0.01 * k
    est = Estimates(A=csc_matrix(A), C=C, b=None, f=None, dims=dims)
    est.YrA = 0.01 * rng.normal(size=(n, T))
    est.S = np.zeros((n, T))
    est.bl = np.zeros(n)
    est.c1 = np.zeros(n)
    est.neurons_sn = np.ones(n)
    est.g = np.full((n, 1), 0.9)
    est.idx_components = np.arange(n)
    est.idx_components_bad = np.array([], dtype=int)
    est.SNR_comp = np.arange(n) + 1.0
    est.r_values = np.linspace(0.5, 0.9, n)
    est.imax = np.ones(dims)
    est.cnmf_dict = CNMFParams(params_dict={"p": 1, "fr": 20}).to_dict()
    path = Path(folder) / "NOF_H01_1D_estimates.pickle"  # the window takes the session id from the name
    with open(path, "wb") as f:
        pickle.dump(est, f)
    return path


class App:
    """make_app(...)(Document()) and handles to what the user clicks."""

    def __init__(self, bkapp):
        doc = Document()
        bkapp(doc)
        self.doc = doc
        self.buttons = {}
        sources = []
        for root in doc.roots:
            for m in root.references():
                if isinstance(m, Button):
                    self.buttons[m.label] = m
                elif isinstance(m, ColumnDataSource) and "idx" in m.data:
                    sources.append(m)
        assert len(sources) == 1, len(sources)
        self.src = sources[0]
        self.storage = self.buttons["Delete sel."]._event_callbacks["button_click"][0].keywords["storage"]

    def select(self, rows):
        self.src.selected.indices = list(rows)

    def click(self, label):  # whole chain
        b = self.buttons[label]
        b._trigger_event(ButtonClick(b))

    def first(self, label):  # own callback only
        b = self.buttons[label]
        b._event_callbacks["button_click"][0](ButtonClick(b))

    def shown(self):
        return [int(i) for i in self.src.data["idx"]]

    def partial(self):
        return [int(i) for i in self.storage.estimates_partial.idx_components]
