"""Synthetic sessions and a click driver for the tests of the inspection window (bm_examinator)."""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import json
import pickle
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from bokeh.document.document import Document
from bokeh.document.events import (ColumnDataChangedEvent, ColumnsPatchedEvent, ColumnsStreamedEvent,
                                   DocumentPatchedEvent, ModelChangedEvent)
from bokeh.events import ButtonClick, RangesUpdate
from bokeh.models import Button, ColumnDataSource, Div, GlyphRenderer, MultiLine, Plot, RadioButtonGroup
from bokeh.protocol import Protocol
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams
from scipy.sparse import csc_matrix

PANEL_TEXT = "<b>Neuron Metrics</b>"  # the metrics panel's text when the window opens starts with it


def make_estimates(folder, n=8, T=300) -> Path:
    """n round footprints in rows of four, T frames; enough for delete, merge, zoom and revert.

    The default, 8 footprints on 48x48 and 300 frames, is the session the callback tests were written on
    (same footprints and traces); a larger n widens the field and keeps the same footprints and random
    draws for the first eight."""
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
    est.R = est.YrA  # as CaImAn's CNMF.fit leaves it
    est.S = np.zeros((n, T))
    # distinct per component, so a value carried to the wrong component shows; the window and CaImAn's
    # manual_merge do not read them, so the traces and merges do not depend on them
    k = np.arange(n)
    est.bl = 0.01 * k
    est.c1 = 0.02 * k
    est.neurons_sn = 1.0 + 0.1 * k
    est.g = (0.9 - 0.005 * k)[:, None]
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


def message_bytes(msg) -> int:
    """Size of a Bokeh protocol message as the server writes it: header, metadata, content, buffers."""
    n = len(msg.header_json) + len(msg.metadata_json) + len(msg.content_json)
    return n + sum(len(json.dumps(b.ref)) + len(b.to_bytes()) for b in msg.buffers)


def patch_doc_bytes(events) -> int:
    """Size of the PATCH-DOC message built from the document events of one action (0: nothing sent)."""
    return message_bytes(Protocol().create("PATCH-DOC", events)) if events else 0


def changed_columns(events) -> set:
    """(source, column) of every data column the events send to the browser."""
    out = set()
    for e in events:
        if isinstance(e, (ColumnDataChangedEvent, ColumnsStreamedEvent, ColumnsPatchedEvent)):
            if isinstance(e, ColumnDataChangedEvent):
                cols = e.cols if e.cols is not None else list(e.model.data)
            elif isinstance(e, ColumnsStreamedEvent):
                cols = list(e.data)
            else:
                cols = list(e.patches)
            out |= {(e.model, c) for c in cols}
        elif isinstance(e, ModelChangedEvent) and isinstance(e.model, ColumnDataSource) and e.attr == "data":
            out |= {(e.model, c) for c in e.new}
    return out


class App:
    """make_app(...)(Document()) and handles to what the user clicks."""

    def __init__(self, bkapp):
        doc = Document()
        bkapp(doc)
        self.doc = doc
        self.buttons = {}
        self.sorting = []
        sources = []
        self.panel = None
        self.traces_plot = None
        for root in doc.roots:
            for m in root.references():
                if isinstance(m, Button):
                    self.buttons[m.label] = m
                elif isinstance(m, RadioButtonGroup):
                    self.sorting.append(m)
                elif isinstance(m, ColumnDataSource) and "idx" in m.data:
                    sources.append(m)
                elif isinstance(m, Div) and m.text.startswith(PANEL_TEXT):
                    self.panel = m
                elif isinstance(m, GlyphRenderer) and isinstance(m.glyph, MultiLine) and self.traces_plot is None:
                    self.traces_plot = [p for p in root.references() if isinstance(p, Plot) and m in p.renderers][0]
        assert len(sources) == 1, len(sources)
        self.src = sources[0]
        self.storage = self.buttons["Delete sel."]._event_callbacks["button_click"][0].keywords["storage"]

    def _slots(self):
        """Data source row of each row on screen, in the window's row order (component numbers ascending)."""
        return [self.storage.slot_of(c) for c in self.storage.shown]

    def select(self, rows):
        """A click on the rows at these positions of the list on screen (the golden's row numbers)."""
        slots = self._slots()
        self.src.selected.indices = [slots[r] for r in rows]

    def click(self, label):  # whole chain
        b = self.buttons[label]
        b._trigger_event(ButtonClick(b))

    def sort(self, label):
        """A click on a sorting button: the browser sets 'active', then sends the click (legacy mode)."""
        group, = [g for g in self.sorting if label in g.labels]
        group.active = group.labels.index(label)
        group._trigger_event(ButtonClick(group))

    def first(self, label):  # own callback only
        b = self.buttons[label]
        b._event_callbacks["button_click"][0](ButtonClick(b))

    def zoom(self, x0, x1):
        """The browser's message after a zoom or pan of the traces plot to x0..x1 seconds."""
        p = self.traces_plot
        p._trigger_event(RangesUpdate(p, x0=x0, x1=x1, y0=p.y_range.start, y1=p.y_range.end))

    def shown(self):
        """Component numbers of the rows the browser draws (the view's rows of the data source)."""
        view = [r for r in self.doc.roots[0].references()
                if isinstance(r, GlyphRenderer) and r.data_source is self.src][0].view
        comps = [int(self.src.data["idx"][s]) for s in view.filter.indices]
        assert comps == [int(c) for c in self.storage.shown], (comps, self.storage.shown)
        return comps

    def partial(self):
        return [int(i) for i in self.storage.shown]

    def time_axis(self):
        """The one time array the traces are drawn against."""
        tsrc, = [m for m in self.doc.roots[0].references()
                 if isinstance(m, ColumnDataSource) and set(m.data) == {"t"}]
        return np.asarray(tsrc.data["t"])

    def heights(self):
        """{component: vertical position of its trace} for the rows on screen."""
        return {int(self.src.data["idx"][s]): float(self.src.data["dummy_id"][s]) for s in self._slots()}

    @contextmanager
    def events(self):
        """The document events (what goes to the browser) of the actions inside the block."""
        self.doc.to_json()  # as after the browser's first pull: every model is known to it
        got = []

        def keep(event):
            if isinstance(event, DocumentPatchedEvent):
                got.append(event)
        self.doc.on_change(keep)
        try:
            yield got
        finally:
            self.doc.remove_on_change(keep)
