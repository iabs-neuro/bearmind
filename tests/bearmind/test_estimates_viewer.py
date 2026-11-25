from concurrent.futures import Future
from types import SimpleNamespace

from bokeh.document import Document
from bokeh.models import BoxSelectTool
import numpy as np
import pandas as pd
from scipy import sparse

from bearmind import estimates_viewer as viewer
from bearmind.paths import SeriesPaths


def test_read_only_viewer_filters_saved_decisions_without_renumbering(tmp_path, monkeypatch):
    paths = SeriesPaths(tmp_path)
    source = viewer.Source(tmp_path / "test_processed.pickle", "Autoinspected")
    monkeypatch.setattr(viewer, "_catalog", lambda *_: ({}, {"test": [source]}, None))
    data = pd.DataFrame([
        {"component_idx": 1, "delete": 1, "ml_keep_probability": .4, "decision": "delete"},
        {"component_idx": 3, "delete": 0, "decision": "from_merge"},
        {"component_idx": 2, "delete": 1, "ml_keep_probability": .95, "decision": "delete"},
        {"component_idx": 4, "delete": 1, "decision": "delete", "failed_area": 1, "area": .4},
        {"component_idx": 5, "delete": 1, "decision": "delete", "duplicate_of_input": 3,
         "duplicate_distance_px": 4.2, "duplicate_snr_difference": 12.1},
    ])
    est = SimpleNamespace(C=np.arange(60).reshape(6, 10), imax=np.zeros((3, 5)), idx_components=[1, 2, 3, 4, 5],
                          metrics_df=data, autoinspection_config={"ml_threshold": .72, "enable_merge": True,
                                                                  "deletion_rules": "DEFAULT_DELETION_RULES"})
    statuses, metrics, threshold = viewer._component_info(est)

    class Executor:
        def submit(self, *_):
            future = Future()
            future.set_result((est, np.arange(10), "Saved timestamps", statuses, metrics, threshold,
                               [np.ones(3)] * 6, [np.ones(3)] * 6))
            return future

    doc = Document()
    v = viewer.Viewer(doc, paths, "test", None, Executor())
    doc.session_callbacks[0].callback()
    assert v.filter.options == [viewer.ALL, viewer.KEPT, viewer.ALL_REJECTED, viewer.UNSELECTED, viewer.MODEL,
                                viewer.POSSIBLE_DUPLICATE, viewer.MERGED, viewer.RULES, viewer.DUPLICATE]
    assert v.contours.selected.indices == list(range(6))
    assert v.stack.data["id"] == list(range(6))
    assert isinstance(v.image.toolbar.active_drag, BoxSelectTool)
    assert "Footprint area (px²) = 0.4" in v.contours.data["reason"][4]
    assert "rejection condition: < 1" in v.contours.data["reason"][4]
    assert "input component #3" in v.contours.data["reason"][5]
    v.contours.selected.indices = [2, 4]
    assert v.stack.data["id"] == [2, 4]
    v.component.value = "4"  # focus an existing selection without clearing the group
    assert v.stack.data["id"] == [2, 4]
    v.filter.value = viewer.KEPT
    assert v.index_filter.indices == [3]
    assert v.stack.data["id"] == [3]
    v.filter.value = viewer.MODEL
    assert v.index_filter.indices == [1]
    assert v.component.value == "1" and list(v.trace.data["signal"]) == list(range(10, 20))
    assert not v.details.visible
    v._metrics_visible()
    assert v.details.visible and "ml_keep_probability" in v.details.text
    assert not list(tmp_path.iterdir())


def test_manual_statuses_follow_footprints_when_indices_change():
    original = SimpleNamespace(A=sparse.eye(4, format="csc"), C=np.zeros((4, 10)), idx_components=[0, 3])
    reordered = SimpleNamespace(A=original.A[:, [3, 2, 0]], C=np.zeros((3, 10)), idx_components=[2])
    statuses, *_ = viewer._component_info(reordered, original)
    assert statuses == [viewer.MANUAL_REJECTED, viewer.CNMF_REJECTED, viewer.KEPT]
