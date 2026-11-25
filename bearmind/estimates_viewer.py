"""Read-only Bokeh viewer for estimates produced by BEARMiND."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from html import escape
import logging
from pathlib import Path
import pickle
import re
from time import monotonic
from urllib.parse import quote
import webbrowser

from bokeh.events import RangesUpdate
from bokeh.layouts import column, row
from bokeh.models import (BoxSelectTool, Button, CDSView, ColumnDataSource, Div, HoverTool, IndexFilter,
                          LinearColorMapper, Range1d, Select, Slider, TapTool)
from bokeh.plotting import figure
from bokeh.server.server import Server
import numpy as np
from tornado.ioloop import IOLoop, PeriodicCallback
from tornado.web import StaticFileHandler

from bearmind import component_ids, registry
from bearmind.decision_rules import DEFAULT_DELETION_RULES
from bearmind.paths import SeriesPaths
from bearmind.timestamps import session_time_s
from bearmind.trace_view import decimate_minmax
from naming import extract_session_id


log = logging.getLogger(__name__)
ALL = "All components"
KEPT = "Kept"
MERGED = "Product of a merge"
RULES = "Rejected by rules"
MODEL = "Rejected by ML"
DUPLICATE = "Removed as a duplicate"
POSSIBLE_DUPLICATE = "Possible duplicate"
REJECTED = "Rejected"
UNSELECTED = "Not selected in this file"
CNMF_REJECTED = "Rejected by CaImAn"
MANUAL_REJECTED = "Removed in manual inspection"
ALL_REJECTED = "All rejected"
COLORS = {KEPT: "#19cf93", MERGED: "#65c9ff", RULES: "#ff92a7", MODEL: "#ffb459",
          DUPLICATE: "#ca9bff", POSSIBLE_DUPLICATE: "#ca9bff", REJECTED: "#fc7f92",
          UNSELECTED: "#bbbfc5", CNMF_REJECTED: "#a9b2bc", MANUAL_REJECTED: "#f45cb4"}
REJECTIONS = set(COLORS) - {KEPT, MERGED}
DEFAULT_FPS = 20.0
CONTOUR_THRESHOLD = 0.3
RULE_NAMES = {"area": "Footprint area (px²)", "circularity": "Circularity",
              "events_per_min": "Detected events per minute", "event_r2_score": "Event reconstruction R²"}


@dataclass(frozen=True)
class Source:
    path: Path
    label: str


def sources_for(paths, rec):
    sid = rec.sid
    estimates = paths.estimates(sid)
    cnmf = rec.stages.get("cnmf")
    auto = rec.stages.get("autoinspect")
    folder = paths.inspection_dir(sid)
    if cnmf and cnmf.outputs.get("estimates"):
        estimates = paths.root / cnmf.outputs["estimates"]
    processed = paths.root / auto.outputs["processed"] if auto and auto.outputs.get("processed") else (
        folder / f"{sid}_processed.pickle")
    versions = [
        Source(processed, "Autoinspected"),
        Source(paths.manual_result(sid), "Manual inspection"),
        Source(estimates, "CNMF-E (full)"),
        Source(estimates.with_stem(f"{estimates.stem}_light"), "CNMF-E (light)"),
    ]
    versions.extend(Source(p, f"Progress: {p.name}")
                    for p in sorted(paths.inspection_progress_dir(sid).glob("*.pickle"), reverse=True))
    return [s for s in versions if s.path.is_file()]


def _catalog(paths, extra_file=None):
    records = {rec.sid: rec for rec in registry.all_records(paths)}
    catalog = {sid: files for sid, rec in records.items() if (files := sources_for(paths, rec))}
    custom_sid = None
    if extra_file is not None:
        sid = extract_session_id(extra_file.name)
        custom_sid = sid if sid in records else "Selected file"
        files = catalog.setdefault(custom_sid, [])
        chosen = Source(extra_file, f"Selected file: {extra_file.name}")
        files[:] = [chosen, *(f for f in files if f.path.resolve() != extra_file)]
    return records, catalog, custom_sid


def _mc_preview(paths, sid):
    preview = paths.preview(sid, "mc")
    original = paths.mc_video(sid)
    try:
        if preview.is_file() and (not original.is_file() or preview.stat().st_mtime_ns >= original.stat().st_mtime_ns):
            return preview
    except OSError:
        pass  # a preview may be replaced or removed between the file checks
    return None


def _finite(value):
    try:
        return bool(np.isfinite(value))
    except TypeError:
        return False


def _decision(metrics, threshold, enable_merge):
    if metrics.get("decision") == "from_merge":
        return MERGED
    if not (metrics.get("delete") == 1 or metrics.get("decision") == "delete"):
        return KEPT
    if any(k.startswith("failed_") and v == 1 for k, v in metrics.items()):
        return RULES
    if _finite(metrics.get("duplicate_of_input")):
        return DUPLICATE
    score = metrics.get("ml_keep_probability")
    if _finite(score) and threshold is not None:
        if score < threshold:
            return MODEL
        if enable_merge:
            return POSSIBLE_DUPLICATE
    return REJECTED


def _component_info(est, baseline=None, *, cnmf_file=False):
    count = est.C.shape[0]
    indices = getattr(est, "idx_components", None)
    selected = set(range(count) if indices is None else indices)
    metrics = getattr(est, "metrics_df", None)
    if metrics is None or "component_idx" not in metrics.columns:
        saved = {}
    else:
        saved = {int(m["component_idx"]): m for m in metrics.to_dict("records")}
    cfg = getattr(est, "autoinspection_config", None) or {}
    threshold = cfg.get("ml_threshold")
    threshold = float(threshold) if _finite(threshold) else None
    statuses = [KEPT if i in selected else CNMF_REJECTED if cnmf_file else UNSELECTED for i in range(count)]
    if baseline is not None:
        matched = component_ids.match(baseline, est)
        original_selected = component_ids.selected(baseline)
        for i, original in matched.items():
            if statuses[i] == UNSELECTED:
                statuses[i] = MANUAL_REJECTED if original in original_selected else CNMF_REJECTED
    for i, record in saved.items():
        if 0 <= i < count:
            statuses[i] = _decision(record, threshold, cfg.get("enable_merge", False))
    return statuses, saved, threshold


def _rule_reason(metrics, cfg):
    rules = cfg.get("deletion_rules")
    if rules == "DEFAULT_DELETION_RULES":
        rules = DEFAULT_DELETION_RULES
    rules = rules if isinstance(rules, (list, tuple)) else ()
    reasons = []
    predicates = {"<": np.less, ">": np.greater, "<=": np.less_equal, ">=": np.greater_equal}
    for key, failed in metrics.items():
        if not key.startswith("failed_") or failed != 1:
            continue
        metric = key.removeprefix("failed_")
        if metric == "corner_artifact":
            reasons.append("Flagged by corner artifact detection")
            continue
        terms = [m for rule in rules if isinstance(rule, str) and
                 (m := re.fullmatch(r"\s*([\w]+)\s*(<=|>=|<|>)\s*([-+\d.eE]+)\s*", rule))
                 and m[1] == metric]
        value = metrics.get(metric)
        name = RULE_NAMES.get(metric, metric.replace("_", " ").capitalize())
        no_value = not _finite(value) or (metric in {"t_rise", "t_off"} and value < 0)
        if no_value:
            reasons.append(f"{name}: no valid measured value")
        else:
            matched = [f"{m[2]} {m[3]}" for m in terms if predicates[m[2]](value, float(m[3]))]
            explanation = " or ".join(matched) if matched else "condition not stored"
            reasons.append(f"{name} = {value:.4g}; rejection condition: {explanation}")
    return "; ".join(reasons) or "Rejected by a rule whose details were not stored."


def _reason(status, metrics, cfg):
    if status == RULES:
        return _rule_reason(metrics, cfg)
    if status == MODEL:
        return f"ML score {metrics['ml_keep_probability']:.5g} is below {float(cfg['ml_threshold']):g}."
    if status == DUPLICATE:
        return (f"Lower SNR than input component #{int(metrics['duplicate_of_input'])}. "
                f"Contour distance: {metrics['duplicate_distance_px']:.2f} px. "
                f"SNR difference: {metrics['duplicate_snr_difference']:.2f}.")
    if status == POSSIBLE_DUPLICATE:
        distance, snr = cfg.get("pxlthr_distance_boundary"), cfg.get("d_snr_thr")
        criterion = (f"Contour distance ≤ {distance:g} px and SNR difference > {snr:g}."
                     if _finite(distance) and _finite(snr) else "The duplicate rule requires nearby contours.")
        return (f"ML score passed. {criterion} The comparison uses a correlation group. "
                "This file does not store the rejection partner or its distance.")
    return {
        CNMF_REJECTED: "Excluded during CaImAn component evaluation.",
        MANUAL_REJECTED: "Accepted by CaImAn and excluded in the saved manual selection.",
        MERGED: "Kept product of automatic merging. No ML score is computed for this product.",
        UNSELECTED: "Outside the accepted indices stored in this file. Earlier status unavailable.",
    }.get(status, status)


def _valid_time(value, count):
    if value is None:
        return None
    axis = np.asarray(value, dtype=float)
    if axis.ndim != 1 or len(axis) != count or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        return None
    return axis - axis[0]


def _time_axis(est, paths, rec):
    count = est.C.shape[1]
    saved = _valid_time(getattr(est, "time", None), count)
    if saved is not None:
        return saved, "Saved timestamps"
    if rec is not None:
        try:
            actual = _valid_time(session_time_s(paths, rec), count)
        except (OSError, ValueError):
            actual = None
        if actual is not None:
            return actual, "Session timestamps"
    fps = rec.fps if rec is not None else getattr(est, "fr", None)
    if not _finite(fps) or fps <= 0:
        fps = DEFAULT_FPS
    return np.arange(count, dtype=float) / fps, f"Uniform time axis ({fps:.4g} fps)"


def _contours(est, threshold, method="nrg"):
    from caiman.utils.visualization import get_contours

    count = est.C.shape[0]
    contours = get_contours(est.A, dims=est.imax.shape, thr=threshold, thr_method=method) if count else []
    coords = [c["coordinates"].reshape(-1, 2) for c in contours]
    height = est.imax.shape[0]
    return [c[:, 0] for c in coords], [height - c[:, 1] for c in coords]


def _load(source, paths, rec):
    with source.path.open("rb") as f:
        est = pickle.load(f)
    if getattr(est, "C", None) is None or est.C.ndim != 2 or not est.C.shape[1]:
        raise ValueError("The file does not contain component traces.")
    time, time_label = _time_axis(est, paths, rec)
    baseline = None
    cnmf_file = source.label.startswith("CNMF-E")
    if rec is not None:
        full = next((s.path for s in sources_for(paths, rec) if s.label == "CNMF-E (full)"), None)
        cnmf_file |= full is not None and full.resolve() == source.path.resolve()
        manual_file = (source.path.name.endswith("_manual_estimates.pickle")
                       or source.path.parent == paths.inspection_progress_dir(rec.sid))
        if manual_file and full is not None:
            try:
                with full.open("rb") as f:
                    baseline = pickle.load(f)
            except (OSError, ValueError, pickle.UnpicklingError):
                log.warning("Could not read the CaImAn decisions for %s", source.path, exc_info=True)
    statuses, metrics, threshold = _component_info(est, baseline, cnmf_file=cnmf_file)
    del baseline
    xs, ys = _contours(est, CONTOUR_THRESHOLD)
    return est, time, time_label, statuses, metrics, threshold, xs, ys


class Viewer:
    def __init__(self, doc, paths, sid, extra_file, executor):
        self.doc = doc
        self.paths = paths
        self.executor = executor
        self.records, self.catalog, custom_sid = _catalog(paths, extra_file)
        self.est = None
        self.metrics = {}
        self.statuses = []
        self.visible = []
        self.selected = []
        self.changing = False
        self.closed = False
        self.load_number = 0
        self.component_id = None
        self.doc.title = "BEARMiND · Estimates viewer"
        self.doc.on_session_destroyed(self._close)
        if not self.catalog:
            doc.add_root(Div(text="No estimates files found in this series."))
            return
        initial = custom_sid or sid or next(iter(self.catalog))
        self.session = Select(title="Session", options=list(self.catalog), value=initial, width=145)
        self.file = Select(title="Estimates file", width=335, sizing_mode="stretch_width", min_width=220)
        self.filter = Select(title="Components", options=[ALL], value=ALL, width=220)
        self.component = Select(title="Focus component", width=138)
        self.previous = Button(label="Previous session", width=116)
        self.following = Button(label="Next session", width=106)
        self.select_all = Button(label="Select all", width=83)
        self.clear = Button(label="Clear selection", width=105)
        self.message = Div(text="Loading estimates…", styles={"color": "#4d6174"}, sizing_mode="stretch_width")
        self.details = Div(visible=False, sizing_mode="stretch_width")
        self.toggle = Button(label="Show metrics", width=102)
        self.video = Div(visible=False, sizing_mode="stretch_width")
        self.toggle_video = Button(label="Show MC video", width=156, disabled=True)
        self.contours = ColumnDataSource(dict(id=[], xs=[], ys=[], decision=[], reason=[], score=[], color=[]))
        self.background = ColumnDataSource(dict(image=[], width=[], height=[]))
        self.trace = ColumnDataSource(dict(time=[], signal=[]))
        self.raw_trace = ColumnDataSource(dict(time=[], signal=[]))
        self.stack = ColumnDataSource(dict(id=[], time=[], signal=[], color=[], decision=[]))
        self.index_filter = IndexFilter(indices=[])
        tools = "pan,wheel_zoom,box_zoom,reset,save"
        self.image = figure(title="Spatial footprints", match_aspect=True, height=520, min_width=380,
                            sizing_mode="stretch_width", tools=tools + ",tap", toolbar_location="right",
                            outline_line_color="#d6e0e6", background_fill_color="#15242e")
        self.image.axis.visible = False
        self.image.grid.visible = False
        self.image.image(image="image", dw="width", dh="height", x=0, y=0, source=self.background,
                         color_mapper=LinearColorMapper(palette="Greys256", low=0, high=255))
        self.patches = self.image.patches("xs", "ys", source=self.contours, color="color",
                                          view=CDSView(filter=self.index_filter),
                                          fill_alpha=0.25, line_width=1.5, line_alpha=1,
                                          nonselection_fill_alpha=0.035, nonselection_line_alpha=0.24)
        self.image.select_one(TapTool).renderers = [self.patches]
        area = BoxSelectTool(renderers=[self.patches], description="Select footprints in an area")
        self.image.add_tools(area)
        self.image.toolbar.active_drag = area
        self.image.add_tools(HoverTool(renderers=[self.patches],
                                      tooltips=[("Component", "@id"), ("Decision", "@decision"),
                                                ("Reason", "@reason"), ("ML score", "@score")]))
        self.stack_plot = figure(title="Selected calcium traces · C", x_axis_label="Time (s)",
                                 height=520, min_width=380, sizing_mode="stretch_width", tools=tools,
                                 x_range=Range1d(0, 1), y_range=Range1d(0, 1), toolbar_location="right",
                                 outline_line_color="#d6e0e6", background_fill_color="#182a36")
        stack_lines = self.stack_plot.multi_line("time", "signal", source=self.stack, line_color="color",
                                                  line_width=1.1)
        self.stack_plot.yaxis.visible = False
        self.stack_plot.ygrid.visible = False
        self.stack_plot.xgrid.grid_line_alpha = 0.15
        self.stack_plot.add_tools(HoverTool(renderers=[stack_lines], tooltips=[
            ("Component", "@id"), ("Decision", "@decision")]))
        self.trace_plot = figure(title="Calcium trace", x_axis_label="Time (s)", y_axis_label="Signal (a.u.)",
                                 height=240, visible=False, sizing_mode="stretch_width",
                                 tools=tools, x_range=self.stack_plot.x_range, toolbar_location="right")
        self.trace_plot.line("time", "signal", source=self.raw_trace, color="#7d8794", alpha=0.65, legend_label="C + YrA")
        self.trace_plot.line("time", "signal", source=self.trace, color="#146db1", line_width=2, legend_label="C")
        self.trace_plot.legend.click_policy = "hide"
        width = Slider(title="Contour line width", start=0.5, end=6, step=0.5, value=1.5, width=142)
        opacity = Slider(title="Contour fill opacity", start=0, end=1, step=0.05, value=0.25, width=146)
        self.threshold = Slider(title="Contour threshold", start=0.05, end=0.95, step=0.05,
                                value=CONTOUR_THRESHOLD, width=145)
        self.contour_method = Select(title="Contour metric", options=[("nrg", "Energy"), ("max", "Peak amplitude")],
                                     value="nrg", width=150)
        width.on_change("value", lambda attr, old, new: setattr(self.patches.glyph, "line_width", new))
        opacity.on_change("value", lambda attr, old, new: setattr(self.patches.glyph, "fill_alpha", new))
        self.threshold.on_change("value_throttled", self._change_contours)
        self.contour_method.on_change("value", self._change_contours)
        self.session.on_change("value", self._change_session)
        self.file.on_change("value", self._change_file)
        self.filter.on_change("value", self._change_filter)
        self.component.on_change("value", self._choose_component)
        self.contours.selected.on_change("indices", self._select_area)
        self.stack_plot.on_event(RangesUpdate, self._zoom)
        self.trace_plot.on_event(RangesUpdate, self._zoom)
        self.previous.on_click(partial(self._step, -1))
        self.following.on_click(partial(self._step, 1))
        self.select_all.on_click(lambda: self._set_selection(self.visible))
        self.clear.on_click(lambda: self._set_selection([]))
        self.toggle.on_click(self._metrics_visible)
        self.toggle_video.on_click(self._video_visible)
        header = Div(text="<strong>Estimates viewer</strong>",
                     styles={"font-size": "19px", "color": "#183044"}, margin=(4, 0, 6, 0))
        navigation = row(self.session, self.file, self.previous, self.following, sizing_mode="stretch_width",
                         align="end")
        selection = row(self.filter, self.component, self.select_all, self.clear, self.toggle, self.toggle_video,
                        sizing_mode="stretch_width", align="end")
        visuals = row(column(self.image, self.video, sizing_mode="stretch_width"),
                      column(self.stack_plot, self.trace_plot, self.details, sizing_mode="stretch_width"),
                      sizing_mode="stretch_width", spacing=12)
        geometry = row(self.contour_method, self.threshold, width, opacity, sizing_mode="stretch_width")
        doc.add_root(column(header, navigation, selection, self.message, visuals, geometry,
                            sizing_mode="stretch_width", max_width=1480, margin=(14, 18, 16, 18), spacing=8))
        self._change_session("value", "", initial)

    def _close(self, _context):
        self.closed = True
        self.est = None

    def _set_loading(self, loading):
        options = self.session.options
        i = options.index(self.session.value)
        self.session.disabled = self.file.disabled = loading
        self.previous.disabled = loading or i == 0
        self.following.disabled = loading or i == len(options) - 1
        disabled = loading or self.est is None
        self.filter.disabled = self.component.disabled = disabled
        self.select_all.disabled = self.clear.disabled = disabled
        self.threshold.disabled = self.contour_method.disabled = disabled

    def _step(self, step):
        options = self.session.options
        i = options.index(self.session.value) + step
        if 0 <= i < len(options):
            self.session.value = options[i]

    def _change_session(self, _attr, _old, new):
        if new not in self.catalog:
            return
        srcs = self.catalog[new]
        previous_label = self.source.label if hasattr(self, "source") else None
        preferred = next((i for i, f in enumerate(srcs) if f.label == previous_label), 0)
        self.changing = True
        self.file.options = [(str(i), s.label) for i, s in enumerate(srcs)]
        self.file.value = str(preferred)
        self.changing = False
        self._change_file("value", "", self.file.value)

    def _change_file(self, _attr, _old, new):
        if self.changing or not any(value == new for value, _ in self.file.options):
            return
        sid = self.session.value
        if sid not in self.catalog:
            return
        self.source = self.catalog[sid][int(new)]
        self.load_number += 1
        self.est = None
        self.component_id = None
        self.selected = []
        self.source_text = ""
        self.contours.selected.indices = []
        self.index_filter.indices = []
        self.contours.data = dict(id=[], xs=[], ys=[], decision=[], reason=[], score=[], color=[])
        self.background.data = dict(image=[], width=[], height=[])
        self.video.visible = False
        self.video.text = ""
        has_preview = sid in self.records and _mc_preview(self.paths, sid) is not None
        self.toggle_video.disabled = not has_preview
        self.toggle_video.label = "Show MC video" if has_preview else "MC video unavailable"
        self._draw_trace()
        self._draw_stack()
        self.details.text = ""
        self.message.text = f"Loading {escape(self.source.path.name)}…"
        self._set_loading(True)
        job = self.executor.submit(_load, self.source, self.paths, self.records.get(sid))
        job.add_done_callback(partial(self._loaded, self.load_number))

    def _loaded(self, number, job):
        if not self.closed:
            self.doc.add_next_tick_callback(partial(self._show, number, job))

    def _show(self, number, job):
        if self.closed or number != self.load_number:
            return
        try:
            self.est, self.time, time_label, self.statuses, self.metrics, threshold, xs, ys = job.result()
        except Exception as exc:
            log.exception("Could not load %s", self.source.path)
            self.message.text = f"Could not open {escape(self.source.path.name)}. {escape(str(exc))}"
            self._set_loading(False)
            return
        height, width = self.est.imax.shape
        image = np.clip(np.rint(np.nan_to_num(self.est.imax, nan=0)), 0, 255).astype(np.uint8)
        self.background.data = dict(image=[np.flipud(image)], width=[width], height=[height])
        self.image.x_range.start, self.image.x_range.end = 0, width
        self.image.y_range.start, self.image.y_range.end = 0, height
        self.cfg = getattr(self.est, "autoinspection_config", None) or {}
        self.reasons = [_reason(k, self.metrics.get(i, {}), self.cfg) for i, k in enumerate(self.statuses)]
        self.contours.data = dict(id=list(range(len(xs))), xs=xs, ys=ys, decision=self.statuses,
                                  reason=self.reasons,
                                  score=[self._score(i) for i in range(len(xs))],
                                  color=[COLORS[k] for k in self.statuses])
        self.changing = True
        options = list(dict.fromkeys(self.statuses))
        rejected = [ALL_REJECTED] if any(k in REJECTIONS for k in options) else []
        available = [k for k in options if k != KEPT]
        filters = [ALL, *([KEPT] if KEPT in options or MERGED in options else []), *rejected, *available]
        self.filter.options = filters
        self.filter.value = ALL
        self.threshold.value = CONTOUR_THRESHOLD
        self.contour_method.value = "nrg"
        self.changing = False
        self.trace_plot.x_range.start, self.trace_plot.x_range.end = 0, max(1, self.time[-1])
        self.trace_plot.x_range.reset_start, self.trace_plot.x_range.reset_end = 0, max(1, self.time[-1])
        self.trace_mins = np.min(self.est.C, axis=1)
        self.trace_scales = np.maximum(np.max(self.est.C, axis=1) - self.trace_mins, 1e-12)
        threshold_text = f" · ML threshold {threshold:g}" if threshold is not None else ""
        self.source_text = f"{escape(self.source.path.name)} · {time_label}{threshold_text}"
        self._change_filter("value", "", self.filter.value)
        self._set_loading(False)
        self.component.disabled = not self.visible

    def _score(self, i):
        score = self.metrics.get(i, {}).get("ml_keep_probability")
        return f"{score:.5f}" if _finite(score) else "—"

    def _change_filter(self, _attr, _old, new):
        if self.changing or self.est is None:
            return
        self.visible = [i for i, k in enumerate(self.statuses)
                        if new == ALL or k == new or (new == KEPT and k == MERGED)
                        or (new == ALL_REJECTED and k in REJECTIONS)]
        self.index_filter.indices = self.visible
        self.changing = True
        options = [("", "—"), *((str(i), f"#{i}") for i in self.visible)]
        self.component.options = options
        self.component.disabled = not self.visible
        self.changing = False
        self._set_selection(self.visible)

    def _choose_component(self, _attr, _old, new):
        if self.changing or self.est is None:
            return
        if new and not any(value == new for value, _ in self.component.options):
            return
        chosen = int(new) if new else None
        if chosen is not None and chosen not in self.selected:
            self._set_selection([chosen])
            return
        self.component_id = chosen
        axis = self.stack_plot.x_range
        self._draw_trace(axis.start, axis.end)
        self._refresh_details()

    def _select_area(self, _attr, _old, new):
        if not self.changing and self.est is not None:
            self._set_selection(new)

    def _set_selection(self, ids):
        if self.est is None:
            return
        self.selected = sorted(set(ids).intersection(self.visible))
        self.component_id = self.selected[0] if len(self.selected) == 1 else None
        self.changing = True
        self.contours.selected.indices = self.selected
        self.component.value = str(self.component_id) if self.component_id is not None else ""
        self.changing = False
        self.message.text = (f"{self.source_text} · {len(self.selected)} selected / "
                             f"{len(self.visible)} shown / {len(self.statuses)} in file")
        self.stack_plot.y_range.start = self.stack_plot.y_range.reset_start = -0.12
        self.stack_plot.y_range.end = self.stack_plot.y_range.reset_end = max(1, len(self.selected))
        self._zoomed()
        self._refresh_details()

    def _zoom(self, event):
        if event.x0 is not None and event.x1 is not None:
            self._draw_stack(event.x0, event.x1)
            self._draw_trace(event.x0, event.x1)

    def _zoomed(self):
        axis = self.stack_plot.x_range
        self._draw_stack(axis.start, axis.end)
        self._draw_trace(axis.start, axis.end)

    def _sample_bounds(self, lo, hi):
        start = 0 if lo is None else max(0, int(np.searchsorted(self.time, lo)) - 1)
        end = len(self.time) if hi is None else min(len(self.time), int(np.searchsorted(self.time, hi)) + 1)
        return start, end

    def _draw_stack(self, lo=None, hi=None):
        ids = self.selected
        traces = dict(id=[], time=[], signal=[], color=[], decision=[])
        if self.est is not None and ids:
            start, end = self._sample_bounds(lo, hi)
            budget = min(2000, max(110, 230_000 // len(ids)))
            for order, i in enumerate(ids):
                frames, signal = decimate_minmax(np.asarray(self.est.C[i, start:end]), budget)
                normalized = (signal - self.trace_mins[i]) / self.trace_scales[i]
                traces["id"].append(i)
                traces["time"].append(self.time[start + frames])
                traces["signal"].append((len(ids) - 1 - order + .8 * normalized).astype(np.float32))
                traces["color"].append(COLORS[self.statuses[i]])
                traces["decision"].append(self.statuses[i])
        self.stack.data = traces
        self.stack_plot.title.text = f"Selected calcium traces · {len(ids)}"

    def _draw_trace(self, lo=None, hi=None):
        i = self.component_id
        self.trace_plot.visible = self.est is not None and i is not None
        if self.est is None or i is None:
            self.trace.data = self.raw_trace.data = dict(time=[], signal=[])
            self.trace_plot.title.text = "Calcium trace"
            return
        start, end = self._sample_bounds(lo, hi)
        trace = np.asarray(self.est.C[i, start:end])
        frames, signal = decimate_minmax(trace, 3000)
        self.trace.data = dict(time=self.time[start + frames], signal=signal)
        residual = getattr(self.est, "YrA", None)
        if residual is not None and residual.shape == self.est.C.shape:
            frames, signal = decimate_minmax(trace + residual[i, start:end], 3000)
            self.raw_trace.data = dict(time=self.time[start + frames], signal=signal)
        else:
            self.raw_trace.data = dict(time=[], signal=[])
        self.trace_plot.title.text = f"#{i} · {self.statuses[i]} · ML score {self._score(i)}"

    def _metrics_visible(self):
        self.details.visible = not self.details.visible
        self.toggle.label = "Hide metrics" if self.details.visible else "Show metrics"

    def _video_visible(self):
        sid = self.session.value
        preview = _mc_preview(self.paths, sid) if sid in self.records else None
        if preview is None:
            return
        self.video.visible = not self.video.visible
        self.video.text = (
            f'<video aria-label="Motion-corrected video" controls preload="metadata" width="540" '
            f'src="/mc-preview/{quote(preview.name)}"></video>'
        ) if self.video.visible else ""
        self.toggle_video.label = "Hide MC video" if self.video.visible else "Show MC video"

    def _refresh_details(self):
        i = self.component_id
        if i is None:
            self.details.text = "Focus a component to inspect its decision and metrics."
            return
        metrics = self.metrics.get(i, {})
        if not metrics:
            metrics = {key: values[i] for key in ("SNR_comp", "r_values")
                       if (values := getattr(self.est, key, None)) is not None and len(values) == len(self.statuses)}
        fields = []
        for key, value in metrics.items():
            if (key in {"component_idx", "center", "delete", "merge", "decision"}
                    or key.startswith(("failed_", "duplicate_")) or not _finite(value)):
                continue
            fields.append(f"<tr><td>{escape(key)}</td><td>{float(value):.5g}</td></tr>")
        summary = f"<b>#{i} · {escape(self.statuses[i])}</b><p>{escape(self.reasons[i])}</p>"
        self.details.text = summary + ("<table>" + "".join(fields) + "</table>" if fields else "No saved metrics.")

    def _change_contours(self, attr, _old, new):
        if self.changing or self.est is None:
            return
        est = self.est
        threshold = new if attr == "value_throttled" else self.threshold.value
        method = self.contour_method.value
        job = self.executor.submit(_contours, est, threshold, method)

        def done(future):
            if not self.closed:
                self.doc.add_next_tick_callback(partial(self._show_contours, est, threshold, method, future))

        job.add_done_callback(done)

    def _show_contours(self, est, threshold, method, job):
        if (self.closed or self.est is not est or self.threshold.value != threshold
                or self.contour_method.value != method):
            return
        try:
            xs, ys = job.result()
        except Exception:
            log.exception("Could not update contours")
            self.message.text = "Could not update contours at this threshold."
            return
        self.contours.data = dict(self.contours.data, xs=xs, ys=ys)


def run(series, sid=None, *, extra_file=None, port=0, open_browser=True):
    paths = SeriesPaths(series)
    if extra_file is not None:
        extra_file = Path(extra_file).resolve()
        if extra_file.suffix.lower() != ".pickle" or not extra_file.is_file():
            raise ValueError("Choose an existing estimates .pickle file.")
    _records, catalog, _ = _catalog(paths, extra_file)
    if sid is not None and sid not in catalog:
        raise ValueError(f"No estimates files found for {sid}.")
    if not catalog:
        raise ValueError("No estimates files found in this series.")
    loop = IOLoop.current()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="estimates-viewer") as executor:
        server = Server({"/": lambda doc: Viewer(doc, paths, sid, extra_file, executor)},
                        port=port, address="localhost", io_loop=loop,
                        check_unused_sessions_milliseconds=5000, unused_session_lifetime_milliseconds=5000,
                        extra_patterns=[(r"/mc-preview/([A-Za-z0-9_-]+_mc\.mp4)", StaticFileHandler,
                                         {"path": str(paths.root / "Preview")})])
        last_open = monotonic()

        def exit_if_idle():
            nonlocal last_open
            if server.get_sessions("/"):
                last_open = monotonic()
            elif monotonic() - last_open >= 120:
                loop.stop()

        timer = PeriodicCallback(exit_if_idle, 5000)
        try:
            server.start()
            timer.start()
            url = f"http://localhost:{server.port}/"
            print(f"[OK] Estimates viewer: {url}", flush=True)
            if open_browser:
                webbrowser.open(url)
            loop.start()
        finally:
            timer.stop()
            server.stop()
    return 0
