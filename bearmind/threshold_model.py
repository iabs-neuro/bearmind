"""Model of the Thresholds tab without Qt: the ready maps of the thresholds stage, the passing mask,
blob counters and the list of sessions with their states."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from bearmind import actions, config, registry, runner
from bearmind.registry import State

STATE_TEXT = {"waiting": "waiting", "chosen": "chosen", "cnmf_done": "CNMF-E done", "no_maps": "no maps",
              "excluded": "excluded", "not_needed": "thresholds not needed yet", "out_of_date": "maps out of date",
              "config_error": "cannot be scheduled"}
EXCLUDED_NOTE = "The session is excluded: the batch does not process it."

_EIGHT = np.ones((3, 3), dtype=int)  # 8-connectivity


@dataclass
class ThresholdMaps:
    gsigs: list[int]
    mean: np.ndarray
    max: np.ndarray
    cn: dict[int, np.ndarray]
    pnr: dict[int, np.ndarray]
    cn_mirror: dict[int, np.ndarray]      # empty when the file has no mirrored maps
    pnr_mirror: dict[int, np.ndarray]

    @property
    def usable_gsigs(self) -> list[int]:
        """The gSigs that can be shown and chosen: both maps present (a damaged file may list more)."""
        return [g for g in self.gsigs if g in self.cn and g in self.pnr]


@dataclass
class Counters:
    n_blobs: int
    single_frac: float
    n_mirror_blobs: int | None


@dataclass
class ThresholdItem:
    sid: str
    mouse: str
    # "excluded" | "config_error" | "no_maps" | "cnmf_done" | "out_of_date" | "chosen" | "waiting" | "not_needed"
    state: str
    reason: str         # the short why of no_maps, out_of_date and config_error (and of stale maps when excluded)
    suggested: dict | None
    chosen: dict | None  # {"gSig", "min_corr", "min_pnr"} from the resolved config
    # The batch waits for this session's thresholds: actions.waits_for_thresholds, the rule Monitor and the CLI
    # count by. True exactly in the state "waiting".
    waiting: bool = False
    first_stale: str | None = None  # the main stage the batch would (re)start the session from (StageView)
    # the sentence shown in place of the maps, which is also why Accept is refused; "" when the maps are shown
    note: str = ""
    text: str = ""  # what the session list says; "" for the plain STATE_TEXT (state_text)


def load_maps(path: Path) -> ThresholdMaps:
    with np.load(path) as z:
        gsigs = [int(g) for g in z["gsig"]]
        get = lambda prefix: {g: z[f"{prefix}_{g}"] for g in gsigs if f"{prefix}_{g}" in z.files}  # noqa: E731
        return ThresholdMaps(gsigs, z["mean"], z["max"], get("cn"), get("pnr"),
                             get("cn_mirror"), get("pnr_mirror"))


def passing_mask(cn: np.ndarray, pnr: np.ndarray, min_corr: float, min_pnr: float) -> np.ndarray:
    # comparisons with NaN are False already; inf passes a threshold, so it is cut explicitly
    with np.errstate(invalid="ignore"):
        return np.isfinite(cn) & np.isfinite(pnr) & (cn >= min_corr) & (pnr >= min_pnr)


def _group_sizes(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask, structure=_EIGHT)
    if n == 0:
        return np.zeros(0, dtype=int)
    return np.bincount(labels.ravel())[1:]


def _known(maps: ThresholdMaps, g: int) -> int:
    usable = maps.usable_gsigs
    return g if g in usable else config.fallback_gsig(usable)


def counters(maps: ThresholdMaps, g: int, min_corr: float, min_pnr: float, min_blob_px: int) -> Counters:
    g = _known(maps, g)
    sizes = _group_sizes(passing_mask(maps.cn[g], maps.pnr[g], min_corr, min_pnr))
    n_blobs = int((sizes >= min_blob_px).sum())
    single = float((sizes == 1).sum() / len(sizes)) if len(sizes) else 0.0
    n_mirror = None
    if g in maps.cn_mirror and g in maps.pnr_mirror:
        ms = _group_sizes(passing_mask(maps.cn_mirror[g], maps.pnr_mirror[g], min_corr, min_pnr))
        n_mirror = int((ms >= min_blob_px).sum())
    return Counters(n_blobs, single, n_mirror)


def display_u8(img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5) -> np.ndarray:
    img = np.asarray(img, dtype=float)
    fin = np.isfinite(img)
    if not fin.any():
        return np.zeros(img.shape, dtype=np.uint8)
    lo, hi = np.percentile(img[fin], [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    out = (np.clip(np.where(fin, img, lo), lo, hi) - lo) / (hi - lo)
    out[~fin] = 0
    return np.rint(out * 255).astype(np.uint8)


def gsig_thumbnail(maps: ThresholdMaps, g: int, size: int = 160) -> np.ndarray:
    g = _known(maps, g)
    prod = np.nan_to_num(maps.cn[g] * maps.pnr[g], nan=0.0, posinf=0.0, neginf=0.0)
    scale = size / max(prod.shape)
    if scale < 1:
        prod = ndimage.zoom(prod, scale, order=1)
    return display_u8(prod)


def last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _chosen(cnmf_cfg: dict) -> dict | None:
    v = {k: cnmf_cfg.get(k) for k in ("gSig", "min_corr", "min_pnr")}
    return v if all(x is not None for x in v.values()) else None


def _stale(th, first_stale) -> tuple[str, str, bool]:
    """(short reason, sentence, the batch recomputes them without a person) for maps that are not up to date
    while their old file is still there."""
    error = last_line(th.error).rstrip(".") if th is not None else ""
    state = th.state if th is not None else State.PENDING
    if state is State.RUNNING:
        return "maps being recomputed", "The maps are being recomputed by the batch.", True
    if state is State.PENDING:
        return ("maps out of date: the batch will recompute them",
                "The maps are out of date: the batch will recompute them.", True)
    if state is State.FAILED:
        return ("maps out of date: the thresholds stage failed",
                f"The maps are out of date: the thresholds stage failed: {error or 'no error text'}. "
                "Retry it in Monitor.", False)
    if state is State.SKIPPED:  # "skipped: mc failed": an earlier stage must be retried first
        return (f"maps out of date ({error or 'skipped'})",
                f"The maps are out of date ({error or 'skipped'}). Retry the failed stage in Monitor.", False)
    # DONE but not current: an input changed since (the crop rule, the box, MC settings) or an output is gone
    return (f"maps out of date: the batch redoes {first_stale} first",
            f"The maps are out of date: the batch redoes {first_stale} first.", True)


def _item(paths, view, rec) -> ThresholdItem:
    """The session's state by the runner's rule (view: runner.StageView), the same view for every session."""
    th = rec.stages.get("thresholds")
    suggested = (th.metrics.get("suggested") if th is not None else None) or None
    cfg = view.cfg(rec)  # None: the config cannot be read (view.problem says why)
    cnmf_cfg = cfg.get("cnmf") if cfg is not None else None
    chosen = _chosen(cnmf_cfg) if isinstance(cnmf_cfg, dict) else None
    waiting = actions.waits_for_thresholds(view, rec)
    first_stale = view.first_not_current(rec)
    problem = view.problem(rec)
    has_file = paths.threshold_maps(rec.sid).is_file()
    maps_current = view.current_through(rec, "thresholds")
    reason, note, text = "", "", ""
    if rec.excluded:
        state = "excluded"
        # the batch skips an excluded session: nothing will recompute its maps, or make them
        if not has_file:
            note = f"No maps. {EXCLUDED_NOTE}"
        elif problem is None and not maps_current:
            reason = "maps out of date"
            note = "The maps are out of date, and the session is excluded: the batch does not process it."
    elif problem is not None:
        # the batch cannot tell which stages are up to date either: it offers nothing until the config is fixed
        state, reason = "config_error", problem
        note = f"The session cannot be scheduled: {problem}"
    elif not has_file:
        state = "no_maps"
        if th is not None and th.state is State.FAILED:
            reason = last_line(th.error) or "failed"
        elif th is not None and th.state is State.DONE:
            reason = "maps file not found"
        else:
            reason = "not computed yet"
        cnmf = rec.stages.get("cnmf")
        if cnmf is not None and cnmf.state is State.DONE:
            reason += " (CNMF-E done)"
        note = f"No maps: {reason}"
    elif view.current_through(rec, "cnmf"):
        state = "cnmf_done"
    elif not maps_current:
        # The stage leaves the old file until it replaces it, and the runner would apply a choice written now to
        # maps nobody has seen: the maps are not shown and Accept is refused with the sentence (the author may
        # relax this later).
        state = "out_of_date"
        reason, note, recomputed = _stale(th, first_stale)
        if chosen is not None:
            text = "chosen · maps will be recomputed" if recomputed else f"chosen · {reason}"
        else:
            text = reason
    elif chosen is not None:
        state = "chosen"
    else:
        # waits_for_thresholds holds here (maps current, no thresholds, not excluded, config readable); the
        # fallback only guards a cnmf section that the hashes accept but the predicate cannot read
        state = "waiting" if waiting else "not_needed"
    if state in ("no_maps", "config_error"):
        text = f"{STATE_TEXT[state]}: {reason}"
    return ThresholdItem(rec.sid, rec.mouse, state, reason, suggested, chosen, waiting, first_stale, note, text)


def state_text(item: ThresholdItem) -> str:
    """What the session list says about the session."""
    return item.text or STATE_TEXT.get(item.state, item.state)


def threshold_items(paths, view=None, records=None) -> list[ThresholdItem]:
    """One item per session. `view` (runner.StageView) and `records` come from the caller when it has them (the
    tab's, with cached configs and the last good records); else they are built here: one view for all."""
    if records is None:
        records, _broken = registry.read_all(paths)
    view = view if view is not None else runner.StageView(paths)
    return [_item(paths, view, r) for r in sorted(records, key=lambda r: (r.mouse, r.sid))]


def initial_values(item: ThresholdItem, maps: ThresholdMaps, cfg: dict) -> tuple[int, float, float, str]:
    # a stored or suggested gSig without its maps (range changed since, or a damaged file) cannot be shown: the
    # fallback among the usable gSigs takes its place, the thresholds stay; missing_gsig_note says so
    usable = maps.usable_gsigs
    fb_g = config.fallback_gsig(usable)
    for src, v in (("chosen", item.chosen), ("suggested", item.suggested)):
        if v:
            g = int(v["gSig"])
            return (g if g in usable else fb_g), float(v["min_corr"]), float(v["min_pnr"]), src
    fb = cfg["thresholds"]["fallback"]
    return fb_g, float(fb["min_corr"]), float(fb["min_pnr"]), "fallback"


def missing_gsig_note(item: ThresholdItem, maps: ThresholdMaps, source: str) -> str:
    """Text for the window when the stored/suggested gSig has no map, else an empty string."""
    v = item.chosen if source == "chosen" else item.suggested if source == "suggested" else None
    usable = maps.usable_gsigs
    if v and int(v["gSig"]) not in usable:
        return f"gSig {int(v['gSig'])} has no map; showing {config.fallback_gsig(usable)}"
    return ""


def start_values(item: ThresholdItem, maps: ThresholdMaps, cfg: dict) -> tuple[int, float, float, str, str]:
    """(gSig, min_corr, min_pnr, source, note): initial_values on the usable gSigs, and the note when the stored
    or suggested gSig had to be swapped. The window's start and its Reset both take their values from here."""
    g, min_corr, min_pnr, source = initial_values(item, maps, cfg)
    return g, min_corr, min_pnr, source, missing_gsig_note(item, maps, source)


CORR_DECIMALS, PNR_DECIMALS = 2, 1  # the slider steps of the window: 0.01 and 0.1


def value_text(x, decimals: int) -> str:
    """x with the slider step's decimals, or with all of its own when it lies between steps (an untouched
    suggestion is written exactly, so the window shows it exactly)."""
    x = float(x)
    short = f"{x:.{decimals}f}"
    return short if float(short) == x else repr(x)


_LAYERS = ("session", "mouse", "series")  # closest first: a closer layer overrides the next


def choice_source(paths, sid: str, mouse: str) -> str:
    """The config layers the session's thresholds come from, closest first: "session config", "mouse and series
    config". "" when none of the three sets them, or when a layer cannot be read (nothing is claimed then)."""
    files = {"session": paths.session_config(sid), "mouse": paths.mouse_config(mouse), "series": paths.series_config}
    try:
        cnmf = {name: config.read_yaml(f).get("cnmf") for name, f in files.items()}
    except Exception:  # a broken layer: the tab shows "cannot be scheduled" for it anyway
        return ""
    found = set()
    for k in actions.THRESHOLD_KEYS:
        # the closest layer that has the key gives the resolved value (a null there means no choice)
        name = next((n for n in _LAYERS if isinstance(cnmf[n], dict) and k in cnmf[n]), None)
        if name is not None and cnmf[name][k] is not None:
            found.add(name)
    names = [n for n in _LAYERS if n in found]
    if not names:
        return ""
    return f"{', '.join(names[:-1])} and {names[-1]} config" if len(names) > 1 else f"{names[0]} config"


def source_text(source: str, item: ThresholdItem, chosen_from: str = "") -> str:
    """`chosen_from`: choice_source of the session, for an earlier choice."""
    if source == "chosen":
        return f"Your earlier choice ({chosen_from})" if chosen_from else "Your earlier choice"
    if source == "suggested" and item.suggested:
        s = item.suggested
        return (f"Autotune suggestion ({s.get('method', '?')}): gSig {int(s['gSig'])}, "
                f"min_corr {value_text(s['min_corr'], CORR_DECIMALS)}, "
                f"min_pnr {value_text(s['min_pnr'], PNR_DECIMALS)}")
    return "No suggestion — start values from the config"


def slider_range_pnr(maps: ThresholdMaps, g: int) -> float:
    v = maps.pnr[_known(maps, g)]
    v = v[np.isfinite(v)]
    return max(30.0, float(np.percentile(v, 99.9))) if v.size else 30.0
