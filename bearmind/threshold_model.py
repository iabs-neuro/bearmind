"""Model of the Thresholds tab without Qt: the ready maps of the thresholds stage, the passing mask,
blob counters and the list of sessions with their states."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from bearmind import actions, config, registry
from bearmind.registry import State

STATE_TEXT = {"waiting": "waiting", "chosen": "chosen", "cnmf_done": "CNMF-E done", "no_maps": "no maps",
              "excluded": "excluded", "not_needed": "thresholds not needed yet"}

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
    state: str          # "excluded" | "no_maps" | "cnmf_done" | "chosen" | "waiting" | "not_needed"
    reason: str         # for no_maps: last line of the stage error, or "not computed yet"
    suggested: dict | None
    chosen: dict | None  # {"gSig", "min_corr", "min_pnr"} from the resolved config
    # The batch waits for this session's thresholds: the rule of actions.waiting_threshold_sids. It is
    # independent of `state`, so a session without maps can wait and the window count equals Monitor's.
    waiting: bool = False


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


def _item(paths, rec) -> ThresholdItem:
    th = rec.stages.get("thresholds")
    cnmf = rec.stages.get("cnmf")
    suggested = (th.metrics.get("suggested") if th is not None else None) or None
    try:
        chosen = _chosen(config.resolve(paths, sid=rec.sid, mouse=rec.mouse)["cnmf"])
    except Exception:  # a broken config is shown by Monitor; here the session has no choice
        chosen = None
    waiting = not rec.excluded and (actions.waiting_for(rec, "cnmf") == "thresholds"
                                    or actions.suggested_unset(paths, rec))
    reason = ""
    if rec.excluded:
        state = "excluded"
    elif not paths.threshold_maps(rec.sid).is_file():
        state = "no_maps"
        if th is not None and th.state is State.FAILED:
            reason = last_line(th.error) or "failed"
        elif th is not None and th.state is State.DONE:
            reason = "maps file not found"
        else:
            reason = "not computed yet"
        if cnmf is not None and cnmf.state is State.DONE:
            reason += " (CNMF-E done)"
    elif cnmf is not None and cnmf.state is State.DONE:
        state = "cnmf_done"
    elif chosen is not None:
        state = "chosen"
    elif waiting:
        state = "waiting"
    else:
        state = "not_needed"  # no choice, and the batch does not wait for one (CNMF-E runs, failed or not reached)
    return ThresholdItem(rec.sid, rec.mouse, state, reason, suggested, chosen, waiting)


def threshold_items(paths) -> list[ThresholdItem]:
    recs, _broken = registry.read_all(paths)
    return [_item(paths, r) for r in sorted(recs, key=lambda r: (r.mouse, r.sid))]


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


def source_text(source: str, item: ThresholdItem) -> str:
    if source == "chosen":
        return "Your earlier choice"
    if source == "suggested" and item.suggested:
        s = item.suggested
        return (f"Autotune suggestion ({s.get('method', '?')}): gSig {int(s['gSig'])}, "
                f"min_corr {s['min_corr']:g}, min_pnr {float(s['min_pnr']):.1f}")
    return "No suggestion — start values from the config"


def slider_range_pnr(maps: ThresholdMaps, g: int) -> float:
    v = maps.pnr[_known(maps, g)]
    v = v[np.isfinite(v)]
    return max(30.0, float(np.percentile(v, 99.9))) if v.size else 30.0
