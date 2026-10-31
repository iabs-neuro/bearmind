"""Per-stage quality checks: warnings/errors recorded in metadata, never stopping the batch."""
import math
import statistics

from bearmind.registry import Check


def _fmt(x) -> str:
    return f"{x:g}"


def _level(value, warn, error, higher_is_worse: bool) -> str:
    bad = (lambda v, t: v > t) if higher_is_worse else (lambda v, t: v < t)
    if error is not None and bad(value, error):
        return "error"
    if warn is not None and bad(value, warn):
        return "warning"
    return "ok"


def _thresholds_text(th: dict, higher_is_worse: bool) -> str:
    op = ">" if higher_is_worse else "<"
    parts = [f"{k} {op} {_fmt(th[k])}" for k in ("warn", "error") if th.get(k) is not None]
    return ", ".join(parts)


def _absolute(stage, name, metrics, key, cfg, higher_is_worse, template):
    th = cfg.get("checks", {}).get(name)
    value = metrics.get(key)
    if th is None or value is None:
        return None
    if not math.isfinite(value):  # NaN compares False with every threshold and would grade "ok"
        return Check(stage, name, "warning", None, f"{key} is not a number")
    level = _level(value, th.get("warn"), th.get("error"), higher_is_worse)
    message = template.format(v=_fmt(value) if not isinstance(value, float) or key != "max_shift_px"
                              else f"{value:.1f}") + f" ({_thresholds_text(th, higher_is_worse)})"
    return Check(stage, name, level, float(value), message)


def _components_vs_median(metrics, series_metrics, cfg):
    checks_cfg = cfg.get("checks", {})
    th = checks_cfg.get("cnmf_n_components_vs_median")
    value = metrics.get("n_components")
    if th is None or value is None:
        return None
    values = [m["n_components"] for m in series_metrics if m.get("n_components") is not None]
    if len(series_metrics) < checks_cfg.get("min_sessions_for_relative", 3) or not values:
        return None
    median = statistics.median(values)
    if median <= 0:
        return None
    ratio = value / median
    level = _level(ratio, th.get("warn"), th.get("error"), True)
    message = (f"{value:g} components, {ratio:.1f}x series median ({median:g}) "
              f"({_thresholds_text(th, True)})")
    return Check("cnmf", "cnmf_n_components_vs_median", level, float(ratio), message)


def evaluate(stage: str, metrics: dict, series_metrics: list[dict], cfg: dict) -> list[Check]:
    out = []
    if stage == "mc":
        out.append(_absolute(stage, "mc_max_shift_px", metrics, "max_shift_px", cfg, True,
                             "max shift {v} px"))
        out.append(_absolute(stage, "mc_template_corr", metrics, "mean_template_corr", cfg, False,
                             "template corr {v}"))
    elif stage == "cnmf":
        out.append(_components_vs_median(metrics, series_metrics, cfg))
        out.append(_absolute(stage, "cnmf_missed_cells", metrics, "missed_cells", cfg, True,
                             "{v} missed cells"))
    elif stage == "autoinspect":
        out.append(_absolute(stage, "auto_deleted_frac", metrics, "deleted_frac", cfg, True,
                             "{v} of components deleted"))
    elif stage == "export":
        out.append(_absolute(stage, "export_n_neurons", metrics, "n_neurons", cfg, False,
                             "{v} neurons"))
        out.append(_absolute(stage, "export_median_snr", metrics, "median_snr", cfg, False,
                             "median SNR {v}"))
    return [c for c in out if c is not None]
