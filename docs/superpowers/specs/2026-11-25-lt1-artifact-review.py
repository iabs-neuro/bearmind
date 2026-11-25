"""Reproduce the wave and edge detector check on the current LT1 estimates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from caiman.utils.visualization import get_contours
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from bearmind import artifact_review as review, component_ids, fileio, registry
from bearmind.paths import SeriesPaths
from bearmind.stages import gate
from bearmind.timestamps import session_time_s


WAVE_MICE = {"J08", "J10", "J17", "J18"}
EXCLUDED_MICE = {"J26", "J30"}
WAVE_SESSIONS = {"LT1_J01_1D", "LT1_J01_4D"}
RETAINED_WAVES = {"LT1_J01_2D"}
EDGE_EXAMPLES = {"LT1_J03_2D", "LT1_J05_4D"}
WAVE_EXAMPLES = {"LT1_J08_1D", "LT1_J26_1D"}


def label(rec):
    mouse = rec.mouse.rsplit("_", 1)[-1]
    if mouse in EXCLUDED_MICE:
        return "excluded for waves"
    if mouse in WAVE_MICE or rec.sid in WAVE_SESSIONS:
        return "waves reported"
    if rec.sid in RETAINED_WAVES:
        return "rare waves left in"
    return "unlabelled"


def plot_contours(ax, est, ids, color, width=1.0):
    if len(ids):
        curves = get_contours(est.A[:, sorted(ids)], dims=est.imax.shape, thr=.3)
        lines = [c["coordinates"] for c in curves if len(c["coordinates"])]
        ax.add_collection(LineCollection(lines, colors=color, linewidths=width))


def save(fig, path):
    FigureCanvasAgg(fig)
    fileio.write_atomic(path, lambda f: fig.savefig(f, format="png", dpi=175, bbox_inches="tight"), binary=True)


def edge_example(figures, rec, est, ids, lit, activity, edges, removed):
    fig = Figure(figsize=(10.5, 5.2), layout="constrained")
    axes = fig.subplots(1, 2)
    for ax in axes:
        ax.imshow(est.imax, cmap="gray")
        ax.set_axis_off()
        plot_contours(ax, est, ids - edges - removed, "#99c2bd", .36)
    if lit is not None and np.any(lit) and not np.all(lit):
        axes[0].contour(lit, [.5], colors=["#33cbff"], linewidths=.9)
    if activity is not None and np.any(activity) and not np.all(activity):
        axes[0].contour(activity, [.5], colors=["#eaae70"], linewidths=.9)
    plot_contours(axes[0], est, edges, "#ff553e")
    plot_contours(axes[1], est, removed, "#bb73ff")
    plot_contours(axes[1], est, edges & removed, "#fff098")
    axes[0].set_title(f"Автопоиск края · {len(edges)}")
    axes[1].set_title(f"Ручное исключение · {len(removed)}, совпало {len(edges & removed)}")
    fig.suptitle(rec.sid)
    save(fig, figures / f"edge-{rec.sid}.png")


def wave_example(figures, rec, est, ids, times, events):
    if not events:
        return
    event = max(events, key=lambda e: e["duration_s"])
    highlighted = set(event["components"])
    fig = Figure(figsize=(10.5, 5.0), layout="constrained")
    image, trace = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1, 1.25]})
    image.imshow(est.imax, cmap="gray")
    image.set_axis_off()
    plot_contours(image, est, ids - highlighted, "#9cbdba", .36)
    plot_contours(image, est, highlighted, "#ffcf50", 1.2)
    image.set_title(f"Участвуют {len(highlighted)} компонент")
    a, b = event["first_frame"], event["last_frame"]
    left = np.searchsorted(times, times[a] - 2)
    right = np.searchsorted(times, times[b] + 2, side="right")
    trace.axvspan(0, event["duration_s"], alpha=.35, color="#ffce70")
    for j, idx in enumerate(sorted(highlighted)):
        v = est.C[idx, left:right].astype(float)
        trace.plot(times[left:right] - times[a], j + (v - v.min()) / max(np.ptp(v), 1e-6),
                   linewidth=.7, color="#354a63")
    trace.set(xlabel="Время от начала эпизода, с", ylabel="Трейсы C", yticks=[],
              title=f"Эпизод длится {event['duration_s']:.2f} с")
    fig.suptitle(rec.sid)
    save(fig, figures / f"wave-{rec.sid}.png")


def wave_overview(figures, rows):
    fig = Figure(figsize=(10.5, 5.0), layout="constrained")
    ax = fig.subplots()
    groups = [("waves reported", "Волны отмечены", "#b56920"),
              ("excluded for waves", "Мышь исключена", "#a43b56"),
              ("rare waves left in", "Редкие волны оставлены", "#47845b"),
              ("unlabelled", "Без пометки о волнах", "#738897")]
    labels = []
    for i, (state, title, color) in enumerate(groups):
        group = [r for r in rows if r["expert"] == state]
        for j, r in enumerate(group):
            y = 3 - i + ((j * 7) % 19 - 9) * .018
            ax.scatter(100 * r["wave_fraction"], y, s=25, color=color, alpha=.7)
        labels.append(f"{title} · {len(group)}")
    ax.set(yticks=[3, 2, 1, 0], yticklabels=labels, ylim=(-.5, 3.5),
           xlabel="Доля кадров в кандидатах на волны, %")
    ax.axvline(2.5, linestyle="--", color="#30363b", linewidth=1.0, label="Порог тяжёлой сессии")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=.2)
    save(fig, figures / "waves-overview.png")


def collect(paths, rec, figures):
    est = review._read(gate._estimates_file(paths, rec))
    ids = component_ids.selected(est)
    ordered = np.array(sorted(ids), dtype=int)
    spatial, centres = review._spatial(est, ordered)
    try:
        lit, activity = review._support(paths, rec, est.imax.shape)
    except FileNotFoundError:
        lit = activity = None
    times = session_time_s(paths, rec)
    waves = review.wave_candidates(est, ordered, centres, times, rec.fps)
    edges = review.edge_candidates(spatial, ordered, lit, activity)
    flagged = {e["component"] for e in edges or []}
    manual_path = paths.manual_result(rec.sid)
    kept = removed = unchanged_kept = changed_or_unmatched = None
    if manual_path.is_file():
        manual = review._read(manual_path)
        kept_ids = component_ids.selected(manual)
        kept = len(kept_ids)
        matches = component_ids.match(est, manual)
        unchanged_kept = {i for j, i in matches.items() if j in kept_ids and i in ids}
        removed = {i for j, i in matches.items() if j not in kept_ids and i in ids}
        changed_or_unmatched = ids - unchanged_kept - removed
        del manual
    if rec.sid in EDGE_EXAMPLES and removed is not None:
        edge_example(figures, rec, est, ids, lit, activity, flagged, removed)
    if rec.sid in WAVE_EXAMPLES:
        wave_example(figures, rec, est, ids, times, waves["events"])
    row = dict(sid=rec.sid, expert=label(rec), excluded=rec.excluded, cnmf_total=est.A.shape[1],
               cnmf_accepted=len(ids), manual_kept=kept,
               wave_events=len(waves["events"]), wave_fraction=waves["frame_fraction"], severe=waves["severe"],
               edge_ids=None if edges is None else len(flagged), activity_mask=activity is not None,
               manual_removed=None if removed is None else len(removed),
               manual_unmatched=None if changed_or_unmatched is None else len(changed_or_unmatched),
               edge_kept=None if unchanged_kept is None or edges is None else len(flagged & unchanged_kept),
               edge_removed=None if removed is None or edges is None else len(flagged & removed))
    details = dict(row, waves=waves["events"], edges=edges)
    if removed is not None:
        details["manual_removed_ids"] = sorted(removed)
        details["cnmf_rejected_ids"] = sorted(set(range(est.A.shape[1])) - ids)
    return row, details


def main(root, output):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    rows, details, errors = [], [], []
    for rec in registry.all_records(SeriesPaths(root)):
        try:
            row, detail = collect(SeriesPaths(root), rec, figures)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            print(f"[FAILED] {rec.sid}: {ascii(str(exc))}", flush=True)
            errors.append({"sid": rec.sid, "error": str(exc)})
            continue
        rows.append(row)
        details.append(detail)
        print(f"[OK] {rec.sid}: {row['wave_events']} wave candidates, {row['edge_ids']} edge candidates", flush=True)
    write_summary(output, rows, details, errors)
    return int(bool(errors))


def write_summary(output, rows, details, errors):
    if rows:
        wave_overview(output / "figures", rows)

        def write_rows(stream):
            writer = csv.DictWriter(stream, rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        fileio.write_atomic(output / "facts.csv", write_rows, newline="")
    fileio.write_atomic(output / "candidates.json", lambda f: json.dump(
        {"sessions": details, "errors": errors}, f, indent=2))


def update_annotations(root, output):
    data = json.loads((output / "candidates.json").read_text(encoding="utf-8"))
    labels = {r.sid: label(r) for r in registry.all_records(SeriesPaths(root))}
    rows = []
    for detail in data["sessions"]:
        detail["expert"] = labels[detail["sid"]]
        rows.append({k: v for k, v in detail.items()
                     if k not in {"waves", "edges", "manual_removed_ids", "cnmf_rejected_ids"}})
    write_summary(output, rows, data["sessions"], data["errors"])
    return int(bool(data["errors"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("series", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--annotations-only", action="store_true")
    args = parser.parse_args()
    run = update_annotations if args.annotations_only else main
    raise SystemExit(run(args.series, args.output))
