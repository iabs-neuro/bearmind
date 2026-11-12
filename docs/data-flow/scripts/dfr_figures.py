"""Figures of the data-flow page (docs/data-flow/img). Read-only on H:; single frames and small files only.

Run with the bearmind env python. Console output ASCII. Each figure is independent: a failure is reported and the
others are still made.
"""
import json
import pickle
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind import threshold_model  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import load_record  # noqa: E402
from bearmind.timestamps import session_timestamps  # noqa: E402
from bearmind.video import VideoArray  # noqa: E402

OUT = Path(r"C:\Users\User\PycharmProjects\bearmind\docs\data-flow\img")
OUT.mkdir(parents=True, exist_ok=True)
SCR = Path(__file__).parent
LT1 = SeriesPaths(r"H:\Projects\LT1\CalciumData")
TEST = SeriesPaths(r"H:\Projects\_bearmind_test\lt1_J03_J06")

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
CRIT, WARN, MUTED, INK = "#d03b3b", "#c98500", "#898781", "#0b0b0b"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110, "savefig.dpi": 110,
                     "axes.titlesize": 9.5, "axes.titleweight": "bold"})
made = {}


def save(fig, name):
    p = OUT / name
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    made[name] = p.stat().st_size
    print(f"[OK] {name} {p.stat().st_size // 1024} KB")


def fig_crop_overview():
    im = Image.open(LT1.crop_overview).convert("RGB")
    w, h = im.size
    top = im.crop((0, 0, w, min(h, 70 + 4 * 325)))
    top.thumbnail((1400, 4000))
    top.save(OUT / "crop_overview_lt1_top.png", optimize=True)
    t = Image.open(TEST.crop_overview).convert("RGB")
    t.thumbnail((1400, 4000))
    t.save(OUT / "crop_overview_test.png", optimize=True)
    for n in ("crop_overview_lt1_top.png", "crop_overview_test.png"):
        made[n] = (OUT / n).stat().st_size
        print(f"[OK] {n}")


def fig_frames():
    """Raw frame with lit area / field / box, the MC frame at the same index, a dF/F preview frame."""
    sid, idx = "LT1_J01_1D", 20000
    rec = load_record(LT1, sid)
    m = rec.stages["crop"].metrics
    part = rec.parts[0]
    raw = VideoArray(LT1.root / part.folder / f"{sid}_20.avi")[0]  # chunks of 1000 frames: frame 20000
    mc = VideoArray(LT1.mc_video(sid))[idx]
    import av
    with av.open(str(LT1.preview(sid, "mc_dff"))) as c:
        s = c.streams.video[0]
        dff = None
        k = idx // 2  # tbin 2: one preview frame per two movie frames
        rate, tb = float(s.average_rate), float(s.time_base)
        target = int(round(k / rate / tb))
        c.seek(target, stream=s, backward=True)
        for fr in c.decode(s):
            if fr.pts is not None and fr.pts >= target:
                dff = fr.to_ndarray(format="rgb24")
                break
    fig = plt.figure(figsize=(12, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], width_ratios=[608, 512])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(raw, cmap="gray", vmin=np.percentile(raw, 1), vmax=np.percentile(raw, 99.7))
    for key, col, ls, lab in (("fov_lit", "#ffd400", (0, (6, 3)), "освещённая область"),
                              ("fov", "#2ecc40", (0, (2, 2)), "поле активности"),
                              ("box", "#00d5ff", "-", "рамка мыши (кроп)")):
        t_, l_, h_, w_ = m[key]
        ax.add_patch(Rectangle((l_ - .5, t_ - .5), w_, h_, fill=False, ec=col, ls=ls, lw=1.6, label=lab))
    ax.set_title(f"{sid}: сырой кадр {idx} (608 x 608)")
    ax.legend(loc="lower right", fontsize=7, framealpha=.85)
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(mc, cmap="gray", vmin=np.percentile(mc, 1), vmax=np.percentile(mc, 99.7))
    ax.set_title(f"тот же кадр в 3_MC/{sid}_MC.avi ({mc.shape[0]} x {mc.shape[1]})")
    ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    if dff is not None:
        ax.imshow(dff)
        ax.set_title(f"Preview/{sid}_mc_dff.mp4, кадр {idx // 2} (среднее кадров {idx}-{idx + 1}): raw | dF/F | high-pass")
    ax.axis("off")
    save(fig, "frames_LT1_J01_1D.png")


def fig_mc():
    sid = "LT1_J01_1D"
    rec = load_record(LT1, sid)
    sh = np.load(LT1.mc_shifts(sid))
    t_min = np.arange(len(sh)) / rec.fps / 60
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw={"width_ratios": [3, 1.2, 1.2]})
    ax = axs[0]
    ax.plot(t_min, sh[:, 0], lw=.5, color=BLUE, label="сдвиг по строкам (y)")
    ax.plot(t_min, sh[:, 1], lw=.5, color=ORANGE, label="сдвиг по столбцам (x)")
    ax.axhline(0, color="#c3c2b7", lw=.8)
    ax.set_xlabel("время, мин (кадр / fps)")
    ax.set_ylabel("сдвиг, px")
    ax.set_title(f"{sid}: жёсткие сдвиги MC, {len(sh)} кадров; max |сдвиг| = {np.abs(sh).max():.1f} px")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    d = json.loads((SCR / "dfr_sessions.json").read_text(encoding="utf-8"))["rows"]
    ms = np.array([r["mc_max_shift_px"] for r in d if r["mc_max_shift_px"] is not None])
    tc = np.array([r["mc_template_corr"] for r in d if r["mc_template_corr"] is not None])
    ax = axs[1]
    ax.hist(ms, bins=np.arange(0, max(12, ms.max() + 1), 1.0), color=BLUE, ec="white")
    ax.axvline(10, color=WARN, lw=1.2, ls="--")
    ax.text(10, ax.get_ylim()[1] * .9, " WARN > 10", color=WARN, fontsize=7)
    ax.set_xlabel("max |сдвиг|, px")
    ax.set_ylabel("сессий")
    ax.set_title(f"LT1, {len(ms)} сессий: max сдвиг")
    ax = axs[2]
    ax.hist(tc, bins=np.arange(0.5, 1.001, 0.02), color=AQUA, ec="white")
    ax.axvline(0.7, color=WARN, lw=1.2, ls="--")
    ax.text(0.7, ax.get_ylim()[1] * .9, " WARN < 0.7", color=WARN, fontsize=7)
    ax.set_xlabel("средняя корреляция с шаблоном")
    ax.set_title(f"LT1: корреляция с шаблоном")
    fig.tight_layout()
    save(fig, "mc_shifts_LT1_J01_1D.png")
    print(f"   mc: n={len(ms)} max shift range {ms.min():.1f}-{ms.max():.1f}; corr range {tc.min():.3f}-{tc.max():.3f}; "
          f"corr<0.7: {(tc < 0.7).sum()}; shift>10: {(ms > 10).sum()}")


def fig_thresholds():
    sid, g, c0, p0 = "LT1_J01_1D", 4, 0.92, 19.0
    maps = threshold_model.load_maps(LT1.threshold_maps(sid))
    cnt = threshold_model.counters(maps, g, c0, p0, 2)
    print(f"   thresholds counters {sid}: blobs {cnt.n_blobs}, single {cnt.single_frac:.3f}, mirror {cnt.n_mirror_blobs}")
    fig, axs = plt.subplots(2, 3, figsize=(12.5, 7))
    cn, pnr, cnm, pnrm = maps.cn[g], maps.pnr[g], maps.cn_mirror[g], maps.pnr_mirror[g]
    pmax = np.percentile(pnr, 99.5)

    def show(ax, img, title, vmin, vmax, cmap):
        h = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(h, ax=ax, fraction=.035, pad=.02)

    show(axs[0, 0], cn, f"cn_{g}: локальная корреляция (фильтр gSig={g})", 0.3, 1.0, "viridis")
    show(axs[0, 1], pnr, f"pnr_{g}: пик / шум", 0, pmax, "magma")
    mask = threshold_model.passing_mask(cn, pnr, c0, p0)
    axs[0, 2].imshow(maps.mean, cmap="gray")
    ov = np.zeros(mask.shape + (4,))
    ov[mask] = (0.0, 0.84, 1.0, 1.0)
    axs[0, 2].imshow(ov)
    axs[0, 2].set_title(f"проходят cn >= {c0} и pnr >= {p0}: {cnt.n_blobs} пятен")
    axs[0, 2].axis("off")
    show(axs[1, 0], cnm, f"cn_mirror_{g}: то же на инвертированном видео", 0.3, 1.0, "viridis")
    show(axs[1, 1], pnrm, f"pnr_mirror_{g}", 0, pmax, "magma")
    mm = threshold_model.passing_mask(cnm, pnrm, c0, p0)
    axs[1, 2].imshow(maps.mean, cmap="gray")
    ov = np.zeros(mm.shape + (4,))
    ov[mm] = (1.0, 0.25, 0.25, 1.0)
    axs[1, 2].imshow(ov)
    axs[1, 2].set_title(f"инвертированное видео, те же пороги: {cnt.n_mirror_blobs} пятен (шум)")
    axs[1, 2].axis("off")
    fig.suptitle(f"{sid}: карты порогов (_bearmind/thresholds/{sid}_maps.npz), выбор gSig={g}, min_corr={c0}, "
                 f"min_pnr={p0}", fontsize=10)
    fig.tight_layout()
    save(fig, "threshold_maps_LT1_J01_1D.png")


def _contours(A, dims, idx, thr=0.35):
    out = []
    A = A.tocsc()
    for i in idx:
        col = A[:, i].toarray().ravel()
        if col.max() <= 0:
            out.append(None)
            continue
        out.append(col.reshape(dims, order="F") / col.max())
    return out


def fig_cnmf():
    sid = "LT1_J03_1D"
    rec = load_record(TEST, sid)
    with open(TEST.estimates(sid), "rb") as f:
        est = pickle.load(f)
    dims = est.imax.shape
    good = np.asarray(est.idx_components, int)
    bad = np.asarray(est.idx_components_bad, int)
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.5])
    ax = fig.add_subplot(gs[0])
    ax.imshow(est.imax, cmap="gray")
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    for idx, col in ((bad, ORANGE), (good, "#00d5ff")):
        for a in _contours(est.A, dims, idx):
            if a is not None:
                ax.contour(a, levels=[0.35], colors=[col], linewidths=.6)
    ax.set_title(f"{sid}: imax; контуры {len(good)} принятых (голубые), {len(bad)} отброшенных (оранж.)")
    ax.axis("off")
    ax = fig.add_subplot(gs[1])
    t = np.asarray(session_timestamps(TEST, rec).t_ms, float)
    t = (t - t[0]) / 1000
    snr = np.asarray(est.SNR_comp)
    pick = good[np.argsort(-snr[good])][:8]
    for k, i in enumerate(pick):
        c = est.C[i]
        ax.plot(t, k * 1.1 + (c - c.min()) / (np.ptp(c) or 1), lw=.5, color=BLUE)
        ax.text(t[-1] + 5, k * 1.1 + .3, f"#{i} SNR {snr[i]:.1f}", fontsize=7, color="#52514e")
    ax.set_xlabel("время, с (таймстемпы сессии)")
    ax.set_yticks([])
    ax.set_title("C: трейсы 8 принятых компонент с наибольшим SNR (нормированы)")
    fig.tight_layout()
    save(fig, "cnmf_LT1_J03_1D.png")
    del est


def fig_residual():
    im = Image.open(TEST.residual_png("LT1_J03_1D")).convert("RGB")
    im.thumbnail((1400, 1400))
    im.save(OUT / "residual_LT1_J03_1D.png", optimize=True)
    made["residual_LT1_J03_1D.png"] = (OUT / "residual_LT1_J03_1D.png").stat().st_size
    print("[OK] residual_LT1_J03_1D.png")


def fig_export():
    sid = "LT1_J03_1D"
    from bearmind.timestamps import read_timestamps
    with np.load(TEST.export_npz(sid)) as z:
        C = z["C"]
        fps = float(z["fps"])
        idx = z["component_indices"]
    ts = read_timestamps(TEST.export_timestamps(sid))
    t = (ts.t_ms - ts.t_ms[0]) / 1000
    Z = (C - C.min(1, keepdims=True)) / np.maximum(np.ptp(C, 1, keepdims=True), 1e-9)
    order = np.argsort(-Z.mean(1))
    fig, axs = plt.subplots(2, 1, figsize=(13, 5.6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    axs[0].imshow(Z[order], aspect="auto", cmap="Blues", extent=[t[0], t[-1], len(Z), 0], interpolation="nearest")
    axs[0].set_ylabel("нейрон (по средней активности)")
    axs[0].set_title(f"6_Export/npz/{sid}_data.npz: C {C.shape[0]} нейронов x {C.shape[1]} кадров, "
                     f"каждый ряд нормирован 0..1; ось времени из {sid}_timestamps.csv")
    d = np.diff(ts.t_ms)
    axs[1].plot(t[1:], d, lw=.4, color=BLUE)
    axs[1].axhline(1000 / fps, color=ORANGE, lw=1, ls="--")
    axs[1].text(t[-1] * 0.99, 80, f"пунктир: 1000/fps = {1000 / fps:.2f} мс", color=ORANGE, fontsize=7.5,
                va="top", ha="right")
    axs[1].set_ylabel("интервал, мс")
    axs[1].set_xlabel("время, с")
    fig.tight_layout()
    save(fig, "export_LT1_J03_1D.png")


def fig_intervals():
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.4))
    for ax, sid, col in ((axs[0], "LT1_J10_4D", BLUE), (axs[1], "LT1_J27_3D", AQUA)):
        rec = load_record(LT1, sid)
        t = np.asarray(session_timestamps(LT1, rec).t_ms, float)
        d = np.diff(t)
        m = np.median(d)
        r = d / m
        ax.hist(np.clip(r, 0, 3), bins=np.arange(0, 3.02, 0.02), color=col, ec="none")
        ax.set_yscale("log")
        for x, lab, c in ((0.5, "0.5 m", MUTED), (1.5, "1.5 m", MUTED), (2.0, "порог разрыва 2.0 m", CRIT)):
            ax.axvline(x, color=c, lw=1, ls="--")
            ax.text(x, ax.get_ylim()[1] * 0.5, " " + lab, color=c, fontsize=7, rotation=90, va="top")
        ng = int((d > 2 * m).sum())
        ax.set_title(f"{sid}: {len(d)} интервалов, медиана m = {m:.3f} мс; > 2 m: {ng}; fps {rec.fps:.4f}")
        ax.set_xlabel("интервал / медиана (значения > 3 собраны в последний столбец)")
        ax.set_ylabel("число интервалов (лог)")
    fig.tight_layout()
    save(fig, "intervals_LT1.png")


def fig_fps_bias():
    d = json.loads((SCR / "dfr_sessions.json").read_text(encoding="utf-8"))["rows"]
    rs = [r for r in d if r["fps"] > 25]
    ng = np.array([r["n_gaps"] for r in rs])
    f = np.array([r["fps"] for r in rs])
    fs = np.array([r["fps_naive_span"] for r in rs])
    dur = np.array([r["span_s"] for r in rs]) / 60
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.9))
    ax = axs[0]
    for sel, mk, lab in ((dur > 20, "o", "40 мин"), (dur <= 20, "s", "10 мин")):
        ax.scatter(ng[sel], f[sel], s=24, marker=mk, color=BLUE, ec="white", lw=.6, label=f"robust_fps (registry), {lab}")
        ax.scatter(ng[sel], fs[sel], s=24, marker=mk, color=ORANGE, ec="white", lw=.6,
                   label=f"(N-1) / (t_last - t_first), {lab}")
    ax.set_xlabel("число «разрывов» в сессии (интервал > 2 x медиана)")
    ax.set_ylabel("частота кадров, Гц")
    ax.set_title(f"A. {len(rs)} сессий LT1 ~29.78 Гц: две оценки частоты")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax = axs[1]
    gw = json.loads((SCR / "dfr_gapwindow.json").read_text(encoding="utf-8"))
    x = np.array([e[2] for e in gw["res"]["30"]])
    c = np.array(gw["ctrl"]["30"])
    bins = np.arange(-1, 4.6, 0.1)
    ax.hist(np.clip(c, -1, 4.5), bins=bins, color="#c3c2b7", ec="white", lw=.3, density=True,
            label=f"контроль: {len(c)} окон без разрыва")
    ax.hist(np.clip(x, -1, 4.5), bins=bins, color=BLUE, ec="white", lw=.3, density=True, alpha=.85,
            label=f"{len(x)} окон вокруг разрыва")
    ax.axvline(1, color=CRIT, ls="--", lw=1)
    ax.text(1.05, ax.get_ylim()[1] * .6, "1 потерянный кадр", color=CRIT, fontsize=7)
    ax.set_xlabel("лишнее время в окне +-30 интервалов, в периодах кадра")
    ax.set_ylabel("доля (плотность)")
    ax.set_title(f"B. < 0.5 кадра: {(x < 0.5).sum()}; 0.5-1.5: {((x >= 0.5) & (x < 1.5)).sum()}; "
                 f">= 1.5: {(x >= 1.5).sum()}")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fps_bias_LT1.png")


SEL = sys.argv[1:]
for fn in (fig_crop_overview, fig_frames, fig_mc, fig_thresholds, fig_cnmf, fig_residual, fig_export,
           fig_intervals, fig_fps_bias):
    if SEL and fn.__name__ not in SEL:
        continue
    try:
        fn()
    except Exception:
        print(f"[FAILED] {fn.__name__}")
        traceback.print_exc()
print("[OK] done:", ", ".join(sorted(made)))
