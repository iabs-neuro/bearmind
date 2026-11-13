"""Build report-lt1-actions-2026-11-13.html from batch_snapshot.json (collect.py), dropped_after.json
(dry_rescan.py), page_template.html and gpu_section.html. Also writes facts.md: every number on the page with
what it counts and where it is read from.
"""
import json
import statistics as st
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).parent
NAME = "report-lt1-actions-2026-11-13"
NEW_CODE = datetime(2026, 11, 13, 15, 57, 0)   # batch restarted on the merged maps speedup (989c88b)
CLEAN = datetime(2026, 11, 13, 18, 10, 0)      # the CNMF-E tsub experiment ended; later runs had the machine alone
GH = "https://github.com/iabs-neuro/bearmind/blob/bearmind-batch/"
SPECS = "docs/superpowers/specs/"
SDD = "../../../../.superpowers/sdd/2026-11-12-lt1-fixes/"


def long_(r):
    return r["minutes"] > 30


def at_edge(s):
    """mirror_fdr found no point inside its grid (corr 0.5-0.98 x pnr 2-20, bm_qc.py:375-376): its suggestion is on
    the grid border, or it gave none and the stage fell back to robust_bg (stages/thresholds.py _suggest)."""
    if s["method"] == "mirror_fdr":
        return s["min_corr"] >= 0.98 or s["min_pnr"] >= 20.0 or s["min_corr"] <= 0.5 or s["min_pnr"] <= 2.0
    return True


def compact(sids):
    """LT1_J01_1D, LT1_J01_2D, LT1_J03_1D -> LT1_J01 1D-2D, LT1_J03 1D (days listed as given)."""
    by = {}
    for sid in sids:
        m, day = sid.rsplit("_", 1)
        by.setdefault(m, []).append(day)
    return "; ".join(m + " " + (d[0] if len(d) == 1 else d[0] + "–" + d[-1]) for m, d in by.items())


def link(md_rel=None, local=None, label=None):
    parts = []
    if md_rel:
        parts.append(f'<a href="{GH}{md_rel}">GitHub</a>')
        parts.append(f'<a href="{"../../../../" + md_rel}">файл</a>')
    if local:
        parts.append(f'<a href="{local}">{label or "файл"}</a>')
    return " · ".join(parts)


def main():
    snap = json.loads((HERE / "batch_snapshot.json").read_text(encoding="utf-8"))
    rows = snap["sessions"]
    now = datetime.fromisoformat(snap["snapshot"])
    done = [r for r in rows if r["thresholds_state"] == "done"]
    running = [r for r in rows if r["thresholds_state"] == "running"]
    left = [r for r in rows if r["thresholds_state"] != "done"]

    def rate(r):
        return r["thresholds_min"] / r["mpx_mframes"]

    clean = [r for r in done if r["maps_code"] == "new" and datetime.fromisoformat(r["thresholds_started"]) >= CLEAN]
    rates = {g: [rate(r) for r in clean if long_(r) == g] for g in (False, True)}
    med = {g: st.median(v) for g, v in rates.items()}

    def project(pick):
        t = datetime.fromisoformat(running[0]["thresholds_started"]) if running else now
        pts = []
        for r in sorted(left, key=lambda r: r["sid"]):  # the runner takes sessions in sid order
            t = t + timedelta(minutes=pick[long_(r)] * r["mpx_mframes"])
            if r["thresholds_state"] == "running" and t < now:
                t = now
            pts.append({"sid": r["sid"], "t": t.isoformat(timespec="seconds")})
        return pts

    proj = project(med)
    lo = project({g: min(v) for g, v in rates.items()})
    hi = project({g: max(v) for g, v in rates.items()})
    eta, eta_lo, eta_hi = (datetime.fromisoformat(p[-1]["t"]) for p in (proj, lo, hi))

    def med_min(code, g):
        return st.median([r["thresholds_min"] for r in done if r["maps_code"] == code and long_(r) == g])

    sugg = [r["suggested"] for r in done if r["suggested"]]
    edge_n = sum(at_edge(s) for s in sugg)
    edge_fallback = sum(s["method"] != "mirror_fdr" for s in sugg)
    chosen = sorted(r["sid"] for r in rows if r.get("thresholds_chosen"))

    drop = json.loads((HERE / "dropped_after.json").read_text(encoding="utf-8"))
    df_after_s = len(drop)
    df_after_f = sum(d["data"]["n_dropped"] for d in drop)
    cert_f = sum(d["data"]["frames_certain"] for d in drop)
    cert_s = sum(d["data"]["frames_certain"] > 0 for d in drop)
    df_before_s, df_before_f = 55, 334  # registry before the rescan: see facts.md

    hm = lambda t: t.strftime("%H:%M")

    def hm10(t):  # a forecast shown to the minute would claim precision it does not have
        t = t + timedelta(minutes=5)
        return t.replace(minute=t.minute // 10 * 10, second=0, microsecond=0).strftime("%H:%M")
    gpu = (HERE / "gpu_section.html").read_text(encoding="utf-8") if (HERE / "gpu_section.html").exists() else ""
    g_short, g_rec, g_sec = "разведка ещё идёт.", "Решение — после разведки.", "<div class='card'><p>Разведка ещё идёт.</p></div>"
    if gpu:
        g_short, g_rec, g_sec = gpu.split("\n@@\n", 2)

    pipeline = [
        {"name": "Скан", "done": 84, "total": 84, "note": "Rescan — Д2"},
        {"name": "Кроп", "done": sum(r["crop_state"] == "done" for r in rows), "total": 84},
        {"name": "MC", "done": sum(r["mc_state"] == "done" for r in rows), "total": 84},
        {"name": "Карты порогов", "done": len(done), "total": 84, "now": True, "note": f"≈ {hm10(eta)} все"},
        {"name": "Вы: пороги", "you": True, "note": f"выбрано {len(chosen)} из 84"},
        {"name": "CNMF-E", "done": sum(r["cnmf_state"] == "done" for r in rows), "total": 84},
        {"name": "Вы: инспекция", "you": True, "note": "ручная или «не нужна»"},
        {"name": "Автоинспекция", "done": 0, "total": 84},
        {"name": "Экспорт", "done": 0, "total": 84},
    ]
    keep = ("sid", "minutes", "n_frames", "box", "thresholds_started", "thresholds_finished", "thresholds_min", "maps_code")
    data = {
        "sessions": [{k: r[k] for k in keep} for r in rows],
        "projection": proj, "pipeline": pipeline,
        "t_first": min(r["thresholds_started"] for r in done),
        "snapshot_iso": now.isoformat(timespec="seconds"), "new_code_iso": NEW_CODE.isoformat(),
        "eta_iso": eta.isoformat(timespec="seconds"), "eta_hi_iso": eta_hi.isoformat(timespec="seconds"),
        "eta": hm10(eta),
    }
    tok = {
        "snapshot": now.strftime("%d.%m.%Y %H:%M"), "snapshot_hm": hm(now),
        "maps_done": len(done), "maps_left": len(left), "running_sid": running[0]["sid"] if running else "—",
        "eta": hm10(eta), "eta_lo": hm10(eta_lo), "eta_hi": hm10(eta_hi),
        "df_before_s": df_before_s, "df_after_s": df_after_s, "df_before_f": df_before_f, "df_after_f": df_after_f,
        "cert_f": cert_f, "cert_s": cert_s, "edge_n": edge_n, "sugg_n": len(sugg),
        "edge_mirror": edge_n - edge_fallback, "edge_fallback": edge_fallback,
        "chosen_n": len(chosen), "unchosen_n": 84 - len(chosen), "chosen_list": compact(chosen),
        "old_long": f"{med_min('old', True):.0f}", "new_long": f"{med_min('new', True):.0f}",
        "old_short": f"{med_min('old', False):.1f}".replace(".", ","),
        "new_short": f"{med_min('new', False):.1f}".replace(".", ","),
        "gpu_short": g_short, "gpu_rec": g_rec, "gpu_section": g_sec,
        "L_speedup": link(local=SDD + "maps-speedup-report.md", label="отчёт") + " · " +
                     link(local=SDD + "maps-speedup-fix1-report.md", label="проверка на LT1_J01_1D") + " <span>(локально)</span>",
        "L_subsample": link(SPECS + "2026-11-12-maps-subsample-validation.md"),
        "L_sources": link(SPECS + "2026-11-12-caiman-tsub-thresholds-sources.md"),
        "L_tsub": link(SPECS + "2026-11-13-cnmf-tsub-comparison.md"),
        "L_scan1": link(local=SDD + "scan-step1-report.md", label="отчёт (локально)") + " · руководство студента: " +
                   link("docs/students/series-workflow.md"),
        "L_dataflow": link(local="../../../data-flow/index.html", label="открыть страницу"),
        "L_design": link(SPECS + "2026-11-12-scan-approval-design.md"),
        "L_feedback": link(SPECS + "2026-11-11-lt1-feedback.md"),
        "L_ml": link(SPECS + "2026-11-09-ml-retrain-recon.md"),
    }
    html = (HERE / "page_template.html").read_text(encoding="utf-8")
    html = html.replace("/*DATA*/", json.dumps(data, ensure_ascii=False))
    for k, v in tok.items():
        html = html.replace("{{" + k + "}}", str(v))
    assert "{{" not in html, html[html.index("{{"):html.index("{{") + 40]
    (HERE / f"{NAME}.html").write_text(html, encoding="utf-8")

    facts = f"""# Fact sheet: {NAME}.html

Snapshot {now.isoformat(timespec='seconds')}. Each line: claim -> what is counted -> source.

- maps done {len(done)} / 84, running {tok['running_sid']}, left {len(left)} -> registry stages.thresholds.state over 84 records -> batch_snapshot.json (collect.py reads H:/Projects/LT1/CalciumData/_bearmind/registry/*.json)
- ETA {eta:%d.%m %H:%M} (range {eta_lo:%H:%M}-{eta_hi:%H:%M}) -> derived: remaining sessions in sid order, minutes = rate x box_w x box_h x n_frames / 1e12; rate = median (min/max for the range) of new-code runs started >= {CLEAN:%H:%M} 13.11, separately for 40-min ({len(rates[True])} runs, median {med[True]:.0f}) and 10-min ({len(rates[False])} runs, median {med[False]:.0f}) sessions; box from stages.mc.params.box
- thresholds stage minutes, median: 40-min old {med_min('old', True):.1f}, new {med_min('new', True):.1f}; 10-min old {med_min('old', False):.1f}, new {med_min('new', False):.1f} -> finished - started of stages.thresholds; old = started before 13.11 15:57 -> batch_snapshot.json
- no point inside the grid {edge_n} / {len(sugg)} (mirror_fdr on the border {edge_n - edge_fallback}, robust_bg fallback {edge_fallback}) -> stages.thresholds.metrics.suggested of done sessions; rule at_edge() -> batch_snapshot.json
- thresholds already chosen: {len(chosen)} sessions ({", ".join(chosen)}) -> config/sessions/<SID>.yaml has cnmf.gSig/min_corr/min_pnr -> batch_snapshot.json thresholds_chosen; same rows in _bearmind/threshold_choices.csv
- dropped_frames before rescan: 55 sessions, 334 frames -> registry anomalies code dropped_frames, sum data.n_dropped, read 13.11 ~20:25
- dropped_frames after rescan: {df_after_s} sessions, {df_after_f} frames; certain {cert_f} frames in {cert_s} sessions -> dry_rescan.py (scan_series with writes captured in memory, merged code d5f4d03) -> dropped_after.json
- dry rescan changes parts/n_frames/fps/mode: none; excluded flips: none -> dry_rescan.py console output
- window started 17:15, scan merge 18:16 -> Get-Process 31856 StartTime; git log d5f4d03 (2026-11-13 18:16:35)
- tsub: TK loses 9-22 % of R1 accepted cells; fit 1.4-2.1x faster -> docs/superpowers/specs/2026-11-13-cnmf-tsub-comparison.md lines 7, 11
- tsub: J03_2D 91 %, J06_2D 78 % of R1 accepted kept -> same file line 8; cnmf_tsub_components.csv (148/162, 216/277)
- subsample maps: 17 sessions, min_pnr median -9.5, seeds x1.63 -> 2026-11-11-lt1-feedback.md item 14; 2026-11-12-maps-subsample-validation.md
- speedup equality <= 1e-4 in the lit field (J01_1D, gSig 7) -> .superpowers/sdd/2026-11-12-lt1-fixes/maps-speedup-fix1-report.md; feedback item 5
- item 17: frame 0 black in 10 sessions, 5 ERROR -> feedback item 17
- item 22: template corr 0.7 flags 21 / 84; est.time 0-592 s vs 599.56 s -> feedback item 22
- design rule flags 4 LT1 sessions -> 2026-11-12-scan-approval-design.md line 129
- R8: 298 vs 300 components, R1 at pool 31 ~15 min, ~110 GB free -> 2026-11-13-cnmf-tsub-comparison.md line 17
- GPU: 5 of 32 logical cores busy; H: 3.9 GB/s, ~2 GB per session; prototype 0.31-0.34 ms per frame per gSig, 4.3 GiB VRAM, 3 runs bit-identical; stage 32 -> ~3 min (est); 7 strip workers 1.4-1.7x -> .superpowers/sdd/2026-11-12-lt1-fixes/gpu-recon-report.md "Bottom line" 1-4, section 2.4-2.5
- GPU: use_cuda is a no-op -> caiman/motion_correction.py:204-205 (warning "cuda is no longer supported")
- LT1 hours by stage: crop 1.5, MC 13.3, thresholds 18-28, CNMF-E 75-107 (model), autoinspect 15 -> gpu-recon-report.md section 3.1
- AUC 0.978 vs held-out 0.9655 -> 2026-11-09-ml-retrain-recon.md; production_models/ebm_v9_iter8_summary.json test_metrics
"""
    (HERE / "facts.md").write_text(facts, encoding="utf-8")
    print("[OK]", HERE / f"{NAME}.html")
    print("ETA", eta, eta_lo, eta_hi, "edge", edge_n, len(sugg), "rates", {k: round(v) for k, v in med.items()})


if __name__ == "__main__":
    main()
