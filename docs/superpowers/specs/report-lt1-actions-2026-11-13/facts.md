# Fact sheet: report-lt1-actions-2026-11-13.html

Snapshot 2026-11-13T20:44:59. Each line: claim -> what is counted -> source.

- maps done 47 / 84, running LT1_J26_4D, left 37 -> registry stages.thresholds.state over 84 records -> batch_snapshot.json (collect.py reads H:/Projects/LT1/CalciumData/_bearmind/registry/*.json)
- ETA 14.11 02:47 (range 02:30-03:16) -> derived: remaining sessions in sid order, minutes = rate x box_w x box_h x n_frames / 1e12; rate = median (min/max for the range) of new-code runs started >= 18:10 13.11, separately for 40-min (5 runs, median 1481) and 10-min (5 runs, median 1169) sessions; box from stages.mc.params.box
- thresholds stage minutes, median: 40-min old 65.9, new 30.8; 10-min old 6.9, new 4.9 -> finished - started of stages.thresholds; old = started before 13.11 15:57 -> batch_snapshot.json
- no point inside the grid 31 / 47 (mirror_fdr on the border 20, robust_bg fallback 11) -> stages.thresholds.metrics.suggested of done sessions; rule at_edge() -> batch_snapshot.json
- thresholds already chosen: 8 sessions (LT1_J01_1D, LT1_J01_2D, LT1_J01_3D, LT1_J01_4D, LT1_J03_1D, LT1_J03_2D, LT1_J03_3D, LT1_J05_1D) -> config/sessions/<SID>.yaml has cnmf.gSig/min_corr/min_pnr -> batch_snapshot.json thresholds_chosen; same rows in _bearmind/threshold_choices.csv
- dropped_frames before rescan: 55 sessions, 334 frames -> registry anomalies code dropped_frames, sum data.n_dropped, read 13.11 ~20:25
- dropped_frames after rescan: 27 sessions, 58 frames; certain 18 frames in 5 sessions -> dry_rescan.py (scan_series with writes captured in memory, merged code d5f4d03) -> dropped_after.json
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
