# BEARMiND recon (2026-10-30, branch bearmind-batch after 5affd57/6c26631)

Subagent report; spot-checked by main session: FPS rounding (export_estimates_data.py:52-63, table has 526x29.78, 165x19.76) and ae_launch defaults (brain='ml', ebm_v6_no3dm, 0.71 at ae_launch.py:204-206) confirmed. "nb N.L" = notebook cell N (0-based), line L; sources in recon/nb_cells.txt.

## Headline
Only autoinspection and export run headless. Crop, MC, CNMF-E and parameter choice go through the notebook: file dialogs, hand-typed per-session arrays, name[...] slicing. Session identity, FPS and parameters each live in 3-6 places that disagree.

## 1. Stage map
0 Setup: nb1 installs startup.py into IPython profile; nb3 update_config; nb5 create_mouse/session_configs -> config.json in CWD; <dirname(ROOT)>/MOUSE_CONFIGS, SESSION_CONFIGS (config.py:64-92).
1 Crop: nb7 askopenfilenames -> LoadSelectedVideos -> DrawCropper (sliders, Save) -> <session>_l_r_u_d_cropping.pickle next to avi (bm_batch_routines.py:212-227). Session = parent folder (bonsai) or path[-5] (legacy) (config.py:95-103).
2 Crop+timestamps: nb11 DoCropAndRewrite, extract_and_copy_ts: all *.avi natsorted+concat -> ROOT/<session>_CR.tif; <folder>_Mini_TS.csv -> ROOT/<session>_timestamp.csv (bm_batch_routines.py:250-400). Uses undefined `folder_structure` (nb 11.2).
3 MC: nb12 (glob ROOT/*_CR.tif) or nb17 (dialog) DoMotionCorrection -> *_CR_MC.tif, _MC.mp4, optional dF/F mp4 (403-445).
4 Params: nb12/13/23 plot_gsig_range, plot_min_corr_and_pnr_range; nb15 bm_qc.batch_suggest_thresholds -> param_images/*; person types arrays into nb24.7-24.10 -> SESSION_CONFIGS/<sid>.json['cnmf_params'].
5 CNMF-E(+QC): nb26 dialog over *_MC.tif, DoCNMF(qc=True) -> ROOT/<sid>_estimates.pickle (+cnmf_dict, tif_name abs path, imax); <sid>_qc.mp4/_qc.json/_qc_residual_maps.png. FPS from glob(name[:-10]+'*.csv')[0] (nb26.8); session config from name[-20:-10] (nb26.10).
6 Compression (optional): nb33 -> <dir>_ultra_lightweight/.
7 Autoinspection: nb36 single, nb37 batch (dialogs), batch_autoinspect_all.py, ae_launch CLI, legacy batch_autoinspect.py -> <artifacts_path>/inspection_artifacts_<sid>/ (metrics csv, rejected, FCD/FBD/match npy, <stem>_processed.pickle). artifacts_path './output' relative to CWD (nb36.61). Legacy writes capcan_artifacts_<name>/. FPS from fps_data.csv.
8 Manual inspection: nb39 ExamineCells (Bokeh) saves <sid><timestamp>_estimates.pickle and <sid>final_estimates.pickle into CWD (bm_examinator.py:1268-1282); SaveResults traces.csv/filters next to pickle; feedback.csv into inspection_artifacts_<sid>.
9 Export: nb41 dialog, batch_export.py, batch_export_all.py, export_estimates_data CLI; nb42 regenerates metadata -> <sid>_data.npz/_metadata.json/_filters.mat next to pickle. FPS from table, raises if missing.
10 FOF split: resplit_fof.py (hardcoded W:\ paths; deletes Synchro/traces, metadata, synchronized without confirmation).
11 Events: wavelet asp computed in autoinspection and exported; nb48/50 threshold FitEvents; nb53 wavelet (calls undefined events_to_csv, get_events_from_ridges); collectors read Synchro/synchronized/NOF_*_aligned.npz produced outside repo (inferred).
No manifest or stage status anywhere; discovery = suffix glob in flat ROOT, dialog, or hardcoded path.

## 2. Human touchpoints (headless alternative?)
1 Edit ROOT/DATA_PATHWAY/TEMP in cell (nb3); no CLI.
2 folder_structure undefined (nb11).
3 Crop sliders + Save (nb7); hidden "load/save crop to config" buttons (bm_batch_routines.py:185-205); DoCropAndRewrite reads only *cropping.pickle. No headless path.
4 Repair broken AVIs with ffmpeg by hand (nb10.2).
5 File dialogs in nb 15,17,19,26,28,33,36,37,39,41,42,44,53; input() in nb33.31. Globs only as commented lines.
6 gSig/min_corr/min_pnr by eye from PNG grids -> typed arrays (nb24.7-24.10). batch_suggest_thresholds gives 3 auto methods (bm_qc.py:258-345) but result not written to SESSION_CONFIGS; gsig_range fixed [4].
7 mc_params saved for one hardcoded session (nb20, 'NOF_H02_0D').
8 Bokeh curation (nb39); only producer of FP/FN feedback.csv used by export.
9 ManualSeeds (nb44) + ReDoCNMF (nb46); _qc.json residual_excess_outside (candidate missed neurons) read by nothing.
10 New sessions added to fps_data.csv by hand; export raises, ae_launch silently uses 30; get_fps_from_timestamps exists (bm_examinator.py:149-160) unused.
11 Dataset dicts/paths inside scripts (batch_autoinspect_all.py:58-71, batch_export_all.py:21-25, resplit_fof.py:21-26); chain runner polls a log the autoinspect script never writes (stdout redirect by hand, inferred).
12 Bokeh port 'localhost:8888' hardcoded (nb31.2, startup.py:29-30).
13 Wavelet params tuned by editing cells (nb53-56).

## 3. Hardcodes
Paths: config.py:8-9 (C:\Users\1\...), nb3, nb10, batch_autoinspect_all.py:60/64/68 (W:\DATASET, 'ClaciumData' typo in path), resplit_fof.py:21-26, collect_nof_aligned_events.py:30,174, bm_examinator.py:1814 (dead), model path relative to CWD, utils.py:21 Europe/Moscow.
Machine values: n_processes=14 (bm_batch_routines.py:408), max_mem_gb=16 (nb15/26/28), memory_fact=RAM/16.
FPS: default 20 (bm_examinator 108/123/149, nb19/26/28, bm_qc 668-670/883, batch_autoinspect 49, capcan_validation 121/337); 30 (ae_launch 33, nb24 'fr':30, nb39/53, bm_batch_routines 437/901, test_export_e2e 117); 19.76 (collect_nof 178). fr=fps/2 with tsub=2 (nb26.13-14) undocumented. Table FPS rounded to int (ae_launch 78/83, export_estimates_data 52/58/63, capcan_validation 55) -> exported fps 20 instead of 19.76 (1.2%).
ML: ml_threshold 0.72 (nb, script, export, test) / 0.71 (ae_launch 206,657) / 0.5 (auto_inspector 1536/1604/1671, bm_examinator 66/289/618). Model v9_iter8 (nb, script) vs ml/production_models/ebm_v6_no3dm.pkl (ae_launch 205,655). Brain 'ml' (ae_launch) / 'thresholds' (auto_inspector 1669) / 'hybrid' (nb, script).
Capcan: n_iter 2 vs 3; detect_corner_artifacts True vs False; corr_thr 0.4/0.35/0.6; cthr 0.35/0.3; pearson vs spearman; match_threshold 3 vs 1 (no effect at num_sessions=1; FOF stitched 3 days but num_sessions=1, inferred to matter); pxlthr=15 copied in 7 places; d_snr_thr=10 x5.
CNMF/MC: full cnmf_dict only inside nb24 loop; mc_dict twice (nb12 vs nb17, only first has write_mp4); noise_ampl=1e-5; PNR clip 70 (x3); images[::5].
Frames: end_frame=90000 needed because ExamineCells default end_frame=0 slices empty (bm_examinator 635/298-302); maxframes 20000/12000/10000/2000; 608x608 special case (bm_batch_routines 291-317).
Filename slicing: name[:-4-6], name[-20:-10] (359 of 941 IDs not 10 chars), split('/')[6:], partition('_seeds')[0][:-4], Filename[:10] (36 colliding prefixes), sname[:10], splt_path[-2]/[-5], split('_')[1], name[:-4][-2:], rpartition('/'), split('\\'), partition('estimates'/'.').
Dataset lists: CC/RFC/FOF dicts, regex (CC|RFC|FOF), DAYS, BOF/RFC arrays in nb24.

## 4. Hacks / bugs
Bugs: capcan_validation.load_fps_data wrong separator -> KeyError at import (verified), breaks collect_wavelet_event_stats; ReDoCNMF builds opts before seed settings -> seeds unused (inferred) and uses leftover cnmf_dict; nb17 mc_dict lacks write_mp4 -> KeyError after MC; trailing commas in nb55 make tuples; undefined events_to_csv/get_events_from_ridges/folder_structure; update_config rebinds CONFIG but modules hold old dict; get_timestamps glob relative to CWD -> time axis fallback (inferred); GUI saves to CWD but nb46 globs ROOT; ROOT+'*traces.csv' without separator; unknown DATA_PATHWAY -> UnboundLocalError; 'dview' in locals() always False.
Broad excepts: kinetics cascade (auto_inspector 203/216/236/257), 693/766/848/1079; bare except in bm_spike_detection 44/48, estimates_compression 262, table_routines 27; calc_fps returns None; FPS read failure -> default; batch loops swallow all to stdout; memmap PermissionError -> sleep 100x1s (nb26).
Monkeypatch/env: sys.modules['torch']=None; SSQ_GPU set after driada import in auto_inspector (wavelet_backend arg may be ineffective, inferred); BOKEH origin[7:-1]; warnings ignored at import; np.Inf pins NumPy<2.
Import side effects: config.py creates config.json in CWD; startup.py Tk window; bm_examinator output_notebook(); capcan_validation loads table; ae_launch imports bm_examinator (bokeh, ipywidgets, caiman, moviepy) just for LoadEstimates; auto_inspector depends on ml.data_utils.
Copy-paste: ML-filter+feedback export x4 (nb41, nb42, batch_export.py, export_estimates_data CLI; CLI applies feedback only with flag); batch autoinspect loop x3; capcan param set x4; FPS table readers x4; PNR->imax x3; get_fps name clash (table_routines vs export_estimates_data).
Naming: naming.SESSION_PATTERN matches all 941 IDs but slicing bypasses it; two artifact folder conventions (inspection_artifacts_ vs capcan_artifacts_); processed pickle named by stem, folder by sid.
Dead/clutter: bm_examinator 1765-1823 string block; bm_batch_routines 586-590, 106-111, hidden buttons; nb24.1-5; stale batch_autoinspect.py; tracked .DS_Store, .ipynb_checkpoints, MoviePlay.ipynb (2.9 MB).

## 5. Config and provenance
Config sources: config.json (CWD, global CONFIG at import), MOUSE/SESSION_CONFIGS (only cnmf_params read; mc_params never read; crop from pickle), notebook literal dicts, diverging function defaults, script constants, fps_data.csv (941 rows, hand-edited), env vars SSQ_GPU/SSQ_PARALLEL/BOKEH_ALLOW_WS_ORIGIN.
Recorded: crop in pickle; est.cnmf_dict, tif_name (abs), autoinspection_config -> metadata; export metadata (session, fps, timestamp, feedback, ML counts); _qc.json; threshold JSON.
Not recorded: MC params; noise_ampl; who/what chose CNMF values; raw AVI list; input pickle path; code version (no git hash anywhere); model hash; resolved deletion rules (string only); match_threshold/num_sessions/sf/ef/ds/comps_to_select; FPS source and rounding; manual GUI edits; wavelet params; FOF trim decisions. Absolute tif_name breaks QC/re-runs after moving data (inferred).

## 6. Top 10 opportunities
1 Per-session manifest/registry (ID, raw folder, dataset, FPS, stage status, outputs).
2 Headless CLI per stage with skip-existing + structured failure log, one orchestrator.
3 One session-ID and path API (naming.py + path builder) everywhere.
4 One versioned run-parameter file as single source for MC/CNMF/capcan/export; delete divergent defaults.
5 Automatic CNMF-E params: batch_suggest_thresholds + gSig rule -> session config.
6 Crop without GUI: reuse mouse/session crop config or auto FOV detection.
7 Exact FPS once from <sid>_timestamp.csv, stored in manifest; no int rounding.
8 Provenance block in every output: git hash, param file hash, model hash, inputs, FPS source.
9 Split headless core from GUI and import side effects.
10 QC-gated review queue instead of manual inspection/seed loop (residual_excess_outside, ML-probability bands); GUI saves next to artifacts; fix end_frame=0 default and ReDoCNMF.
Also: merge 4 export copies; fix capcan_validation separator.
