# Code Improvement Ideas Tracking

**Purpose**: Track identified improvements, inefficiencies, and tech debt discovered during development.
**Rules**: Check for duplicates before adding. Do NOT fix issues marked here without proper task definition and testing.

---

## 🔴 HIGH PRIORITY

### [BUG] auto_inspector: Deprecated function still in use
- **Location**: `auto_inspector.py:24-37`
- **Issue**: `get_hvals()` marked as DEPRECATED but still present in codebase
- **Suggestion**: Remove if truly unused, or document why it remains
- **Risk**: LOW - Appears unused based on grep
- **Benefit**: Reduce code confusion, remove dead code

---

## 🟡 MEDIUM PRIORITY

### [TECH-DEBT] Large debug notebook in repository
- **Location**: `BEARMiND_full_pipeline_debug.ipynb` (2.5MB)
- **Issue**: Very large notebook committed, likely contains outputs/data
- **Suggestion**: Clear outputs, or move to separate debug branch
- **Risk**: LOW - Just repository bloat
- **Benefit**: Cleaner repository, faster cloning

### [REFACTOR] bm_examinator.py is 1059 lines
- **Location**: `bm_examinator.py:1-1059`
- **Issue**: Single file too large, multiple responsibilities (GUI + logic)
- **Suggestion**: Consider splitting GUI components from data logic
- **Risk**: HIGH - Core interactive module, many dependencies
- **Benefit**: Better maintainability, easier testing

### [REFACTOR] Duplicate geometry functions
- **Location**: `utils.py:70-92` and `polygon.py`
- **Issue**: `calculate_polygon_area()` in utils.py, but polygon.py has similar functions
- **Suggestion**: Consolidate geometric operations in polygon.py
- **Risk**: MEDIUM - Need to check all callers
- **Benefit**: Single source of truth for geometry

---

## 🟢 LOW PRIORITY

### [PERF] Checkpoint cleanup needed
- **Location**: `.ipynb_checkpoints/` throughout project
- **Issue**: Multiple checkpoint directories not in .gitignore (were already ignored after .gitignore added)
- **Suggestion**: Clean up existing checkpoint files
- **Risk**: LOW - Just cleanup
- **Benefit**: Cleaner directory structure

### [REFACTOR] Hardcoded Moscow timezone
- **Location**: `utils.py:21`
- **Issue**: Timezone hardcoded to 'Europe/Moscow'
- **Suggestion**: Make timezone configurable in config.py
- **Risk**: LOW - Purely convenience
- **Benefit**: More flexible for international users

---

## 📋 DOCUMENTATION NEEDS

### [DOCS] Missing docstrings in core functions
- **Location**: Multiple files (bm_batch_routines.py, auto_inspector.py)
- **Issue**: Many functions lack docstrings explaining parameters and return values
- **Suggestion**: Add comprehensive docstrings following NumPy style
- **Risk**: LOW - Pure documentation
- **Benefit**: Better code understanding, easier onboarding

---

## ⚠️ INVESTIGATION NEEDED

### [UNKNOWN] warnings.filterwarnings('ignore') in bm_batch_routines
- **Location**: `bm_batch_routines.py:31`
- **Issue**: All warnings suppressed - might hide important issues
- **Suggestion**: Investigate what warnings are being hidden, use specific filters
- **Risk**: MEDIUM - Could hide real problems
- **Benefit**: Better error visibility

---

**Last Updated**: 2025-11-08
**Total Items**: 8 (1 HIGH, 3 MEDIUM, 2 LOW, 1 DOCS, 1 INVESTIGATION)


---

## COMPONENT ALIGNMENT AUDIT (2026-10-18)

Found while fixing estimates_compression.py, which gave every kept component
another component's bl / c1 / neurons_sn / SNR_comp / r_values / g by taking
the first N entries instead of the kept IDs (fixed together with
tests/test_compression_values.py). The same class of defect - per-component
data aligned by position in one place and by component ID in another - also
exists elsewhere. Not fixed yet.

### [HIGH] Merge leaves SNR_comp / r_values / lam / cnn_preds on the old numbering
- **Location**: `auto_inspector.py` implement_decision -> CaImAn `Estimates.manual_merge`
- **Issue**: CaImAn 1.11.3 manual_merge renumbers A, C, YrA, S, bl, c1, neurons_sn, g
  but not SNR_comp, r_values, lam, cnn_preds. After a merge these arrays keep the old
  length and order, so every component after the first merged source reads another
  component's value (measured on WAW_H02_1D: merge 33+34 -> new 48 reads SNR 4.29,
  its own is 19.52). estimates_to_metrics reads est.SNR_comp[comps], so merged rows in
  metrics_df get unrelated caiman_snr / caiman_r_score; saved processed pickles carry
  the shift. The GUI merge already fixes this (reassign_quality_metrics), the
  automatic pipeline does not.
- **Note**: compression now leaves such stale-length arrays untouched instead of
  cutting them, so it neither fixes nor worsens post-merge files.

### [HIGH] capcan validation counts deleted neurons as kept
- **Location**: `capcan_validation.py` (around lines 225-240 and 389-414);
  `auto_inspector.validate_decision`
- **Issue**: since implement_decision keeps deleted neurons visible in idx_components
  (commit d273702, 2025-12-11), estimates_to_metrics(est_auto) includes them and
  compute_metrics has no 'delete' filter. data/batch_validation_v8_results.csv:
  FOF_F09_1D n_initial 247, n_deleted 198, n_auto 247, precision 0.02. Precision / F1
  in these result files do not measure the pipeline.

### [HIGH] ExamineCells GUI deletes / merges the wrong neuron after a previous action
- **Location**: `bm_examinator.py` del_callback, merge_callback
- **Issue**: selected rows are mapped to IDs through storage.estimates_partial, which
  these callbacks never update (show / restore / revert / discard do). After one
  delete, the table shows the new list but the next selection is mapped through the
  old one, so a different neuron is deleted or merged. Found by reading the code,
  not reproduced in the GUI.

### [MEDIUM] Per-component caches not remapped after a GUI merge
- **Location**: `bm_examinator.py` merge_callback
- **Issue**: metrics_df component_idx, reconstructions, asp_cache and feedback_dict
  stay on pre-merge IDs and are saved like that; EstimatesToSrcFull then attaches
  metrics rows by position. Reported by the audit, not independently re-checked.

### [INVESTIGATE] Lower-confidence findings
- Export filters only on ml_keep_probability and ignores metrics_df['delete'], so a
  neuron deleted by the duplicate rule (lower SNR of a close pair) keeps a high
  probability and is exported: LNOF_J12_1D component 112, p=0.95, delete=1. Honouring
  'delete' would also drop 6 neurons at p=0.74-0.75 from that legacy file (no
  autoinspection_config; apparently inspected at 0.75, exported at 0.72). Merged
  components being dropped is fixed (ml_approved_components).
- ExamineCells save (`bm_examinator.py` ~385-402, `_session.mat`) writes every visible
  neuron, deleted ones included.
- Legacy LNOF_J12_1D holds from_merge rows with component_idx 552-554 while C has 481
  rows; not reproduced with current code.
- implement_decision assigns merged IDs as len(good)+i even if CaImAn drops an empty
  merge, shifting later merged IDs.
- Merged rows get population features (nn_distance_center, local_density, ...)
  computed over merged components only - transform_metrics_df_indices.
- `ml/apply_corrections.py` looks corrections up by CSV row number (iloc[fn_idx-1]).

### [STRUCTURE] Five hand-rolled copies of "change the component set, keep data aligned"
- estimates_compression, dev/create_ultra_lightweight_estimates.py,
  implement_decision + transform_metrics_df_indices + ae_launch remap, GUI
  merge_callback / reassign_quality_metrics, migrate_compressed_files.py. Each covers
  a different subset of fields. CaImAn `select_components` slices all per-component
  arrays by ID and skips (with a warning) any whose length is stale; callers would
  still remap metrics_df / reconstructions / asp_cache themselves.

### [ENV] bearmind env cannot import auto_inspector under conda run
- `import torch, caiman, driada` -> "OMP: Error #15: Initializing libomp.dll, but found
  libiomp5md.dll already initialized" (two OpenMP runtimes); importing ae_launch or
  auto_inspector fails, so test_export_e2e.py cannot run. torch and caiman import
  fine on their own.
