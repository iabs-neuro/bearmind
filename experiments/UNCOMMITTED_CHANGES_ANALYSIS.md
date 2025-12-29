# UNCOMMITTED CHANGES ANALYSIS

**Total uncommitted items: 340**

## EXECUTIVE SUMMARY

### Critical Issues
1. **Modified core files (4)** contain important bug fixes and features - MUST REVIEW
2. **__pycache__/*.pyc** should be in .gitignore - NOT in git
3. **data/** and **output/** directories - LARGE, should be in .gitignore
4. **.claude/** directory - IDE artifacts, should be in .gitignore
5. **130+ experimental scripts** in root - Need cleanup strategy

### Key Decision Points
- **Commit modified core files?** YES - contain critical fixes
- **Commit documentation?** YES - explains important systems
- **Commit trace_range analysis?** MAYBE - completed research, negative result
- **Commit 130+ experimental scripts?** NO - temporary/one-off analysis
- **Add .gitignore rules?** YES - prevent future bloat

---

## MODIFIED FILES (4 core modules)

### Files Changed:
1. `ae_launch.py` (+16 lines)
2. `auto_inspector.py` (+31 lines)
3. `bm_examinator.py` (+33 lines)
4. `ml/data_utils.py` (+54 lines)

### Changes Summary:

#### `ae_launch.py`
- **Bug fix**: Session name prefix extraction for LNOF timestamps
- **Impact**: Fixes file naming for LNOF sessions
- **Status**: PRODUCTION-READY, SHOULD COMMIT

#### `auto_inspector.py`
- **Feature**: Always compute `ellipse_r` (spatial feature)
- **Feature**: Add `half_crossing_rate` computation
- **Enhancement**: New default deletion rules:
  - `events_per_min<=0` (delete neurons without events)
  - `event_r2_score<0.0` (delete neurons with negative event fit)
- **Impact**: MAJOR - changes default autoinspection behavior
- **Status**: PRODUCTION-READY, SHOULD COMMIT

#### `bm_examinator.py`
- **Bug fix**: Pearson correlation for single neuron (was breaking)
- **Bug fix**: Spearman correlation for 2 neurons (scalar → matrix)
- **Enhancement**: Save feedback CSV to inspection_artifacts folder
- **Impact**: Fixes crashes in edge cases
- **Status**: PRODUCTION-READY, SHOULD COMMIT

#### `ml/data_utils.py`
- **Documentation**: Comprehensive comments on NON_FEATURE_COLS
- **Enhancement**: Add data leakage prevention (ml_keep_probability)
- **Organization**: Categorize FEATURE_COLS by type
- **Feature**: Add `half_crossing_rate`, `kinetics_source` to features
- **Impact**: Better maintainability, prevents ML errors
- **Status**: PRODUCTION-READY, SHOULD COMMIT

### Recommendation: **COMMIT ALL 4 FILES**
These are critical bug fixes and production-ready features.

---

## DOCUMENTATION FILES (6 .md files)

### Files:
1. `DRIADA_METHOD_SIGNATURES.md` - Documents Driada API
2. `HOW_NANS_ARE_GENERATED.md` - Explains NaN semantics
3. `PARTIAL_FAILURE_ROOT_CAUSE.md` - Documents partial failure bug
4. `ROBUST_FPS_SYSTEM.md` - FPS lookup system design
5. `WAVELET_OPTIMIZATION_PLAN.md` - Wavelet performance plan
6. `WAVELET_SPEEDUP_ANALYSIS.md` - Wavelet profiling results

### Value:
- **Critical knowledge capture** for future developers
- **Root cause analysis** prevents bug regression
- **System design docs** explain architectural decisions

### Recommendation: **COMMIT ALL 6 FILES**
Important technical documentation that explains complex systems.

---

## DATA FILES (4 in root)

### Files:
1. `LNOF_dataset.csv` - LNOF session dataset
2. `LNOF_dataset_from_processed.csv` - Alternative LNOF dataset
3. `LNOF_feedback.csv` - User corrections for LNOF
4. `config.json` - Configuration (unknown content)

### Issues:
- CSV files may be LARGE (not checked size)
- Datasets should live in `ml/results/` not root
- `config.json` might be user-specific

### Recommendation: **DO NOT COMMIT**
- Add `*.csv` to .gitignore (except in ml/results/ if needed)
- Move important datasets to proper location first
- Check if config.json is user-specific or template

---

## EXPERIMENTAL SCRIPTS (130+ in root, 32 in ml/)

### Categories:

#### Analysis Scripts (many duplicates):
- `analyze_*.py` (20+ variants)
- `compare_*.py` (15+ variants)
- `investigate_*.py` (10+ variants)
- `examine_*.py`, `explain_*.py`, `debug_*.py`

#### v8 Iteration Scripts (obsolete):
- `apply_v8_corrections*.py` (6 iterations)
- `retrain_v8_corrected*.py` (6 iterations)
- `visualize_v8_*.py` (multiple)

#### Test Scripts:
- `test_*.py` (30+ variants)
- One-off experiments, profiling, validation

#### Trace Range Analysis (our recent work):
- `ml/analyze_simple_trace_features.py`
- `ml/compare_range_methods.py`
- `ml/compute_trace_range_full.py`
- `ml/cv_test_trace_range.py`
- `ml/test_lnof_pattern.py`
- `ml/scan_estimates_paths.py`

### Issues:
- **130+ scripts is UNMAINTAINABLE**
- Many are ONE-OFF experiments (not reusable)
- Naming is inconsistent (analyze vs investigate vs examine)
- v8 iteration scripts are OBSOLETE (v9 now)

### Recommendation: **SELECTIVE APPROACH**

**COMMIT:**
- `ml/train_ebm_universal.py` - if it's the production training script
- Trace range analysis scripts - ONLY IF we want to preserve the research

**DO NOT COMMIT:**
- One-off analysis scripts (analyze_, investigate_, explain_)
- Test/debug scripts (test_, debug_)
- Obsolete v8 iteration scripts
- Duplicate/experimental variations

**ALTERNATIVE:**
- Create `experiments/` directory for historical research
- Move completed experiments there (with README)
- Keep root clean for production code only

---

## ML ARTIFACTS (110 results + 43 model folders)

### Results in ml/results/:
- `training_dataset_v9_corrected_iter[1-6].csv` - Intermediate iterations (large)
- `*_visualizations/` - Regenerable PNG folders
- `cv_*.csv` - Cross-validation results (small, useful)
- `*.png` - Plots (regenerable from CSV)
- `dataset_with_trace_stats.csv`, `dataset_with_range_variants.csv` - Our analysis

### Model Artifacts:
- `ml/ebm_v9_iter[1-8]/` - Training folders (large, ~43 dirs)
- `ml/ebm_grid_search*/` - Grid search results (very large)
- `ml/grid_search*/` - More grid searches

### Issues:
- **VERY LARGE** (models, datasets, visualizations)
- **REGENERABLE** (can retrain, replot)
- **INTERMEDIATE ARTIFACTS** (not final production)

### Recommendation: **DO NOT COMMIT**

**Keep in git:**
- Final datasets that are referenced (v9_corrected_iter7 - already committed)
- Final plots used in papers/presentations (manually select)
- Small CSV summaries (cv_comparison, feature_importance)

**Add to .gitignore:**
- `ml/ebm_v9_iter*/` (except iter8 .pkl already committed)
- `ml/*grid_search*/`
- `ml/results/*_visualizations/`
- Large intermediate datasets

**Exception:**
- If trace_range analysis is valuable research, commit:
  - `ml/results/dataset_with_trace_stats.csv`
  - `ml/results/simple_trace_features_analysis.png`
  - `ml/results/range_methods_comparison.png`
  - `ml/results/cv_trace_range_experiment.csv`

---

## OLD PRODUCTION MODELS (4 .pkl files)

### Files:
- `production_models/ebm_v8_corrected_iter1.pkl`
- `production_models/ebm_v8_corrected_iter2.pkl`
- `production_models/ebm_v8_corrected_iter3.pkl`
- `production_models/ebm_v8_corrected_iter4.pkl`

### Issues:
- v8 models are **OBSOLETE** (v9_iter8 is current)
- Should we keep old models for rollback?
- Large files (ML models can be 10s-100s of MB)

### Recommendation: **DO NOT COMMIT**
- Current production model is v9_iter8 (already committed)
- Old models should be archived separately (not in git)
- If rollback needed, use git history of v8 commit

---

## SYSTEM FILES (3 items)

### Files:
1. `.claude/` - Claude Code IDE artifacts
2. `__pycache__/bm_examinator.cpython-310.pyc` - Python bytecode
3. `nul` - Windows null file artifact

### Issues:
- IDE artifacts should NEVER be in git
- Compiled Python bytecode should NEVER be in git
- `nul` is a mistake (likely from redirect error)

### Recommendation: **ADD TO .gitignore**
```gitignore
.claude/
__pycache__/
*.pyc
*.pyo
nul
```

---

## LARGE DIRECTORIES

### `data/`
- Contains raw datasets, estimates files, validation sets
- **VERY LARGE** (likely GBs)
- Should NOT be in git (use external storage)

### `output/`
- Contains processing outputs, inspection artifacts
- **LARGE** and **REGENERABLE**
- Should NOT be in git

### Recommendation: **ADD TO .gitignore**
```gitignore
data/
output/
```

**Exception:** Small reference datasets can be in `ml/results/` if needed.

---

## PROPOSED ACTION PLAN

### 1. IMMEDIATE COMMITS (production-ready)

```bash
# Commit modified core files with bug fixes
git add ae_launch.py auto_inspector.py bm_examinator.py ml/data_utils.py
git commit -m "fix: critical bug fixes and feature additions

- Fix session name prefix extraction for LNOF timestamps
- Fix Pearson/Spearman correlation for edge cases (1-2 neurons)
- Add half_crossing_rate feature computation
- Always compute ellipse_r spatial feature
- Add new deletion rules: events_per_min<=0, event_r2_score<0.0
- Improve data_utils.py documentation and organization
- Save feedback CSV to inspection_artifacts folder"

# Commit documentation
git add DRIADA_METHOD_SIGNATURES.md HOW_NANS_ARE_GENERATED.md \
        PARTIAL_FAILURE_ROOT_CAUSE.md ROBUST_FPS_SYSTEM.md \
        WAVELET_OPTIMIZATION_PLAN.md WAVELET_SPEEDUP_ANALYSIS.md
git commit -m "docs: add technical documentation for critical systems

- DRIADA_METHOD_SIGNATURES.md: Documents Driada API patterns
- HOW_NANS_ARE_GENERATED.md: Explains NaN semantics in features
- PARTIAL_FAILURE_ROOT_CAUSE.md: Documents partial failure bug
- ROBUST_FPS_SYSTEM.md: FPS lookup system design
- WAVELET_OPTIMIZATION_PLAN.md: Performance optimization plan
- WAVELET_SPEEDUP_ANALYSIS.md: Profiling results and findings"
```

### 2. CREATE/UPDATE .gitignore

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# IDE
.vscode/
.idea/
.claude/

# Data
data/
output/
*.pkl
!production_models/*.pkl  # Exception: keep production models

# ML Artifacts
ml/ebm_v*/
ml/*grid_search*/
ml/results/*_visualizations/

# OS
.DS_Store
Thumbs.db
nul

# Large datasets (keep small CSVs in ml/results/)
/*.csv
!ml/results/*.csv

# Jupyter
.ipynb_checkpoints/
```

### 3. OPTIONAL: Commit trace_range research

**IF** we want to preserve this research for future reference:
```bash
git add ml/analyze_simple_trace_features.py \
        ml/compare_range_methods.py \
        ml/compute_trace_range_full.py \
        ml/cv_test_trace_range.py \
        ml/results/cv_trace_range_experiment.csv
git commit -m "research: trace_range feature analysis (negative result)

Investigated whether adding trace_range (max - min) improves v9 model.

Findings:
- trace_range is discriminative in isolation (Cohen's d = +0.564)
- Raw range better than percentile-based (AUC 0.7492 vs 0.6706)
- NO improvement when added to model (F-beta +0.01%, p=0.30)
- Current features already capture this signal (trace_skewness, trace_kurtosis)

Conclusion: Do NOT add trace_range to production model."
```

### 4. CLEANUP: Remove experimental scripts

**Create cleanup branch:**
```bash
git checkout -b cleanup-experimental-scripts

# Create archive for experiments
mkdir -p experiments/v8_iterations
mv apply_v8_*.py retrain_v8_*.py visualize_v8_*.py experiments/v8_iterations/

mkdir -p experiments/analysis
mv analyze_*.py investigate_*.py examine_*.py explain_*.py experiments/analysis/

mkdir -p experiments/tests
mv test_*.py experiments/tests/

# Add README explaining archive
cat > experiments/README.md << 'EOF'
# Experimental Scripts Archive

This directory contains one-off analysis, debugging, and test scripts
from various development phases. These are kept for historical reference
but are not maintained or tested.

## Structure
- `v8_iterations/` - Scripts from v8 iterative correction development
- `analysis/` - One-off analysis scripts
- `tests/` - Experimental test scripts
EOF

git add experiments/
git commit -m "chore: archive experimental scripts to experiments/"
```

### 5. VERIFY: Check git status clean

```bash
git status
# Should show only:
# - .gitignore additions
# - Potentially some remaining scripts to decide on
```

---

## FINAL RECOMMENDATIONS

### ✅ COMMIT NOW:
1. Modified core files (4) - **CRITICAL BUG FIXES**
2. Documentation files (6) - **KNOWLEDGE CAPTURE**

### ⚠️ DECIDE:
1. Trace_range analysis - **COMPLETED RESEARCH**
   - Pro: Preserves methodology, negative results are valuable
   - Con: Adds unused code to codebase
   - **Recommendation: COMMIT** (research documentation is valuable)

### ❌ DO NOT COMMIT:
1. Data files in root (*.csv, data/, output/)
2. ML artifacts (ebm_v9_iter*/, grid_search*/)
3. Old production models (ebm_v8_*.pkl)
4. 130+ experimental scripts (analyze_*, test_*, etc.)
5. System files (.claude/, __pycache__, nul)

### 🔧 INFRASTRUCTURE:
1. **CREATE .gitignore** - Prevent future bloat
2. **ARCHIVE experiments/** - Preserve history without git
3. **DOCUMENT cleanup** - Explain what was removed and why

---

## RISK ANALYSIS

### Risks of NOT committing modified core files:
- **HIGH**: Bug fixes lost if environment corrupted
- **HIGH**: Features (HCR, ellipse_r, deletion rules) not in git
- **MEDIUM**: Merge conflicts if others modify same files

### Risks of committing experimental scripts:
- **LOW**: Bloats repository (~130 files)
- **LOW**: Confuses future developers (which scripts are production?)
- **LOW**: Maintenance burden (untested code in repo)

### Risks of NOT committing trace_range research:
- **LOW**: Research methodology lost
- **LOW**: Future developer may repeat same analysis
- **VERY_LOW**: Negative results are still valuable science

---

## CONCLUSION

**IMMEDIATE ACTION REQUIRED:**
1. Commit modified core files (CRITICAL)
2. Commit documentation files (IMPORTANT)
3. Create .gitignore (PREVENTS FUTURE ISSUES)

**FOLLOW-UP:**
1. Decide on trace_range research commit
2. Archive experimental scripts
3. Clean up data files (move to external storage)

**DO NOT DELAY** committing the modified core files - they contain
production-critical bug fixes and features currently only in working tree.
