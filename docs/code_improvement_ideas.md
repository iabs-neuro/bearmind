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
