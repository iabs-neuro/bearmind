# Robust FPS Lookup System - Implementation Summary

## Problem Statement

**Critical Bug:** LNOF_J53_3D was showing 30 fps instead of correct 20 fps.

**Root Cause:** The FPS lookup regex pattern in `ae_launch.py` was hardcoded to match exactly 3-character experiment codes:
```python
r'([A-Z0-9]{3}_[A-Z]\d+_\d[A-Z])'  # {3} = exactly 3 chars
```

When applied to "LNOF_J53_3D" (4 characters):
- Pattern matched starting from 'N', extracting "NOF_J53_3D" (wrong!)
- Lookup for "NOF_J53_3D" failed (not in CSV)
- Returned default_fps=30 instead of correct 20

## Impact

**39 out of 88 LNOF sessions** were being analyzed with WRONG fps (30 instead of 20):
- Affected wavelet event detection (scale calculations depend on fps)
- Affected kinetics measurements (time-based)
- Affected events per minute calculations
- Affected all temporal metrics

**Additional bug found:** CSV was being read without specifying `sep=';'`, which could cause parsing issues.

## Solution: Robust Pattern Matching System

### Design Principles

1. **Flexible, not hardcoded** - Support ANY code length (2, 3, 4, 5+ chars)
2. **Future-proof** - Handle formats that don't exist yet
3. **Two-strategy lookup** - Exact match first, then pattern extraction
4. **Comprehensive pattern** - Handle trial suffixes (_1T, _2T, etc.)
5. **Correct CSV parsing** - Use `sep=';'` for semicolon-separated file

### Robust Pattern

```python
r'([A-Z0-9]+_[A-Z]\d+_\d[A-Z](?:_\d[A-Z])?)'
```

**Pattern breakdown:**
- `[A-Z0-9]+` - Experiment code (any length: NOF, LNOF, FOF, RFC, 3DM, future codes)
- `_[A-Z]\d+` - Mouse ID (letter + digits: H01, J53, F05, D17)
- `_\d[A-Z]` - Day (digit + letter: 1D, 2D, 3D, 4D)
- `(?:_\d[A-Z])?` - Optional trial suffix (_1T, _2T, etc.)

### Implementation

**File:** `ae_launch.py`
**Function:** `get_fps_from_table()`
**Lines modified:** 33-94 (complete rewrite)

**Key improvements:**
1. Flexible regex pattern (supports any code length)
2. Two-strategy lookup:
   - Strategy 1: Try exact match first (fastest)
   - Strategy 2: Extract with pattern, then lookup
3. Correct CSV separator: `pd.read_csv(FPS_TABLE_PATH, sep=';')`
4. Comprehensive docstring with examples
5. Detailed inline comments explaining pattern structure

## Test Results

### Comprehensive Pattern Test
- **14 test cases**: All passed
- **215 CSV entries**: All matched correctly
- **8 improvements** over old pattern
- **Handles:**
  - Current formats (NOF, FOF, RFC, 3DM, LNOF)
  - Trial suffixes (3DM_D17_1D_1T)
  - Filenames with paths/extensions
  - Future hypothetical formats (XLNOF, AB, VERYLONGCODE)

### Integration Test
- **12 test cases**: All passed
- **Critical bug verified fixed:** LNOF_J53_3D now returns 20 fps
- **All LNOF sessions**: Now use correct fps values

## Impact Assessment

### Before Fix
- 39 LNOF sessions: WRONG fps (30 instead of 20)
- Hardcoded pattern: breaks on future formats
- CSV parsing: potentially buggy (no separator specified)
- Silent failures: no way to detect issues

### After Fix
- **ALL sessions**: Correct fps values
- **Future-proof**: Supports any experiment code length
- **Robust**: Two-strategy lookup with fallbacks
- **Well-documented**: Clear pattern structure and examples
- **Tested**: Comprehensive test coverage

## Supported Formats

### Current Formats
- ✓ 3-char codes: NOF_H01_1D, FOF_F05_1D, RFC_F01_1D
- ✓ 4-char codes: LNOF_J53_3D, LNOF_J01_1D
- ✓ Numeric codes: 3DM_D17_1D
- ✓ With trial suffix: 3DM_D17_1D_1T
- ✓ In filenames: path/NOF_H01_1D.pickle
- ✓ With suffixes: NOF_H32_4D_estimates.pickle

### Future Formats (Automatically Supported)
- ✓ 2-char codes: AB_X9_1D
- ✓ 5-char codes: XLNOF_M123_5D
- ✓ Very long codes: VERYLONGCODE_Z999_9D
- ✓ Any alphanumeric code length
- ✓ Multiple trial suffixes: _2T, _3T, etc.

## Files Created

### Implementation
- `ae_launch.py` (MODIFIED) - Robust FPS lookup function

### Tests
- `test_fps_regex.py` - Demonstrates the original bug
- `test_fps_regex_fix.py` - Verifies the fix works
- `test_robust_fps_system.py` - Comprehensive pattern testing
- `test_robust_fps_integration.py` - Integration test with actual function
- `test_fps_lookup.py` - Simple bug verification

### Documentation
- `ROBUST_FPS_SYSTEM.md` (this file) - Complete summary

## Verification Commands

```bash
# Test the robust pattern matching
conda run -n bearmind python test_robust_fps_system.py

# Integration test with actual function
conda run -n bearmind python test_robust_fps_integration.py

# Quick verification of critical bug fix
conda run -n bearmind python -c "from ae_launch import get_fps_from_table; print(f'LNOF_J53_3D: {get_fps_from_table(\"LNOF_J53_3D\")} fps (expected: 20)')"
```

## Backward Compatibility

✓ **Fully backward compatible**
- All existing 3-char formats still work correctly
- No changes required to calling code
- Default behavior unchanged (returns default_fps if not found)
- Function signature unchanged

## Benefits

1. **Correctness** - 39 LNOF sessions now use correct fps
2. **Robustness** - Handles edge cases, filenames, paths
3. **Future-proof** - No hardcoded character counts
4. **Maintainability** - Clear documentation and structure
5. **Reliability** - Two-strategy lookup with fallbacks
6. **Testability** - Comprehensive test coverage

## Conclusion

The robust FPS lookup system fixes a critical bug affecting 39 sessions and provides a future-proof solution that can handle any experiment identifier format without code changes.

**Status:** ✓ Implemented, tested, and verified
**Critical bug:** ✓ Fixed (LNOF_J53_3D now returns correct 20 fps)
**Test coverage:** ✓ Comprehensive (215 CSV entries + 14 edge cases)
**Future formats:** ✓ Supported automatically
