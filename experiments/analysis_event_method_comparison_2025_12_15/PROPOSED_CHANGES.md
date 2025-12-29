# Proposed Changes

## Change 1: Add Conda Environment Information to CLAUDE.md

**File:** `CLAUDE.md`
**Section:** Add new section after project context
**Issue:** CLAUDE.md doesn't specify which Python environment to use for BEARMiND development
**Impact:** Developers/AI assistants may use wrong Python interpreter, causing dependency issues

**Proposed Addition:**

```markdown
### **PYTHON ENVIRONMENT**

**MANDATORY CONDA ENVIRONMENT:**
```
ALWAYS use the 'bearmind' conda environment for all Python operations:

✅ CORRECT: conda run -n bearmind python script.py
✅ CORRECT: C:/Users/User/.conda/envs/bearmind/python.exe script.py

❌ WRONG: python script.py (uses system Python)
❌ WRONG: Using other conda environments

WHY: BEARMiND requires specific dependency versions (CaImAn, Driada, Bokeh)
      that are configured in the 'bearmind' environment.
```

**TESTING REQUIREMENT:**
Before running any Python script or test, verify environment:
```bash
conda run -n bearmind python -c "import sys; print(sys.executable)"
# Should show: /conda/envs/bearmind/python.exe
```
```

**Justification:**
- Prevents "module not found" errors from wrong Python environment
- Ensures reproducible builds and tests
- Reduces debugging time for environment-related issues

**Testing:**
- Verified that all analysis scripts work with `conda run -n bearmind python`
- Confirmed environment contains required dependencies

---

## Change 2: Document Auto-Inspector Method Distinction

**File:** Auto-inspector documentation (or comments in `auto_inspector.py`)
**Issue:** "wavelet" vs "threshold" naming is misleading
**Impact:** Users don't understand what they're actually choosing

**Proposed Documentation Update:**

```markdown
## Auto-Inspection Methods: Event-Based Filtering

BEARMiND auto-inspector has two modes that differ in rejection criteria:

### Method 1: "wavelet" (Strict Event-Based Filtering)
- **Deletion rate:** ~40% of neurons
- **Criteria applied:**
  - Morphological: corner artifacts, circularity, max edge, convexity
  - Event-based: t_rise, t_off, SNR, r_score
- **Use when:** You want high-quality neurons only, can tolerate false rejections
- **Risk:** May discard valid neurons with unusual kinetics

### Method 2: "threshold" (Morphology-Only Filtering)
- **Deletion rate:** ~15% of neurons
- **Criteria applied:**
  - Morphological: corner artifacts, circularity, max edge, convexity
  - Event-based: NONE (skipped)
- **Use when:** You want to retain more neurons, can tolerate lower quality
- **Risk:** May keep noisy/unreliable neurons

### Key Finding (2025-12-15 Analysis)
Event-based quality metrics (event_r2_score, events_per_min, etc.) are
**identical** between methods. The difference is ONLY in which rejection
criteria are applied during auto-inspection.

**Recommendation:** Choose based on your tolerance for false positives
(bad neurons kept) vs false negatives (good neurons deleted).
```

**Alternative:** Rename methods to "strict" vs "permissive" for clarity.

**Justification:**
- Analysis shows metrics are identical, only rejection criteria differ
- Current naming implies different event detection (misleading)
- Users need clear guidance on trade-offs

---

## Change 3: Add Ground Truth Validation for Disputed Neurons

**File:** New analysis script or notebook
**Priority:** HIGH
**Issue:** Unknown which method (v6 or v7) makes correct decisions

**Proposed Analysis:**

Create `validate_disputed_neurons.py` to:
1. Identify 17,303 neurons accepted by v7 but rejected by v6
2. Check their ground truth match rates
3. Compare quality metrics to known-good neurons
4. Generate recommendations

**Expected outcome:**
- Evidence-based recommendation on which method to use
- Identification of specific criteria thresholds that are too strict/loose
- Potential for optimized middle-ground criteria

**Implementation:** Can be done in analysis folder following deep protocol

---

## Implementation Priority

**HIGH (Do First):**
1. Change 1 (CLAUDE.md conda env) - Prevents immediate development issues

**MEDIUM (Do Soon):**
2. Change 2 (Auto-inspector docs) - Improves user understanding
3. Change 3 (Disputed neurons validation) - Provides evidence for method choice

**LOW (Nice to Have):**
- Implement granular criteria control in auto_inspector.py
- Add criteria configuration to batch processing
- Rename methods for clarity

---

## Notes

All proposed changes are documentation/analysis updates, not code modifications.
Per deep analysis protocol, no code files were modified during this investigation.

User approval required before implementing any changes to codebase.
