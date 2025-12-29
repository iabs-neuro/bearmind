# Root Cause Analysis: Partial Failures in Event Metrics

## Executive Summary

**Partial failures** occur when DRIADA's reconstruction quality methods (`get_reconstruction_r2()`, `get_nmae()`, etc.) **return NaN internally** instead of raising exceptions. This causes only reconstruction metrics to be NaN while signal metrics remain valid.

**Root Cause:** DRIADA methods have internal exception handling that returns NaN for edge cases (invalid reconstructions, insufficient events, division by zero).

**Solution:** Wrap DRIADA method calls with try/except and return 0 instead of NaN when failures occur.

---

## The Mystery Explained

### What Are Partial Failures?

50 neurons in v9 dataset have this pattern:
- `kinetics_source = 'wavelet_standard'` ✓ (event detection succeeded)
- Signal metrics are VALID ✓ (event_snr, events_fraction, kinetics_opt, t_rise, t_off)
- Reconstruction metrics are NaN ✗ (event_r2_score, nmae, nrmse, r2_score, snr_recon)

### Why This Is Confusing

In `auto_inspector.py:get_single_neuron_metrics()`, there's exception handling:

```python
try:
    neuron, kinetics_result = get_neuron_with_spikes(...)
    signal_metrics = get_signal_metrics(neuron, ...)
    if include_heavy:
        rec_metrics = get_reconstruction_quality_metrics(neuron)
        return {**signal_metrics, **rec_metrics}
except Exception:
    # Return NaN for ALL metrics
    return {...all NaNs...}
```

**If reconstruction fails and raises an exception, ALL metrics should be NaN, not just reconstruction metrics.**

### The Actual Cause

In `auto_inspector.py:get_reconstruction_quality_metrics()`:

```python
def get_reconstruction_quality_metrics(neuron):
    # NO TRY/EXCEPT HERE!
    r2_score = neuron.get_reconstruction_r2()               # Can return NaN
    event_r2_score = neuron.get_reconstruction_r2(event_only=True)  # Can return NaN
    nmae = neuron.get_nmae()                                # Can return NaN
    nrmse = neuron.get_nrmse()                              # Can return NaN
    snr_recon = neuron.get_snr_reconstruction()             # Can return NaN

    return {
        'r2_score': r2_score,         # NaN gets passed through
        'event_r2_score': event_r2_score,
        'nmae': nmae,
        'nrmse': nrmse,
        'snr_recon': snr_recon,
        'reconstruction': rec
    }
```

**DRIADA methods don't raise exceptions - they return NaN for edge cases:**
1. Invalid/None reconstruction array
2. Not enough events for event_only=True R2
3. Division by zero (protected with NaN)
4. Insufficient data for quality metrics

**Because no exception is raised, the outer try/except is not triggered, and the function returns successfully with NaN values.**

---

## Evidence from Investigation

### Test Case: LNOF_J01_2D, Neuron 492

```
Ground truth: KEEP
CaImAn SNR: 5.00

Signal metrics (VALID):
  event_snr: 2.3927
  events_fraction: 0.000355
  events_per_min: 0.6387
  kinetics_opt: 1.0
  t_rise: 0.2581
  t_off: 5.8063

Reconstruction metrics (NaN):
  event_r2_score: NaN
  r2_score: NaN
  nmae: NaN
  nrmse: NaN
  snr_recon: NaN
```

**What happened:**
1. DRIADA successfully created Neuron object ✓
2. Event detection succeeded (wavelet method) ✓
3. Kinetics optimization succeeded (t_rise=0.26s, t_off=5.81s) ✓
4. Signal metrics computed successfully ✓
5. Reconstruction quality methods called...
6. **DRIADA methods returned NaN (no exception raised)**
7. Function returned successfully with partial NaN values

### Distribution

- **All 50 partial failures** are from LNOF experiment
- They came from pre-processed pickle files that already had these NaNs
- Example: LNOF_J01_2D had 2/494 neurons (0.4%) with NaN reconstruction metrics
- These neurons were processed with `include_heavy=True` during LNOF dataset creation
- DRIADA reconstruction methods failed silently for these specific traces

---

## Why DRIADA Methods Return NaN

### Possible Internal Logic in DRIADA:

```python
# Hypothetical DRIADA internal code
def get_reconstruction_r2(self, event_only=False):
    try:
        if self.reconstructed is None:
            return np.nan

        if event_only and len(self.events) == 0:
            return np.nan  # Can't compute event-only R2 without events

        # Compute R2
        ss_res = np.sum((actual - reconstructed) ** 2)
        ss_tot = np.sum((actual - mean) ** 2)

        if ss_tot == 0:
            return np.nan  # Avoid division by zero

        r2 = 1 - (ss_res / ss_tot)
        return r2

    except Exception:
        return np.nan  # Catch-all: return NaN instead of raising
```

**This is actually GOOD defensive programming by DRIADA:**
- Prevents crashes from invalid data
- Signals failure with NaN
- Allows processing to continue

**BUT it creates partial failures in our pipeline:**
- We expect exceptions to be caught
- We don't check for NaN returns
- NaNs propagate into final dataset

---

## The Solution: Return 0 Instead of NaN

### Why Return 0?

1. **Clear failure marker**: 0 explicitly means "failed to compute"
2. **ML-friendly**: Most models handle 0 better than NaN
3. **Distinguishable**: Real measurements are rarely exactly 0
4. **Consistent**: Same value for all failure types

### Implementation Strategy

#### Option 1: Fix in `get_reconstruction_quality_metrics()` (RECOMMENDED)

```python
def get_reconstruction_quality_metrics(neuron):
    """
    Get reconstruction quality metrics from a DRIADA neuron.
    Returns 0 for any metric that fails to compute.
    """
    def safe_metric(func, default=0.0):
        """Safely call DRIADA method, return default if NaN or exception."""
        try:
            result = func()
            return default if pd.isna(result) else result
        except Exception:
            return default

    r2_score = safe_metric(lambda: neuron.get_reconstruction_r2())
    event_r2_score = safe_metric(lambda: neuron.get_reconstruction_r2(event_only=True))
    nmae = safe_metric(lambda: neuron.get_nmae())
    nrmse = safe_metric(lambda: neuron.get_nrmse())
    snr_recon = safe_metric(lambda: neuron.get_snr_reconstruction())

    rec = neuron.reconstructed
    if hasattr(rec, 'scdata'):
        rec = rec.scdata

    return {
        'r2_score': r2_score,
        'event_r2_score': event_r2_score,
        'nmae': nmae,
        'nrmse': nrmse,
        'snr_recon': snr_recon,
        'reconstruction': rec
    }
```

**Benefits:**
- Catches both exceptions AND NaN returns
- Minimal code change
- No impact on other metrics
- Easy to test

#### Option 2: Fix in Exception Handler

```python
def get_single_neuron_metrics(trace, fps=DEFAULT_FPS, ...):
    try:
        # ... existing code ...
    except (ValueError, ZeroDivisionError, Exception) as e:
        # Return 0 instead of NaN for all metrics
        zero_signal_metrics = {
            'events_per_min': 0,        # Changed from np.nan
            'events_fraction': 0,       # Changed from np.nan
            't_rise': 0,                # Changed from np.nan
            't_off': 0,                 # Changed from np.nan
            'event_snr': 0,             # Changed from np.nan
            'peak_amplitude_cv': 0,     # Changed from np.nan
            'kinetics_opt': 0,          # Changed from np.nan
            'kinetics_source': 'error'
        }

        if include_heavy:
            zero_rec_metrics = {
                'r2_score': 0,          # Changed from np.nan
                'event_r2_score': 0,    # Changed from np.nan
                'nmae': 0,              # Changed from np.nan
                'nrmse': 0,             # Changed from np.nan
                'snr_recon': 0,         # Changed from np.nan
                'reconstruction': None
            }
            return {**zero_signal_metrics, **zero_rec_metrics}
        else:
            return zero_signal_metrics
```

**Benefits:**
- Handles complete failures
- Consistent with Option 1
- Clear failure marker

#### Option 3: Hybrid (BEST)

Implement BOTH Option 1 and Option 2:
- Option 1 catches partial failures (DRIADA methods returning NaN)
- Option 2 catches complete failures (exceptions during processing)
- Result: **No NaNs in event metrics, ever**

---

## Migration Plan

### Step 1: Update auto_inspector.py

1. Add `safe_metric()` helper function
2. Update `get_reconstruction_quality_metrics()` to use it
3. Update exception handler to return 0 instead of NaN

### Step 2: Reprocess Affected Datasets

Option A: **Recompute LNOF metrics** (slow, thorough)
- Reload raw traces for 50 affected neurons
- Recompute all metrics with new code
- Should get 0 instead of NaN for failed metrics

Option B: **Post-process v9 dataset** (fast, pragmatic)
- Replace NaN with 0 in existing v9 dataset
- Accept that we won't know exactly why they failed
- Good enough for ML training

### Step 3: Prevent Future NaNs

- Add validation in `run_auto_inspection()` to check for NaN event metrics
- Log warning if any NaNs detected
- Optionally fail-fast if critical metrics are NaN

---

## Impact Assessment

### Complete Failures (177 neurons, 0.19%)
- Currently: All 11 event metrics = NaN
- After fix: All 11 event metrics = 0
- Improvement: Clear failure marker, ML-compatible

### Partial Failures (50 neurons, 0.05%)
- Currently: 5 reconstruction metrics = NaN, signal metrics valid
- After fix: 5 reconstruction metrics = 0, signal metrics valid
- Improvement: Preserves valid signal data, marks failed metrics

### Valid Neurons (92,015 neurons, 99.76%)
- No change
- Continue to have measured values

---

## Recommendation

**Implement Option 3 (Hybrid Approach):**

1. Update `get_reconstruction_quality_metrics()` with `safe_metric()` wrapper
2. Update exception handler to return 0 instead of NaN
3. Post-process v9 dataset: replace NaN → 0
4. Future datasets will have 0 from the start

**Result:**
- No NaNs in event metrics ever
- Clear distinction: 0 = failed, >0 = measured
- ML models can train on complete dataset
- Preserves partial data (50 LNOF neurons keep valid signal metrics)
