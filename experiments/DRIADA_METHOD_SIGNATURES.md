# DRIADA Reconstruction Quality Method Signatures

## Investigation Summary

**DRIADA version: 0.6.4**
**LNOF processed files created: 2025-12-22**

## The Mystery

Current DRIADA 0.6.4 methods **RAISE ValueError** instead of returning NaN, but LNOF processed estimates have NaN values for 50 neurons.

### Method Signatures

```python
get_reconstruction_r2(self, event_only=False, n_mad=3.0, use_detected_events=True)
get_nmae(self, n_mad=3.0)
get_nrmse(self, n_mad=3.0)
get_snr_reconstruction(self)
```

### What the Source Code Shows

#### 1. `get_reconstruction_r2(event_only=True)`

**Should raise ValueError, NOT return NaN:**

```python
event_r2 = Neuron._calculate_event_r2(...)
if np.isnan(event_r2):
    raise ValueError('Event R² calculation failed. No events detected or insufficient event data...')
return event_r2
```

**Raises ValueError when:**
- `_calculate_event_r2` returns NaN
- Message: "No events detected or insufficient event data"
- Can happen when: sparse events, event mask is empty

#### 2. `get_nmae(n_mad=3.0)`

**Should raise ValueError, NOT return NaN:**

```python
mae = self.get_mae()
baseline_std = self.get_baseline_noise_std(n_mad=n_mad)
if baseline_std == 0:
    raise ValueError('Baseline noise std is zero, cannot normalize MAE')
return float(mae / baseline_std)
```

**Raises ValueError when:**
- Baseline noise std is zero
- Can happen when: flat baseline, insufficient baseline data

#### 3. `get_nrmse(n_mad=3.0)`

**Should raise ValueError, NOT return NaN:**

```python
rmse = self.get_noise_ampl()
baseline_std = self.get_baseline_noise_std(n_mad)
if baseline_std == 0:
    raise ValueError('Baseline noise std is zero, cannot normalize RMSE')
return float(rmse / baseline_std)
```

**Raises ValueError when:**
- Baseline noise std is zero
- Same conditions as get_nmae()

#### 4. `get_snr_reconstruction()`

**Should raise ValueError, NOT return NaN:**

```python
rmse = self.get_noise_ampl()
signal_std = np.std(self.ca.data)
if rmse == 0:
    raise ValueError('RMSE is zero, cannot compute reconstruction SNR')
self.snr_reconstruction = signal_std / rmse
return self.snr_reconstruction
```

**Raises ValueError when:**
- RMSE is zero
- Perfect reconstruction (theoretical only)

---

## The Paradox

**Expected behavior (based on source code):**
- Methods raise ValueError on failure
- auto_inspector.py exception handler catches it
- Returns NaN for ALL metrics (complete failure)

**Actual behavior (in LNOF data):**
- 50 neurons have NaN reconstruction metrics
- BUT signal metrics are valid (kinetics_source='wavelet_standard')
- This implies methods returned NaN WITHOUT raising exceptions

**Possible explanations:**

### 1. Older DRIADA Version During Processing
- LNOF files were processed with earlier DRIADA version
- Earlier version returned NaN instead of raising ValueError
- File timestamp shows Dec 22, 2025 - recent processing
- But could have been with older environment/code

### 2. Intermediate Method Returns NaN
- Methods like `get_baseline_noise_std()` might return NaN
- Division by NaN produces NaN without exception
- Zero-check wouldn't catch NaN baseline_std

### 3. Exception Handling in Caller
- auto_inspector.py might have wrapped these calls in try/except
- Caught ValueError and stored NaN
- But looking at code: NO try/except around reconstruction quality methods

### 4. Caching Issue
- Methods cache results (self.reconstruction_r2, self.snr_reconstruction)
- Cached NaN from earlier computation?
- But first call should still raise ValueError

---

## Testing Hypothesis

To determine why NaNs appear, we need to:

1. **Check intermediate methods:**
   ```python
   baseline_std = neuron.get_baseline_noise_std(n_mad=3.0)
   print(f"baseline_std: {baseline_std} (is NaN: {pd.isna(baseline_std)})")
   ```

2. **Check if exceptions are being caught somewhere:**
   - Look for try/except in processing pipeline
   - Check if LNOF processing used different code path

3. **Check DRIADA version history:**
   - When did ValueError checks get added?
   - Were LNOF files processed with earlier version?

4. **Check _calculate_event_r2 implementation:**
   - Does it return NaN in some cases?
   - What conditions lead to NaN return?

---

## Recommendations

### Immediate Fix (Pragmatic)

Wrap all DRIADA reconstruction quality method calls with safe wrappers:

```python
def safe_metric_call(func, default=0.0):
    """Safely call DRIADA method, catch both NaN returns AND exceptions."""
    try:
        result = func()
        return default if pd.isna(result) else result
    except (ValueError, ZeroDivisionError, Exception):
        return default
```

This handles:
- ValueError exceptions (current DRIADA behavior)
- NaN returns (if they somehow still happen)
- Any other exceptions

### Investigation Needed

1. Load a partial failure neuron's trace
2. Create fresh Neuron object with current DRIADA 0.6.4
3. Call reconstruction quality methods
4. See if they raise ValueError or return NaN
5. If ValueError: confirms LNOF used older code
6. If NaN: investigate intermediate method returns

### Long-term Solution

1. Update auto_inspector.py to wrap reconstruction method calls
2. Reprocess affected LNOF sessions if needed
3. Add validation: fail-fast if reconstruction metrics are NaN
4. Document expected DRIADA version for compatibility

---

## Key Finding

**The current DRIADA 0.6.4 code should NEVER return NaN from these methods - they should raise ValueError.**

The fact that we have NaN values in the dataset suggests either:
- Processing used older DRIADA version with different behavior
- Some edge case bypasses the ValueError checks
- Intermediate methods return NaN that isn't caught by zero-checks

We need to reproduce the issue with fresh processing to determine root cause.
