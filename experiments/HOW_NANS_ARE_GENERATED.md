# How NaNs Are Generated in v9 Dataset

## Summary

**227 neurons (0.25% of 92,242 total) have NaN values in event-based metrics.**

These NaNs come from two sources:
1. **Complete processing failures** (177 neurons) - DRIADA processing crashed
2. **Partial processing failures** (50 neurons) - Event detection succeeded but reconstruction failed

---

## The Root Cause: Exception Handling in auto_inspector.py

When computing event-based metrics, if DRIADA processing fails, the code catches exceptions and returns NaN for all metrics.

### Location: auto_inspector.py:372-397

```python
def get_single_neuron_metrics(trace, fps=DEFAULT_FPS, include_heavy=False, ...):
    """Extract metrics from a single neuron trace."""
    try:
        neuron, kinetics_result = get_neuron_with_spikes(
            trace, fps=fps, lightweight=not include_heavy, ...
        )
        signal_metrics = get_signal_metrics(neuron, kinetics_result=kinetics_result)
        if include_heavy:
            rec_metrics = get_reconstruction_quality_metrics(neuron)
            return {**signal_metrics, **rec_metrics}
        else:
            return signal_metrics

    except (ValueError, ZeroDivisionError, Exception) as e:
        # Handle flat/zero traces or other processing failures
        # Return NaN for all metrics
        nan_signal_metrics = {
            'events_per_min': np.nan,
            'events_fraction': np.nan,
            't_rise': np.nan,
            't_off': np.nan,
            'event_snr': np.nan,
            'peak_amplitude_cv': np.nan,
            'kinetics_opt': np.nan,
            'kinetics_source': 'error'  # <- Marker for complete failure
        }

        if include_heavy:
            nan_rec_metrics = {
                'r2_score': np.nan,
                'event_r2_score': np.nan,
                'nmae': np.nan,
                'nrmse': np.nan,
                'snr_recon': np.nan,
                'reconstruction': None
            }
            return {**nan_signal_metrics, **nan_rec_metrics}
        else:
            return nan_signal_metrics
```

**When exceptions occur:**
- All signal metrics → NaN
- All reconstruction metrics → NaN (if `include_heavy=True`)
- `kinetics_source` → 'error' (marker for failed processing)

---

## Two Types of NaN Failures

### Type 1: Complete Processing Failures (177 neurons)

**Characteristics:**
- `kinetics_source = 'error'`
- ALL 11 event metrics are NaN
- DRIADA processing crashed entirely

**Distribution:**
- 72 KEEP neurons (ground_truth=1)
- 105 DELETE neurons (ground_truth=0)
- Experiments: RFC (93), LNOF (50), NOF (33), FOF (1)

**Why do these fail?**
Common reasons:
1. Flat/zero traces (no neural activity)
2. Corrupted data (NaN or Inf values in trace)
3. Trace too short for processing
4. Division by zero in DRIADA computations
5. Invalid parameters causing ValueError

**Example neurons:**
```
session_name    component_idx  ground_truth  experiment  kinetics_source
FOF_F48_1D             205            0         FOF          error
NOF_H01_4D              49            1         NOF          error
NOF_H02_2D             215            1         NOF          error
```

All event metrics (event_r2_score, event_snr, events_fraction, etc.) = NaN

---

### Type 2: Partial Reconstruction Failures (50 neurons)

**Characteristics:**
- `kinetics_source = 'wavelet_standard'` (event detection succeeded!)
- Only 5 reconstruction metrics are NaN: event_r2_score, nmae, nrmse, r2_score, snr_recon
- Signal metrics are VALID: event_snr, events_fraction, events_per_min, kinetics_opt, t_off, t_rise

**Distribution:**
- ALL 50 are KEEP neurons (ground_truth=1)
- ALL 50 are from LNOF experiment
- These came from LNOF_dataset_from_processed.csv which already had these NaNs

**Why do these fail?**
This is specific to LNOF dataset creation. Looking at the source:
- LNOF dataset has 100 neurons with NaN reconstruction metrics (0.20%)
- 50 of those made it into v9
- Event detection worked fine (got events, computed SNR, optimized kinetics)
- But reconstruction quality metrics failed to compute

**Example neurons:**
```
session_name    component_idx  ground_truth  caiman_snr  event_snr  events_fraction  kinetics_opt
LNOF_J01_2D           492            1        5.00       2.39         0.000355          1.0
LNOF_J01_2D           493            1       10.49       3.29         0.000177          1.0
```

Signal metrics are valid, but event_r2_score, nmae, nrmse, r2_score, snr_recon = NaN

---

## The Processing Pipeline

### Step 1: Trace → DRIADA Processing
```
Calcium trace → get_neuron_with_spikes() → DRIADA Neuron object
```

**Can fail here with:**
- ValueError (invalid trace data)
- ZeroDivisionError (flat trace)
- Other exceptions

**Result if fails:** Complete failure → all NaNs + kinetics_source='error'

---

### Step 2: Neuron → Signal Metrics
```
DRIADA Neuron → get_signal_metrics() → event_snr, events_fraction, kinetics_opt, etc.
```

**Computed from:**
- Event detection (spike trains)
- Wavelet analysis
- Kinetics optimization

**Usually succeeds if Step 1 succeeded**

---

### Step 3: Neuron → Reconstruction Quality Metrics
```
DRIADA Neuron → get_reconstruction_quality_metrics() → r2_score, event_r2_score, nmae, nrmse, snr_recon
```

**Computed from:**
- `neuron.get_reconstruction_r2()`
- `neuron.get_reconstruction_r2(event_only=True)`
- `neuron.get_nmae()`
- `neuron.get_nrmse()`
- `neuron.get_snr_reconstruction()`

**Can fail independently if:**
- Reconstruction array is invalid
- Not enough events for event-only R2
- Division by zero in quality metrics
- DRIADA bug in reconstruction methods

**Result if fails:** Partial failure → signal metrics OK, reconstruction metrics NaN

---

## NaN Distribution in v9 Dataset

```
Total neurons: 92,242

Neurons with NaN event metrics: 227 (0.25%)
├── Complete failures: 177 (0.19%)
│   ├── KEEP: 72
│   ├── DELETE: 105
│   └── All 11 event metrics = NaN
│
└── Partial failures: 50 (0.05%)
    ├── KEEP: 50 (all from LNOF)
    ├── DELETE: 0
    ├── Signal metrics: VALID
    └── Reconstruction metrics: NaN (event_r2_score, nmae, nrmse, r2_score, snr_recon)
```

---

## What Should We Do?

### Option 1: Replace NaNs with Zeros
```python
# Replace all NaN event metrics with 0
for metric in EVENT_METRICS:
    df[metric] = df[metric].fillna(0)
```

**Pros:**
- Keeps all 92,242 neurons
- ML model can handle 0 values
- Preserves LNOF neurons with partial data

**Cons:**
- 0 doesn't mean "no events" - it means "failed to compute"
- May confuse model (is 0 a real value or a failure marker?)
- Complete failures have invalid data (not just 0)

---

### Option 2: Remove Failed Neurons
```python
# Remove all neurons with ANY NaN event metric
has_nan = df[EVENT_METRICS].isna().any(axis=1)
df_cleaned = df[~has_nan]
```

**Pros:**
- Clean dataset (no ambiguous values)
- Only removes 227 neurons (0.25%)
- Removes genuinely broken traces

**Cons:**
- Loses 122 KEEP neurons (including 50 good LNOF neurons with partial data)
- Reduces training data slightly

---

### Recommendation: Hybrid Approach

1. **Keep partial failures** (50 LNOF neurons)
   - They have valid signal metrics
   - Just replace NaN reconstruction metrics with 0
   - These are real neurons with real events detected

2. **Remove complete failures** (177 neurons)
   - ALL metrics are NaN (unreliable)
   - Only loses 177/92,242 = 0.19% of data
   - Removes 105 DELETE + 72 KEEP neurons

```python
# Keep neurons where at least SOME event metrics were computed
complete_failure = df['kinetics_source'] == 'error'
df_cleaned = df[~complete_failure].copy()

# Replace remaining NaNs with 0 (partial failures only)
df_cleaned[EVENT_METRICS] = df_cleaned[EVENT_METRICS].fillna(0)
```

**Result:**
- 92,065 neurons (99.81% retained)
- No NaN values in event metrics
- Keeps neurons with valid partial data
- Removes only genuinely failed traces
