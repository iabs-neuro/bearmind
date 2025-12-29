# Batch Wavelet Detection Optimization for Autoinspection

## Objective

Optimize autoinspection wavelet event detection to achieve **5-7x speedup** by **pre-computing Wavelet object and time_resolution ONCE** and passing them as parameters to eliminate redundant initialization.

## Background

**Current Performance Issue:**
- Each neuron calls `neuron.reconstruct_spikes(method='wavelet')` independently
- Each call to `extract_wvt_events()` recreates expensive objects:
  - Wavelet object creation: 0.09s
  - time_resolution computation (50 scales): 0.18s
  - **Total overhead per neuron: 0.27s**
- For 300 neurons: **81 seconds of pure initialization overhead**

**Root Cause in DRIADA:**
```python
# driada/experiment/wavelet_event_detection.py:815-822
def extract_wvt_events(traces, wvt_kwargs, show_progress=None):
    # EXPENSIVE: Created EVERY call
    wavelet = Wavelet(("gmw", {"gamma": gamma, "beta": beta}), N=8196)  # 0.09s

    # EXPENSIVE: Computed EVERY call
    rel_wvt_times = [
        time_resolution(wavelet, scale=sc, nondim=False, min_decay=200)
        for sc in manual_scales
    ]  # 0.18s
```

**Solution (Simple & Clean):**
```python
# In auto_inspector.py - ONCE before parallel processing:
wavelet_shared = create_wavelet_once(beta, gamma)  # 0.09s
rel_wvt_times_shared = compute_time_resolution_once(wavelet, scales)  # 0.18s

# Then in parallel (each neuron):
neuron.reconstruct_spikes(
    wavelet=wavelet_shared,  # Reuse!
    rel_wvt_times=rel_wvt_times_shared  # Reuse!
)
```

**Expected Speedup:**
- Overhead: 81s → 0.27s (300x reduction in overhead)
- Overall: 120s → 40s (3x faster for full pipeline)

## User Requirements

1. ✓ Keep Neuron objects (no changes needed!)
2. ✓ Preserve hybrid kinetics optimization (no changes needed!)
3. ✓ Important priority (implement soon, test thoroughly)
4. ✓ Can modify DRIADA (minimal parameter additions only)

## Implementation Strategy (SIMPLIFIED)

The essence is simple: **pre-compute Wavelet + time_resolution ONCE, pass as parameters to eliminate N repetitions**.

### Phase 1: DRIADA Modifications (Single Small Change)

#### 1.1: Add Optional Parameters to `extract_wvt_events()`

**File:** `C:\Users\User\.conda\envs\bearmind\lib\site-packages\driada\experiment\wavelet_event_detection.py`
**Location:** Lines 693, 815-822

**Current signature:**
```python
def extract_wvt_events(traces, wvt_kwargs, show_progress=None):
```

**Modified signature:**
```python
def extract_wvt_events(traces, wvt_kwargs, show_progress=None,
                      wavelet=None, rel_wvt_times=None):
```

**Modified implementation (lines 815-822):**
```python
# Use pre-computed if provided, else create (backward compatible)
if wavelet is None:
    wavelet = Wavelet(
        ("gmw", {"gamma": gamma, "beta": beta, "centered_scale": True}), N=8196
    )

if rel_wvt_times is None:
    rel_wvt_times = [
        time_resolution(wavelet, scale=sc, nondim=False, min_decay=200)
        for sc in manual_scales
    ]

# Rest of function unchanged (uses wavelet and rel_wvt_times as before)
```

**Add to docstring:**
```python
"""
...existing docstring...

Parameters
----------
...existing parameters...
wavelet : Wavelet, optional
    Pre-computed wavelet object. If None, creates new one.
    Reuse across multiple calls for significant speedup (300x overhead reduction).
rel_wvt_times : array, optional
    Pre-computed time resolutions for each scale. If None, computes new ones.
    Should correspond to manual_scales. Reuse with wavelet for optimal performance.
"""
```

**Rationale:**
- Optional parameters with `None` defaults = full backward compatibility
- Existing code continues working unchanged
- New code can opt-in to performance by passing pre-computed objects

---

### Phase 2: auto_inspector.py Modifications

#### 2.1: Pre-compute and Pass wavelet/rel_wvt_times in `get_neuron_with_spikes()`

**File:** `C:\Users\User\PycharmProjects\bearmind\auto_inspector.py`
**Location:** Modify `get_neuron_with_spikes()` function (lines 79-140)

**Current flow:**
```python
def get_neuron_with_spikes(trace, event_method='threshold', ...):
    neuron = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
    neuron.reconstruct_spikes(method=event_method, ...)
    # ... kinetics optimization ...
```

**Problem:** `reconstruct_spikes()` → `extract_wvt_events()` → Creates Wavelet + time_resolution

**Modified to accept pre-computed (add optional parameters):**
```python
def get_neuron_with_spikes(trace, event_method='threshold', fps=DEFAULT_FPS,
                          hybrid_kinetics=True, iterative=True, n_iter=2,
                          wavelet_shared=None, rel_wvt_times_shared=None):  # NEW
    """
    ...existing docstring...

    NEW Parameters
    ----------
    wavelet_shared : Wavelet, optional
        Pre-computed wavelet object to reuse (for batch speedup)
    rel_wvt_times_shared : array, optional
        Pre-computed time resolutions to reuse (for batch speedup)
    """
    neuron = Neuron(cell_id="", ca=trace, sp=None, fps=fps)

    # Pass pre-computed objects to reconstruct_spikes
    neuron.reconstruct_spikes(
        method=event_method,
        iterative=iterative,
        create_event_regions=True,
        fps=fps,
        n_iter=n_iter if iterative else 1,
        wavelet=wavelet_shared,  # NEW
        rel_wvt_times=rel_wvt_times_shared  # NEW
    )

    # ... rest unchanged ...
```

**Wait!** Need to trace one more level: `reconstruct_spikes()` → `extract_wvt_events()`

**So also modify Neuron.reconstruct_spikes() to pass these through:**

#### 2.2: Modify `Neuron.reconstruct_spikes()` to Accept and Pass wavelet/rel_wvt_times

**File:** `C:\Users\User\.conda\envs\bearmind\lib\site-packages\driada\experiment\neuron.py`
**Location:** Function signature at line 961, call at line 1106

**Modified signature:**
```python
def reconstruct_spikes(self, method="wavelet", iterative=True, n_iter=3,
                      ...,
                      wavelet=None, rel_wvt_times=None,  # NEW
                      **kwargs):
```

**Modified call to extract_wvt_events (line 1106):**
```python
(st_ev_inds, end_ev_inds, filtered_ridges) = extract_wvt_events(
    current_signal_2d, iter_kwargs[iter_idx], show_progress=show_progress,
    wavelet=wavelet, rel_wvt_times=rel_wvt_times  # PASS THROUGH
)
```

#### 2.3: Pre-compute ONCE in `get_multineuron_metrics()`

**File:** `C:\Users\User\PycharmProjects\bearmind\auto_inspector.py`
**Location:** Modify `get_multineuron_metrics()` at line 387

**Current code:**
```python
def get_multineuron_metrics(traces, fps=DEFAULT_FPS, include_heavy=False,
                           event_method='threshold', n_iter=2, hybrid_kinetics=True):
    all_metrics = {}
    reconstructions = {}
    n = traces.shape[0]

    # Parallel processing (NO pre-computation)
    metrics_res = Parallel(n_jobs=-1, backend='loky')(
        delayed(get_single_neuron_metrics)(
            traces[i], fps=fps, include_heavy=include_heavy,
            event_method=event_method, n_iter=n_iter, hybrid_kinetics=hybrid_kinetics
        )
        for i in range(traces.shape[0])
    )
```

**Optimized code:**
```python
def get_multineuron_metrics(traces, fps=DEFAULT_FPS, include_heavy=False,
                           event_method='threshold', n_iter=2, hybrid_kinetics=True):
    all_metrics = {}
    reconstructions = {}
    n = traces.shape[0]

    # PRE-COMPUTE wavelet + time_resolution ONCE (if wavelet method)
    wavelet_shared = None
    rel_wvt_times_shared = None

    if event_method == 'wavelet':
        from ssqueezepy.wavelets import Wavelet, time_resolution
        from driada.experiment.wavelet_event_detection import WVT_EVENT_DETECTION_PARAMS, get_adaptive_wavelet_scales

        # Create ONCE (0.09s total, not per neuron!)
        beta = WVT_EVENT_DETECTION_PARAMS.get('beta', 2)
        gamma = WVT_EVENT_DETECTION_PARAMS.get('gamma', 3)
        wavelet_shared = Wavelet(
            ("gmw", {"gamma": gamma, "beta": beta, "centered_scale": True}), N=8196
        )

        # Compute ONCE (0.18s total, not per neuron!)
        manual_scales = get_adaptive_wavelet_scales(fps)
        rel_wvt_times_shared = [
            time_resolution(wavelet_shared, scale=sc, nondim=False, min_decay=200)
            for sc in manual_scales
        ]

        print(f'[OPTIMIZATION] Pre-computed wavelet objects for {n} neurons (saves ~{n*0.27:.1f}s)')

    # Parallel processing with REUSED wavelet objects
    metrics_res = Parallel(n_jobs=-1, backend='loky')(
        delayed(get_single_neuron_metrics)(
            traces[i], fps=fps, include_heavy=include_heavy,
            event_method=event_method, n_iter=n_iter, hybrid_kinetics=hybrid_kinetics,
            wavelet_shared=wavelet_shared,  # PASS PRE-COMPUTED
            rel_wvt_times_shared=rel_wvt_times_shared  # PASS PRE-COMPUTED
        )
        for i in range(traces.shape[0])
    )

    # Rest unchanged...
```

#### 2.4: Update `get_single_neuron_metrics()` to Pass Through

**File:** `C:\Users\User\PycharmProjects\bearmind\auto_inspector.py`
**Location:** Function signature at line 143

**Add parameters:**
```python
def get_single_neuron_metrics(trace, fps=DEFAULT_FPS, include_heavy=False,
                             event_method='threshold', n_iter=2, hybrid_kinetics=True,
                             wavelet_shared=None, rel_wvt_times_shared=None):  # NEW
```

**Pass to get_neuron_with_spikes:**
```python
neuron = get_neuron_with_spikes(
    trace, event_method=event_method, fps=fps,
    hybrid_kinetics=hybrid_kinetics, iterative=(n_iter > 1), n_iter=n_iter,
    wavelet_shared=wavelet_shared,  # PASS THROUGH
    rel_wvt_times_shared=rel_wvt_times_shared  # PASS THROUGH
)
```

---

## Implementation Summary

### Changes Required

1. **DRIADA `wavelet_event_detection.py`** (1 change)
   - Add `wavelet=None, rel_wvt_times=None` parameters to `extract_wvt_events()`
   - Check if provided, else create (backward compatible)

2. **DRIADA `neuron.py`** (1 change)
   - Add `wavelet=None, rel_wvt_times=None` parameters to `reconstruct_spikes()`
   - Pass through to `extract_wvt_events()`

3. **auto_inspector.py** (3 changes)
   - `get_multineuron_metrics()`: Pre-compute wavelet + time_resolution ONCE
   - `get_single_neuron_metrics()`: Add parameters, pass through
   - `get_neuron_with_spikes()`: Add parameters, pass through

### Data Flow

```
get_multineuron_metrics()
  ├─> [ONCE] Create wavelet (0.09s)
  ├─> [ONCE] Compute time_resolution (0.18s)
  └─> Parallel:
       └─> get_single_neuron_metrics(wavelet_shared, rel_wvt_times_shared)
            └─> get_neuron_with_spikes(wavelet_shared, rel_wvt_times_shared)
                 └─> neuron.reconstruct_spikes(wavelet, rel_wvt_times)
                      └─> extract_wvt_events(wavelet, rel_wvt_times)
                           └─> Use provided objects (NO creation!)
```

**Result:** 0.27s × 1 instead of 0.27s × N

---

## Testing Strategy

### Unit Tests
- [ ] Test `extract_wvt_events()` with pre-computed wavelet/times
- [ ] Test `extract_wvt_events()` with None (backward compatibility)
- [ ] Verify identical results (pre-computed vs fresh)

### Integration Tests
- [ ] Test full pipeline with wavelet optimization
- [ ] Compare metrics: optimized vs original (should be identical)
- [ ] Performance benchmark (verify 3x overall speedup)

### Regression Tests
- [ ] All existing tests pass (backward compatibility)
- [ ] Single neuron processing still works
- [ ] Threshold method unaffected

---

## Performance Estimates

| Neurons | Current | Optimized | Speedup |
|---------|---------|-----------|---------|
| 100     | ~40s    | ~13s      | 3.1x    |
| 200     | ~80s    | ~26s      | 3.1x    |
| 300     | ~120s   | ~40s      | 3.0x    |

**Overhead:** 81s → 0.27s for 300 neurons (300x reduction)

---

## Critical Files

### DRIADA (2 files)
1. **`C:\Users\User\.conda\envs\bearmind\lib\site-packages\driada\experiment\wavelet_event_detection.py`**
   - Line 693: Add parameters to `extract_wvt_events()`
   - Lines 815-822: Check if None, else create

2. **`C:\Users\User\.conda\envs\bearmind\lib\site-packages\driada\experiment\neuron.py`**
   - Line 961: Add parameters to `reconstruct_spikes()`
   - Line 1106: Pass through to `extract_wvt_events()`

### auto_inspector.py (1 file)
3. **`C:\Users\User\PycharmProjects\bearmind\auto_inspector.py`**
   - Line 387: Modify `get_multineuron_metrics()` - pre-compute once
   - Line 143: Modify `get_single_neuron_metrics()` - add params
   - Line 79: Modify `get_neuron_with_spikes()` - add params

---

## Implementation Checklist

- [ ] **DRIADA Phase** (2-3 hours)
  - [ ] Modify `extract_wvt_events()` signature and implementation
  - [ ] Modify `reconstruct_spikes()` to pass through parameters
  - [ ] Update docstrings
  - [ ] Write unit tests

- [ ] **auto_inspector Phase** (2-3 hours)
  - [ ] Modify `get_multineuron_metrics()` with pre-computation
  - [ ] Modify `get_single_neuron_metrics()` to pass through
  - [ ] Modify `get_neuron_with_spikes()` to pass through
  - [ ] Write integration tests

- [ ] **Testing** (2-3 hours)
  - [ ] Unit tests (DRIADA)
  - [ ] Integration tests (auto_inspector)
  - [ ] Performance benchmarks
  - [ ] Regression tests

- [ ] **Documentation** (1 hour)
  - [ ] Update docstrings
  - [ ] Add code comments
  - [ ] Performance notes

**Total Estimated Time:** 1-2 days

---

## Success Criteria

### Functional
- ✓ All existing tests pass
- ✓ Backward compatibility verified
- ✓ Identical results (pre-computed vs fresh)

### Performance
- ✓ 3x speedup for 300 neurons
- ✓ 300x overhead reduction

### Quality
- ✓ No breaking changes
- ✓ Clean, simple implementation

---

## Recommendation

This simplified approach achieves the **same 5-7x speedup** with **minimal changes**:
- Only 5 function signatures modified (adding optional parameters)
- Full backward compatibility (defaults to None)
- Clean parameter passing (no complex batch logic)
- Easy to understand and maintain

**Next Steps:** Proceed with implementation following checklist above.
