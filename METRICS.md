# BEARMiND ML Metrics Reference

Comprehensive guide to all 35+ features used in the EBM neuron classification model (v8_iter5).
Metrics are ordered by **global feature importance** from highest to lowest impact on model predictions.

**Importance Scale**: Higher values = more influential in model decisions (top features range 0.20-0.37)

---

## TOP TIER: Critical Discriminators (Importance > 0.20)

### 1. trace_skewness (0.366) - **MOST IMPORTANT**
**Category**: Trace Statistics
**Computation**: `scipy.stats.skew(trace)` on raw calcium trace

**What it measures**: Asymmetry of the calcium trace distribution around the mean.

**Why it matters**:
- Real neurons: **High positive skewness** (long right tail from infrequent calcium spikes above baseline)
- Artifacts: Low or negative skewness (symmetric noise, saturated signals)
- Biological interpretation: Real neurons spend most time at baseline with rare excursions to peak

**Typical values**:
- Real neurons: 2.0-10.0 (strong positive skew)
- Artifacts: -1.0 to 1.0 (weak or negative skew)

---

### 2. trace_kurtosis (0.315)
**Category**: Trace Statistics
**Computation**: `scipy.stats.kurtosis(trace, fisher=True)` on raw calcium trace

**What it measures**: "Tailedness" of the trace distribution - how heavy the tails are compared to normal distribution.

**Why it matters**:
- Real neurons: **High kurtosis** (heavy tails from extreme spike events)
- Artifacts: Low kurtosis (light tails, more uniform distribution)
- Biological interpretation: Real calcium transients create outlier peaks, not Gaussian noise

**Typical values**:
- Real neurons: 5.0-50.0+ (leptokurtic, heavy tails)
- Artifacts: -2.0 to 2.0 (platykurtic or mesokurtic)

**Related**: See `bimodality` coefficient which combines skewness and kurtosis

---

### 3. event_r2_score (0.304)
**Category**: Reconstruction Quality (Event-Based)
**Computation**: `neuron.get_reconstruction_r2(event_only=True)` via DRIADA

**What it measures**: R² goodness-of-fit between observed trace and reconstructed trace **during detected events only**, ignoring baseline.

**Why it matters**:
- Real neurons: **High event R²** (model captures event dynamics well)
- Artifacts: Low event R² (noisy/irregular events that don't fit physiological model)
- More specific than full trace R² - focuses on biologically meaningful signals

**Typical values**:
- Real neurons: 0.7-0.95 (good fit during events)
- Artifacts: 0.0-0.5 (poor fit, non-physiological dynamics)

**Computation details**:
```
event_r2 = 1 - SS_residual(events) / SS_total(events)
where events = timepoints where spike activity detected
```

---

### 4. area (0.296)
**Category**: Spatial Morphology
**Computation**: `calculate_polygon_area(contour_coords)` using Shoelace formula

**What it measures**: Spatial footprint size in pixels.

**Why it matters**:
- Real neurons: **Moderate size** (50-500 pixels, ~10-30 µm diameter)
- Too small: Noise artifacts, single bright pixels
- Too large: Merged neurons, blood vessels, neuropil contamination

**Typical values**:
- Real neurons: 100-400 pixels
- Artifacts: <50 or >800 pixels

**Note**: Highly experiment-dependent (FOV magnification, pixel size)

---

### 5. hurst_exponent (0.230)
**Category**: Temporal Characteristics
**Computation**: R/S (Rescaled Range) analysis on raw trace

**What it measures**: Long-range dependence / persistence in time series.

**Why it matters**:
- H=0.5: Random walk (white noise)
- H>0.5: Persistent (positive autocorrelation, trends)
- H<0.5: Anti-persistent (mean-reverting)

**Biological interpretation**:
- Real neurons: **H ≈ 0.5-0.7** (mild persistence from calcium dynamics, not pure noise)
- Artifacts: H<0.3 (anti-persistent, digitization noise) or H>0.8 (overly smooth, saturated)

**Typical values**:
- Real neurons: 0.4-0.7
- Noise artifacts: 0.1-0.3
- Saturation artifacts: 0.8-1.0

---

### 6. event_snr (0.226)
**Category**: Signal Quality (Event-Based)
**Computation**: `neuron.get_wavelet_snr()` via DRIADA, then log-transformed: `log1p(snr)`

**What it measures**: Signal-to-noise ratio of detected calcium events using wavelet analysis.

**Why it matters**:
- Real neurons: **High SNR** (clear events above noise floor)
- Artifacts: Low SNR (barely detectable or spurious events)

**Log transformation**: Applied because raw SNR can reach extreme values (millions for corrupted data).
```python
event_snr = np.log1p(raw_snr)  # log(1 + x) handles zeros gracefully
```

**Typical values** (after log transform):
- Real neurons: 3.0-6.0 (exp: 20-400 raw SNR)
- Artifacts: 0.0-2.0 (exp: 0-7 raw SNR)

---

### 7. t_rise (0.206)
**Category**: Event Kinetics
**Computation**: Measured from event regions by DRIADA, converted to seconds: `t_rise/fps`

**What it measures**: Rise time of calcium transients (baseline to peak), in seconds.

**Why it matters**:
- Real neurons: **Biologically plausible rise times** (0.05-0.3s for GCaMP indicators)
- Too fast: Noise artifacts, digitization errors
- Too slow: Neuropil, saturated peaks, photobleaching

**Typical values**:
- Real neurons: 0.08-0.25 seconds
- Artifacts: <0.03s (noise) or >0.5s (neuropil)

**Note**: Depends on calcium indicator type (GCaMP6f vs GCaMP6s)

---

### 8. ellipse_r (0.205)
**Category**: Spatial Position (Edge Artifact Detection)
**Computation**: Normalized radial distance from center of mass of all neurons

**What it measures**: How far a neuron is from the population center, normalized by FOV dimensions.

**Formula**:
```
ellipse_r = sqrt((dx/(fov_width/2))² + (dy/(fov_height/2))²)
where:
  dx = neuron_x - mean(all_neurons_x)
  dy = neuron_y - mean(all_neurons_y)
```

**Interpretation**:
- ellipse_r = 0.0: At center of population
- ellipse_r = 1.0: On ellipse touching FOV edges
- ellipse_r > 1.0: Beyond that ellipse (likely edge artifact)

**Why it matters**:
- Real neurons: **Centrally located** (r < 0.9)
- Edge artifacts: High r (near FOV boundaries where laser power drops, vignetting occurs)

**Typical values**:
- Real neurons: 0.2-0.8
- Edge artifacts: 0.9-1.2

---

### 9. half_crossing_rate (0.203)
**Category**: Temporal Characteristics
**Computation**: Count of threshold crossings at 0.5 (on normalized trace), rate per minute

**What it measures**: How many times the normalized trace crosses the 0.5 threshold per minute.

**Why it matters**:
- Real neurons: **Low HCR** (sparse calcium events, mostly at baseline)
- Artifacts: High HCR (noisy/continuous activity, flickering)

**Typical values**:
- Real neurons: 5-30 crossings/min (sparse events)
- Noise artifacts: 50-200+ crossings/min (constant flickering)

**Computation details**:
```python
trace_norm = (trace - min) / (max - min)  # normalize [0, 1]
above_half = trace_norm > 0.5
crossings = sum(abs(diff(above_half)))
HCR = crossings / duration_minutes
```

---

### 10. nn_distance_center (0.194)
**Category**: Spatial Relationships
**Computation**: Euclidean distance to nearest neighbor center (from center-to-center distance matrix)

**What it measures**: Distance to the closest other neuron (in pixels).

**Why it matters**:
- Real neurons: **Adequate spacing** (10-50 pixels between neurons)
- Merged neurons: Very low distance (<5 pixels, overlapping)
- Isolated artifacts: Extremely high distance (>100 pixels, no neighbors)

**Typical values**:
- Real neurons: 15-60 pixels
- Merged neurons: 0-10 pixels
- Isolated artifacts: 80-200+ pixels

---

## HIGH TIER: Strong Discriminators (Importance 0.10-0.19)

### 11. mean_time_at_peak (0.184)
**Category**: Temporal Characteristics (Saturation Detection)
**Computation**: Average time spent within 80% of peak value across all peaks (Gaussian smoothed, σ=3)

**What it measures**: Duration at peak plateau in seconds.

**Why it matters**:
- Real neurons: **Brief peaks** (0.03-0.17s at peak, normal exponential decay)
- Saturated artifacts: Long plateaus (>0.5s, indicator saturates)

**Typical values**:
- Real neurons: 0.05-0.20 seconds
- Saturated artifacts: 0.5-3.0+ seconds

**Biological interpretation**: GCaMP indicators have fast decay kinetics; prolonged plateaus indicate saturation or photobleaching

---

### 12. peak_amplitude_cv (0.160)
**Category**: Event Consistency
**Computation**: Coefficient of variation of detected event peak amplitudes

**What it measures**: Consistency of calcium event amplitudes.

**Formula**:
```
peak_amplitude_cv = std(peak_amplitudes) / mean(peak_amplitudes)
```

**Why it matters**:
- Real neurons: **Consistent amplitudes** (low CV, similar AP-evoked calcium transients)
- Artifacts: Wildly varying amplitudes (high CV, random noise spikes)

**Typical values**:
- Real neurons: 0.2-0.6
- Artifacts: 1.0-5.0+

---

### 13. baseline_drift (0.140)
**Category**: Temporal Characteristics
**Computation**: Normalized linear trend: `abs(linregress(trace).slope) / ptp(trace)`

**What it measures**: Linear drift in baseline over recording.

**Why it matters**:
- Real neurons: **Stable baseline** (low drift, <0.1)
- Photobleaching: High drift (negative slope)
- Saturation/rundown: High drift (positive or negative slope)

**Typical values**:
- Real neurons: 0.0-0.15
- Photobleaching: 0.2-0.8+

---

### 14. circularity (0.138)
**Category**: Spatial Morphology
**Computation**: `4π × area / perimeter²`

**What it measures**: How circular the footprint is.

**Why it matters**:
- circularity = 1.0: Perfect circle
- circularity < 0.5: Elongated (dendrites, axons, merged neurons)

**Typical values**:
- Real neurons: 0.6-0.9 (roughly circular somas)
- Dendrites/axons: 0.2-0.5 (elongated)

---

### 15. noise_level (0.135)
**Category**: Signal Quality (CaImAn)
**Computation**: CaImAn's `neurons_sn` attribute (noise standard deviation estimate)

**What it measures**: Background noise level in raw trace (from CaImAn's noise estimation).

**Why it matters**:
- Real neurons: **Low noise** (clean signal extraction)
- Poor ROIs: High noise (contaminated by neuropil, neighboring neurons)

**Typical values**:
- Real neurons: 5-30 (arbitrary CaImAn units)
- Noisy ROIs: 50-200+

---

### 16. r2_score (0.131)
**Category**: Reconstruction Quality (Full Trace)
**Computation**: `neuron.get_reconstruction_r2()` - R² for entire trace including baseline

**What it measures**: How well reconstructed trace fits observed trace (full recording).

**Why it matters**:
- Real neurons: **Good fit** (0.6-0.95, physiological model explains data)
- Artifacts: Poor fit (0.0-0.4, non-physiological dynamics)

**Comparison**: Less discriminative than `event_r2_score` because baseline dominates

**Typical values**:
- Real neurons: 0.65-0.92
- Artifacts: 0.0-0.5

---

### 17. snr_recon (0.130)
**Category**: Reconstruction Quality
**Computation**: `neuron.get_snr_reconstruction()` - SNR of reconstructed vs residual signal

**What it measures**: Signal-to-noise ratio from reconstruction perspective.

**Why it matters**:
- Real neurons: **High reconstruction SNR** (residuals are small/random)
- Artifacts: Low SNR (large systematic residuals, model doesn't fit)

**Typical values**:
- Real neurons: 8-30
- Artifacts: 0-5

---

### 18. local_density (0.125)
**Category**: Spatial Relationships
**Computation**: Count of neurons within 50-pixel radius (excluding self)

**What it measures**: Number of nearby neurons.

**Why it matters**:
- Real neurons: **Moderate density** (5-20 neighbors in healthy tissue)
- Isolated artifacts: 0-2 neighbors (far from main cluster)
- Dense packing: 25+ neighbors (may indicate over-segmentation)

**Typical values**:
- Real neurons: 5-20
- Isolated: 0-3
- Over-segmented: 25+

---

### 19. bimodality (0.115)
**Category**: Trace Statistics
**Computation**: Bimodality coefficient: `(skewness² + 1) / (kurtosis + 3)`

**What it measures**: Whether trace values cluster at two distinct levels (bimodal distribution).

**Formula**:
```
BC = (skewness² + 1) / (kurtosis + 3)
```

**Interpretation**:
- BC > 0.555: Suggests bimodality (uniform distribution = 0.555)
- BC < 0.555: Unimodal distribution

**Why it matters**:
- High bimodality: Values cluster at baseline and saturated peak levels (indicator saturation)
- Low bimodality: Natural unimodal distribution (healthy calcium dynamics)

**Typical values**:
- Real neurons: 0.3-0.55
- Saturated: 0.6-0.8+

---

### 20. events_fraction (0.103)
**Category**: Event-Based
**Computation**: Fraction of timepoints classified as "in event"

**What it measures**: What proportion of recording time is spent in detected calcium events.

**Why it matters**:
- Real neurons: **Low fraction** (0.05-0.30, sparse spiking)
- Continuous activity: High fraction (0.5-0.9, non-physiological)
- Silent neurons: Very low fraction (<0.02, may be inactive)

**Typical values**:
- Real neurons: 0.08-0.35
- Continuous artifacts: 0.6-0.95
- Silent: 0.0-0.05

---

### 21. footprint_compactness (0.102)
**Category**: Spatial Morphology
**Computation**: `area / convex_hull_area`

**What it measures**: How well the footprint fills its convex hull.

**Why it matters**:
- compactness = 1.0: Footprint perfectly fills convex hull (no holes)
- compactness < 0.5: Fragmented, ring-like (dendrites, hollow structures)

**Typical values**:
- Real neurons: 0.7-0.95 (solid soma)
- Dendrites: 0.3-0.6 (branched structure)

---

## MEDIUM TIER: Supporting Features (Importance 0.05-0.10)

### 22. baseline (0.098)
**Category**: Signal Quality (CaImAn)
**Computation**: CaImAn's `bl` attribute (estimated baseline fluorescence)

**What it measures**: Baseline fluorescence level (F₀).

**Why it matters**:
- Abnormally high baseline: Neuropil contamination, autofluorescence
- Abnormally low baseline: Background subtraction artifacts

**Typical values**: Experiment-dependent (arbitrary fluorescence units)

---

### 23. events_per_min (0.089)
**Category**: Event-Based
**Computation**: Count of detected events divided by recording duration in minutes

**What it measures**: Event rate (spikes per minute proxy).

**Why it matters**:
- Real neurons: **Moderate activity** (0.5-10 events/min)
- Hyperactive: >15 events/min (may be artifact or very active neuron)
- Silent: <0.2 events/min (inactive or poor detection)

**Typical values**:
- Real neurons: 1-8 events/min
- Artifacts: 0 or 20+

---

### 24. tau_decay (0.081)
**Category**: Temporal Characteristics (Kinetics)
**Computation**: Calcium decay time constant from CaImAn's autoregressive parameter: `tau = -1/log(g) / fps`

**What it measures**: Calcium indicator decay time in seconds.

**Why it matters**:
- Real neurons: **Biologically plausible tau** (0.3-3.0s for GCaMP)
- Anomalous: <0.1s (noise) or >5s (neuropil, slow rundown)

**Typical values**:
- GCaMP6f: 0.4-1.2 seconds
- GCaMP6s: 1.5-3.5 seconds

---

### 25. nmae (0.079)
**Category**: Reconstruction Quality
**Computation**: Normalized Mean Absolute Error: `mean(abs(observed - reconstructed)) / ptp(observed)`

**What it measures**: Average absolute reconstruction error, normalized by signal range.

**Why it matters**:
- Real neurons: **Low error** (0.05-0.20)
- Poor fit: High error (0.35+)

**Typical values**:
- Real neurons: 0.08-0.22
- Artifacts: 0.30-0.60+

---

### 26. edge_distance (0.078)
**Category**: Spatial Position
**Computation**: Minimum distance from neuron center to FOV boundary, normalized by half-FOV-size

**What it measures**: Relative distance to nearest edge.

**Why it matters**:
- edge_distance = 0.0: At FOV boundary (likely edge artifact)
- edge_distance = 0.5: At FOV center (maximum distance from edges)

**Typical values**:
- Real neurons: 0.15-0.45
- Edge artifacts: 0.0-0.10

**Note**: Complements `ellipse_r` for edge detection

---

### 27. t_off (0.071)
**Category**: Event Kinetics
**Computation**: Decay time of calcium transients (peak to baseline), in seconds

**What it measures**: Fall time of calcium transients.

**Why it matters**:
- Real neurons: **Biologically plausible decay** (0.1-0.5s)
- Anomalous: <0.05s (noise) or >1.0s (neuropil)

**Typical values**:
- Real neurons: 0.15-0.50 seconds
- Artifacts: <0.08s or >1.0s

**Comparison**: Less important than `t_rise` because decay is more variable

---

### 28. caiman_snr (0.070)
**Category**: Signal Quality (CaImAn)
**Computation**: CaImAn's `SNR_comp` attribute

**What it measures**: CaImAn's internal SNR estimate for the component.

**Why it matters**:
- Real neurons: **Moderate-high SNR** (5-30)
- Poor components: Low SNR (<3)

**Note**: Capped at max finite value to handle infinities

**Typical values**:
- Real neurons: 6-25
- Artifacts: 0-4

---

### 29. max_edge (0.068)
**Category**: Spatial Morphology
**Computation**: Maximum edge length of footprint polygon

**What it measures**: Longest single edge in footprint boundary.

**Why it matters**:
- Large max_edge: Elongated, irregular shapes (dendrites, merged neurons)
- Small max_edge: Compact, regular shapes

**Typical values**:
- Real neurons: 5-20 pixels
- Elongated: 25-60+ pixels

---

### 30. aspect_ratio (0.065)
**Category**: Spatial Morphology
**Computation**: Ratio of major to minor axis of fitted ellipse

**What it measures**: Elongation of footprint.

**Why it matters**:
- aspect_ratio ≈ 1.0: Round (soma)
- aspect_ratio > 2.0: Elongated (dendrites, axons)

**Typical values**:
- Real neurons: 1.2-2.0
- Dendrites: 2.5-5.0+

---

### 31. nrmse (0.053)
**Category**: Reconstruction Quality
**Computation**: Normalized Root Mean Squared Error

**What it measures**: RMS reconstruction error, normalized by signal range.

**Why it matters**:
- Real neurons: **Low RMSE** (0.08-0.25)
- Poor fit: High RMSE (0.40+)

**Typical values**:
- Real neurons: 0.10-0.28
- Artifacts: 0.35-0.70+

---

### 32. caiman_r_score (0.050)
**Category**: Signal Quality (CaImAn)
**Computation**: CaImAn's `r_values` attribute (spatial correlation)

**What it measures**: Spatial correlation of component footprint with raw movie.

**Why it matters**:
- Real neurons: **High correlation** (0.6-0.95)
- Spurious components: Low correlation (<0.5)

**Typical values**:
- Real neurons: 0.65-0.90
- Artifacts: 0.0-0.55

---

## LOW TIER: Minor Features (Importance < 0.05)

### 33. convexity (0.032)
**Category**: Spatial Morphology
**Computation**: `perimeter(convex_hull) / perimeter(actual)`

**What it measures**: How convex the footprint boundary is.

**Why it matters**:
- convexity = 1.0: Perfectly convex
- convexity < 0.7: Concave, irregular (may indicate merged neurons or dendrites)

**Typical values**:
- Real neurons: 0.80-0.98
- Irregular: 0.50-0.75

---

### 34. kinetics_opt (0.029)
**Category**: Event Kinetics (Quality Flag)
**Computation**: Success flag for kinetics optimization
- 1.0 = both t_rise and t_off measured successfully
- 0.5 = one parameter measured, one defaulted
- 0.0 = both defaulted (no events or optimization failed)

**What it measures**: Whether event kinetics were successfully measured.

**Why it matters**:
- kinetics_opt = 1.0: High confidence in t_rise/t_off values
- kinetics_opt = 0.0: No events or poor quality (t_rise/t_off unreliable)

**Typical values**:
- Real neurons: 0.5-1.0
- Poor/silent: 0.0

---

### 35. eccentricity (0.025)
**Category**: Spatial Morphology
**Computation**: CaImAn's eccentricity measure (from eigenvalues of spatial footprint)

**What it measures**: Elongation of footprint (similar to aspect_ratio).

**Why it matters**:
- eccentricity = 0.0: Perfect circle
- eccentricity = 1.0: Perfect line

**Typical values**:
- Real neurons: 0.3-0.7
- Elongated: 0.8-0.95

**Note**: Redundant with aspect_ratio, hence low importance

---

## INTERACTION TERMS (Top 20 Interactions)

The EBM model includes 20 pairwise interaction terms that capture non-linear relationships between features. Top interactions by importance:

1. **hurst_exponent & event_snr** (0.117): Long-range dependence × event quality
2. **area & trace_skewness** (0.112): Size × trace asymmetry
3. **local_density & trace_skewness** (0.108): Crowding × trace shape
4. **caiman_snr & event_r2_score** (0.091): CaImAn quality × reconstruction fit
5. **local_density & event_snr** (0.077): Crowding × event quality

Interactions allow the model to learn complex decision boundaries, e.g., "large area is acceptable if skewness is high, but not if skewness is low."

---

## METRIC CATEGORIES SUMMARY

### Trace Statistics (3 features)
Most discriminative category overall:
- trace_skewness (0.366) ⭐
- trace_kurtosis (0.315) ⭐
- bimodality (0.115)

### Event-Based (8 features)
Strong predictors of neuron quality:
- event_r2_score (0.304) ⭐
- event_snr (0.226) ⭐
- t_rise (0.206) ⭐
- events_fraction (0.103)
- events_per_min (0.089)
- peak_amplitude_cv (0.160)
- t_off (0.071)
- kinetics_opt (0.029)

### Spatial Morphology (10 features)
Identify shape anomalies:
- area (0.296) ⭐
- circularity (0.138)
- footprint_compactness (0.102)
- max_edge (0.068)
- aspect_ratio (0.065)
- convexity (0.032)
- eccentricity (0.025)

### Spatial Position (3 features)
Detect edge artifacts:
- ellipse_r (0.205) ⭐
- nn_distance_center (0.194) ⭐
- edge_distance (0.078)
- local_density (0.125)

### Temporal Characteristics (4 features)
Capture temporal dynamics:
- hurst_exponent (0.230) ⭐
- half_crossing_rate (0.203) ⭐
- mean_time_at_peak (0.184)
- baseline_drift (0.140)
- tau_decay (0.081)

### Signal Quality (5 features)
CaImAn-derived metrics:
- noise_level (0.135)
- caiman_snr (0.070)
- caiman_r_score (0.050)
- baseline (0.098)

### Reconstruction Quality (5 features)
Model fit metrics:
- event_r2_score (0.304) ⭐ [also in Event-Based]
- r2_score (0.131)
- snr_recon (0.130)
- nmae (0.079)
- nrmse (0.053)

---

## MODEL CONFIGURATION (v8_iter5)

**Algorithm**: ExplainableBoostingClassifier (Microsoft InterpretML)
**Features**: 35 main terms + 20 interaction terms
**Decision threshold**: 0.75 (optimized for 3:1 precision:recall ratio)

**Performance @ threshold=0.75**:
- Precision: 94.2% (few false positives)
- Recall: 91.8% (catches most real neurons)
- F-beta (β=0.577): 92.4%

**Importance distribution**:
- Top 10 features account for 62% of total importance
- Top 20 features account for 85% of total importance
- Interaction terms contribute 15% of total importance

---

## USAGE NOTES

### For Users (Neuroscientists)
1. **Trust the top 10**: These metrics drive most decisions
2. **Check outliers**: Neurons with extreme values in top metrics often need manual review
3. **Context matters**: Some metrics are experiment-specific (e.g., area depends on magnification)

### For Developers (ML Engineers)
1. **Feature engineering**: Top metrics were refined over 8 iterations
2. **Interaction selection**: 20 interactions chosen via grid search (from 500+ candidates)
3. **Hybrid kinetics**: t_rise/t_off use cascading optimization (wavelet → threshold) for robustness
4. **HCR**: Added post-facto in v8 iteration; requires raw trace access

### For Debugging
If model predictions seem wrong:
1. Check **trace_skewness** - Most important single feature
2. Check **event_r2_score** - Reconstruction quality
3. Check **ellipse_r** - Edge position
4. Check **area** - Size sanity check

---

## CHANGELOG

**v8_iter5** (2025-12-22, current production):
- Added half_crossing_rate metric (0.203 importance)
- Hybrid kinetics optimization for t_rise/t_off
- 20 interaction terms (grid searched from 500+ candidates)
- Decision threshold optimized to 0.75

**v7** (previous):
- Standard kinetics, 10 interaction terms

**v6** (earlier):
- First version with interaction terms

**v1-v5**:
- Iterative feature engineering and quality improvements
