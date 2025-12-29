# EBM v9 Iteration 8 - Production Model

**Model Path**: `production_models/ebm_v9_iter8.pkl`

**Date**: 2025-12-29
**Status**: Production Ready
**Recommended for**: All neuron quality assessment tasks

---

## Model Overview

This is the best-performing model from the v9 iterative improvement workflow, trained through 8 iterations with cumulative label corrections based on expert review.

### Key Statistics

- **Training dataset**: 92,242 neurons (1,129 cumulative corrections)
- **Training seed**: 46
- **Hyperparameters**:
  - max_bins: 1024
  - interactions: 20
  - max_leaves: 3
  - min_samples_leaf: 5
  - outer_bags: 8
  - learning_rate: 0.01

---

## Performance Metrics

### Cross-Validation Results (10 splits, threshold=0.75)

| Metric    | Mean    | Std     |
|-----------|---------|---------|
| F-beta    | 0.9650  | ±0.0037 |
| AUC       | 0.9782  | ±0.0053 |
| Precision | 0.9763  | ±0.0029 |
| Recall    | 0.9326  | ±0.0069 |
| Accuracy  | 0.9274  | ±0.0082 |

### Optimal Threshold Analysis

**Recommended threshold: 0.70-0.72**

At threshold=0.70 (CV results):
- F-beta: 0.9659 ±0.0036
- Precision: 0.9729 ±0.0030
- Recall: 0.9455 ±0.0064

At threshold=0.75 (current default):
- F-beta: 0.9650 ±0.0037
- Precision: 0.9763 ±0.0029
- Recall: 0.9326 ±0.0069

---

## Comparison to Previous Best (v8_iter5)

### Statistical Significance (paired t-test, 10 CV splits)

| Metric    | v8_iter5 | v9_iter8 | Improvement | p-value  | Significant? |
|-----------|----------|----------|-------------|----------|--------------|
| F-beta    | 0.9605   | 0.9650   | +0.45%      | 0.0014   | YES ✓        |
| AUC       | 0.9674   | 0.9782   | +1.08%      | <0.0001  | YES ✓✓       |
| Precision | 0.9642   | 0.9763   | +1.21%      | -        | -            |
| Recall    | 0.9497   | 0.9326   | -1.71%      | -        | -            |

**Conclusion**: v9_iter8 is statistically significantly better than v8_iter5 (p<0.01)

---

## Usage Guidelines

### Loading the Model

```python
import pickle

with open('production_models/ebm_v9_iter8.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Making Predictions

```python
# Get probabilities
y_proba = model.predict_proba(X)[:, 1]

# Classify with recommended threshold
threshold = 0.72
y_pred = (y_proba >= threshold).astype(int)

# 1 = KEEP, 0 = DELETE
```

### Threshold Selection

- **Conservative (fewer false positives)**: threshold=0.75
  - Higher precision (0.9763), lower recall (0.9326)
  - Use when false positives are costly

- **Balanced (recommended)**: threshold=0.70-0.72
  - Best F-beta (0.9659)
  - Good balance of precision and recall

- **Sensitive (fewer false negatives)**: threshold=0.65-0.70
  - Higher recall, slightly lower precision
  - Use when missing good neurons is worse than keeping bad ones

---

## Training History

### Iterative Improvement Process

| Iteration | Seed | Real Errors | Corrections | CV F-beta | CV AUC  |
|-----------|------|-------------|-------------|-----------|---------|
| Iter 5    | 43   | 16 (14+2)   | 624         | 0.9605    | 0.9685  |
| Iter 6    | 44   | 23 (20+3)   | 798         | 0.9618    | 0.9728  |
| Iter 7    | 45   | 29 (26+3)   | 964         | 0.9636    | 0.9763  |
| **Iter 8** | **46** | **TBD** | **1129**   | **0.9650** | **0.9782** |

### Key Milestones

1. **v8 → v9 transition**: Improved dataset quality
2. **Seed rotation**: Used different seeds (43-46) to prevent overfit
3. **Label corrections**: 1,129 cumulative corrections across 8 iterations
4. **Threshold optimization**: Identified 0.70-0.72 as optimal range

---

## Feature Importance

Top features (from model):
1. Spatial features (footprint quality)
2. Temporal features (trace characteristics)
3. Event detection metrics
4. Correlation features
5. SNR and quality metrics

(Full feature importance available in `ml/ebm_v9_iter8/feature_importance.csv`)

---

## Validation Results

### Test Set Performance (seed=46)

- TP: 18,018
- FP: 515
- FN: 1,622
- TN: 3,429

- Precision: 0.9722
- Recall: 0.9174
- F-beta: 0.9579
- AUC: 0.9655

### Cross-Validation Stability

- Low variance across splits (std < 0.007 for all metrics)
- Consistent performance across different session combinations
- No significant experiment-specific biases detected

---

## Known Limitations

1. **Recall trade-off**: Model is slightly conservative (recall ~0.93)
   - Prioritizes precision over recall
   - May miss ~7% of true good neurons at threshold=0.75

2. **MERGE cases**: Model cannot distinguish spatial duplicates
   - Requires post-processing spatial filtering
   - ~10% of FP errors are merge cases (< 5px apart)

3. **Training data bias**: Primarily trained on NOF, RFC, FOF, LNOF experiments
   - May need recalibration for new experiment types

---

## Maintenance

### When to Retrain

- New experiment types added to dataset
- Significant changes in imaging protocol
- Accumulation of >100 new expert corrections
- Detection of systematic errors in specific sessions

### Recommended Workflow

1. Collect new expert-reviewed data
2. Apply corrections to dataset
3. Retrain with next seed in sequence (47, 48, ...)
4. Validate with cross-validation
5. Compare to current production model
6. Update if statistically significant improvement

---

## Files

- `ebm_v9_iter8.pkl` - Model pickle file
- `ebm_v9_iter8_summary.json` - Training summary
- `ebm_v9_iter8_README.md` - This file

**Related files**:
- Training dataset: `ml/results/training_dataset_v9_corrected_iter7.csv`
- CV results: `ml/results/cv_analysis_iter5_to_iter8.csv`
- PR curves: `ml/results/pr_tradeoff_cv_v8_v9.png`
- Comparison: `ml/results/cv_comparison_v8_iter5_vs_v9_iter8.csv`

---

## Contact

For questions or issues with this model, refer to:
- Model training logs: `ml/ebm_v9_iter8/`
- Cross-validation analysis: `ml/cross_validate_iters5_8.py`
- Threshold optimization: `ml/optimize_threshold.py`

**Last updated**: 2025-12-29
