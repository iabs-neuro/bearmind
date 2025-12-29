"""Compare v8_iter5 and v9 models at a specific threshold."""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, 'ml')
from data_utils import compute_fbeta, FBETA_BETA

def evaluate_at_threshold(model, X, y, threshold):
    """Evaluate model at specific threshold."""
    # Use only features the model was trained on
    model_features = list(model.feature_names_in_)
    X_model = X[model_features].copy()
    y_proba = model.predict_proba(X_model)[:, 1]

    y_pred = (y_proba >= threshold).astype(int)

    tp = ((y_pred == 1) & (y == 1)).sum()
    fp = ((y_pred == 1) & (y == 0)).sum()
    tn = ((y_pred == 0) & (y == 0)).sum()
    fn = ((y_pred == 0) & (y == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fbeta = compute_fbeta(precision, recall)

    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta,
        'total_pred_positive': tp + fp,
        'total_pred_negative': tn + fn,
        'total_true_positive': tp + fn,
        'total_true_negative': tn + fp
    }

# Load models
print('='*80)
print('COMPARING v8_iter5 vs v9 AT THRESHOLD 0.75')
print('='*80)

with open('production_models/ebm_v8_corrected_iter5.pkl', 'rb') as f:
    v8_model = pickle.load(f)

with open('ml/ebm_grid_search_v9/best_model.pkl', 'rb') as f:
    v9_model = pickle.load(f)

print('\nLoaded models:')
print(f'  v8_iter5: {len(v8_model.feature_names_in_)} features')
print(f'  v9: {len(v9_model.feature_names_in_)} features')

# Load dataset
df = pd.read_csv('ml/results/training_dataset_v9.csv')
print(f'\nDataset: {len(df):,} neurons')

# Prepare features
label_col = 'ground_truth'
exclude_cols = {label_col, 'session_name', 'session', 'experiment', 'component_idx',
                'center', 'distance_to_gt', 'is_corner_artifact', 'corr_groups',
                'delete', 'merge', 'decision', 'failed_corner_artifact',
                'failed_area', 'failed_circularity', 'ml_keep_probability'}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
X = df[feature_cols].copy()
X = X.replace([np.inf, -np.inf], np.nan)
y = df[label_col].values

print(f'Features extracted: {len(feature_cols)}')
print(f'Target: {y.sum():,} KEEP, {len(y) - y.sum():,} DELETE')

# Evaluate both models at threshold 0.75
threshold = 0.75
v8_metrics = evaluate_at_threshold(v8_model, X, y, threshold)
v9_metrics = evaluate_at_threshold(v9_model, X, y, threshold)

print('\n' + '='*80)
print(f'RESULTS AT THRESHOLD {threshold}')
print('='*80)

print(f'\n{"Metric":<25} {"v8_iter5":>15} {"v9":>15} {"Difference":>15}')
print('-'*80)

# Performance metrics
print(f'{"F-beta":.<25} {v8_metrics["fbeta"]:>15.4f} {v9_metrics["fbeta"]:>15.4f} {v9_metrics["fbeta"]-v8_metrics["fbeta"]:>+15.4f}')
print(f'{"Precision":.<25} {v8_metrics["precision"]:>15.4f} {v9_metrics["precision"]:>15.4f} {v9_metrics["precision"]-v8_metrics["precision"]:>+15.4f}')
print(f'{"Recall":.<25} {v8_metrics["recall"]:>15.4f} {v9_metrics["recall"]:>15.4f} {v9_metrics["recall"]-v8_metrics["recall"]:>+15.4f}')

print('\n' + '-'*80)
print('Confusion Matrix:')
print('-'*80)

# Confusion matrix
print(f'{"True Positives (TP)":.<25} {v8_metrics["tp"]:>15,} {v9_metrics["tp"]:>15,} {v9_metrics["tp"]-v8_metrics["tp"]:>+15,}')
print(f'{"False Positives (FP)":.<25} {v8_metrics["fp"]:>15,} {v9_metrics["fp"]:>15,} {v9_metrics["fp"]-v8_metrics["fp"]:>+15,}')
print(f'{"True Negatives (TN)":.<25} {v8_metrics["tn"]:>15,} {v9_metrics["tn"]:>15,} {v9_metrics["tn"]-v8_metrics["tn"]:>+15,}')
print(f'{"False Negatives (FN)":.<25} {v8_metrics["fn"]:>15,} {v9_metrics["fn"]:>15,} {v9_metrics["fn"]-v8_metrics["fn"]:>+15,}')

print('\n' + '-'*80)
print('Summary:')
print('-'*80)

print(f'{"Predicted KEEP":.<25} {v8_metrics["total_pred_positive"]:>15,} {v9_metrics["total_pred_positive"]:>15,} {v9_metrics["total_pred_positive"]-v8_metrics["total_pred_positive"]:>+15,}')
print(f'{"Predicted DELETE":.<25} {v8_metrics["total_pred_negative"]:>15,} {v9_metrics["total_pred_negative"]:>15,} {v9_metrics["total_pred_negative"]-v8_metrics["total_pred_negative"]:>+15,}')

# Analysis
print('\n' + '='*80)
print('INTERPRETATION')
print('='*80)

fbeta_diff = v9_metrics["fbeta"] - v8_metrics["fbeta"]
prec_diff = v9_metrics["precision"] - v8_metrics["precision"]
rec_diff = v9_metrics["recall"] - v8_metrics["recall"]
fp_diff = v9_metrics["fp"] - v8_metrics["fp"]
fn_diff = v9_metrics["fn"] - v8_metrics["fn"]

if fbeta_diff > 0:
    print(f'\n[SUCCESS] v9 achieves {fbeta_diff:+.4f} higher F-beta ({100*fbeta_diff:.2f}% improvement)')
else:
    print(f'\n[DECLINE] v9 has {fbeta_diff:.4f} lower F-beta ({100*fbeta_diff:.2f}% decline)')

if prec_diff > 0:
    print(f'[SUCCESS] v9 has {100*prec_diff:.2f}% better precision ({fp_diff:+,} change in FP)')
else:
    print(f'[CAUTION] v9 has {100*prec_diff:.2f}% worse precision ({fp_diff:+,} change in FP)')

if rec_diff > 0:
    print(f'[SUCCESS] v9 has {100*rec_diff:.2f}% better recall ({fn_diff:+,} change in FN)')
else:
    print(f'[CAUTION] v9 has {100*rec_diff:.2f}% worse recall ({fn_diff:+,} change in FN)')

print('\n' + '='*80)
