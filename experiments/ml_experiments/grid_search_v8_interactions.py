"""
Focused grid search for v8 with higher interaction counts.
Test if interactions between new features (hurst_exponent, baseline_drift)
and existing features improve performance.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
import time
from pathlib import Path
from itertools import product
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, fbeta_score
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import FBETA_BETA

warnings.filterwarnings('ignore')


def train_and_evaluate_ebm(params, X_train, y_train, X_test, y_test, thresholds):
    """Train a single EBM model and evaluate at multiple thresholds."""
    from interpret.glassbox import ExplainableBoostingClassifier

    try:
        ebm = ExplainableBoostingClassifier(
            feature_names=list(X_train.columns),
            max_bins=params['max_bins'],
            max_interaction_bins=min(64, params['max_bins'] // 4),
            interactions=params['interactions'],
            outer_bags=params.get('outer_bags', 8),
            inner_bags=0,
            learning_rate=params.get('learning_rate', 0.01),
            greedy_ratio=params.get('greedy_ratio', 0.0),
            smoothing_rounds=params.get('smoothing_rounds', 0),
            interaction_smoothing_rounds=params.get('smoothing_rounds', 0),
            validation_size=0.15,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            early_stopping_tolerance=1e-4,
            max_rounds=params.get('max_rounds', 5000),
            min_samples_leaf=params['min_samples_leaf'],
            max_leaves=params['max_leaves'],
            random_state=params['random_state']
        )

        t_start = time.time()
        ebm.fit(X_train, y_train)
        train_time = time.time() - t_start

        y_train_proba = ebm.predict_proba(X_train)[:, 1]
        y_test_proba = ebm.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_proba)
        test_auc = roc_auc_score(y_test, y_test_proba)

        results = []
        for thresh in thresholds:
            y_train_pred = (y_train_proba >= thresh).astype(int)
            y_test_pred = (y_test_proba >= thresh).astype(int)

            train_prec, train_rec, _, _ = precision_recall_fscore_support(
                y_train, y_train_pred, average='binary', zero_division=0
            )
            test_prec, test_rec, _, _ = precision_recall_fscore_support(
                y_test, y_test_pred, average='binary', zero_division=0
            )
            train_fbeta = fbeta_score(y_train, y_train_pred, beta=FBETA_BETA,
                                      average='binary', zero_division=0)
            test_fbeta = fbeta_score(y_test, y_test_pred, beta=FBETA_BETA,
                                     average='binary', zero_division=0)

            result = {
                **params,
                'threshold': thresh,
                'train_precision': train_prec,
                'train_recall': train_rec,
                'train_fbeta': train_fbeta,
                'train_auc': train_auc,
                'test_precision': test_prec,
                'test_recall': test_rec,
                'test_fbeta': test_fbeta,
                'test_auc': test_auc,
                'train_time_sec': train_time,
            }
            results.append(result)

        return results, ebm

    except Exception as e:
        print(f"  ERROR with params {params}: {e}")
        return None, None


print("="*80)
print("V8 INTERACTION GRID SEARCH (interactions: 20-50)")
print("="*80)

# Load v8 dataset
dataset_path = 'ml/results/training_dataset_v8.csv'
output_dir = Path('ml/ebm_grid_search_v8_interactions')
output_dir.mkdir(parents=True, exist_ok=True)

print(f'\nLoading dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Total samples: {len(df)}')
print(f'Sessions: {df["session"].nunique()}')

# Prepare features
label_col = 'ground_truth'
exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                'distance_to_gt', 'is_corner_artifact', 'corr_groups'}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

print(f'Features: {len(feature_cols)}')

# Create stratified train/test split
sessions = df['session'].unique()
session_experiments = {s: s.split('_')[0] for s in sessions}
experiments = [session_experiments[s] for s in sessions]

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(sessions, experiments))
train_sessions = set(sessions[train_idx])
test_sessions = set(sessions[test_idx])

train_mask = df['session'].isin(train_sessions)
test_mask = df['session'].isin(test_sessions)

X_train = df.loc[train_mask, feature_cols].copy()
y_train = df.loc[train_mask, label_col].values
X_test = df.loc[test_mask, feature_cols].copy()
y_test = df.loc[test_mask, label_col].values

print(f'\nTrain sessions: {len(train_sessions)}')
print(f'Test sessions: {len(test_sessions)}')
print(f'Train samples: {len(X_train)} (KEEP: {y_train.sum()}, {100*y_train.mean():.1f}%)')
print(f'Test samples: {len(X_test)} (KEEP: {y_test.sum()}, {100*y_test.mean():.1f}%)')

# Focused parameter grid - explore interactions
# Keep best parameters from previous search, vary interactions
param_grid = {
    'max_bins': [1024],              # Best from previous
    'interactions': [20, 30, 40, 50], # MAIN VARIABLE
    'greedy_ratio': [10.0],          # Best from previous
    'smoothing_rounds': [0],         # Best from previous
    'min_samples_leaf': [2],         # Best from previous
    'max_leaves': [3, 4],            # Top 2 from previous
    'outer_bags': [8],
    'learning_rate': [0.01],
    'random_state': [42],
}

param_names = list(param_grid.keys())
param_values = list(param_grid.values())
all_params = [dict(zip(param_names, v)) for v in product(*param_values)]

print(f'\nTotal model configurations: {len(all_params)}')

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8]
print(f'Thresholds to evaluate: {thresholds}')
print(f'Total evaluations: {len(all_params) * len(thresholds)}')

print(f'\nRunning grid search with 4 parallel jobs...')
estimated_time = len(all_params) * 40 / 4 / 60  # ~40s per model with more interactions
print(f'Estimated time: {estimated_time:.0f}-{estimated_time*1.5:.0f} minutes\n')


def train_single(params, idx):
    print(f"  [{idx+1}/{len(all_params)}] bins={params['max_bins']}, "
          f"inter={params['interactions']}, greedy={params['greedy_ratio']}, "
          f"leaves={params['max_leaves']}")
    results, model = train_and_evaluate_ebm(
        params, X_train, y_train, X_test, y_test, thresholds
    )
    return results, model, params


parallel_results = Parallel(n_jobs=4, verbose=0)(
    delayed(train_single)(params, idx) for idx, params in enumerate(all_params)
)

# Collect results
all_results = []
config_models = {}

for results, model, params in parallel_results:
    if results is not None:
        all_results.extend(results)

        # Track models by config (for saving best)
        config_key = (
            params['max_bins'], params['interactions'],
            params['greedy_ratio'], params['smoothing_rounds'],
            params['min_samples_leaf'], params['max_leaves']
        )
        if config_key not in config_models:
            config_models[config_key] = {'models': [], 'fbeta_scores': [], 'params': params}

        # Get F-beta at threshold 0.75 (user's preferred)
        for r in results:
            if r['threshold'] == 0.75:
                config_models[config_key]['models'].append(model)
                config_models[config_key]['fbeta_scores'].append(r['test_fbeta'])
                break

results_df = pd.DataFrame(all_results)
results_path = output_dir / 'results.csv'
results_df.to_csv(results_path, index=False)
print(f'\nFull results saved to: {results_path}')

# Analysis
print('\n' + '='*80)
print('RESULTS ANALYSIS')
print('='*80)

# Compare by interaction count at threshold 0.75
df_075 = results_df[results_df['threshold'] == 0.75].copy()

print(f'\n--- PERFORMANCE BY INTERACTION COUNT (threshold=0.75) ---')
interaction_summary = df_075.groupby('interactions').agg({
    'test_fbeta': ['mean', 'max'],
    'test_precision': ['mean'],
    'test_recall': ['mean'],
    'train_time_sec': ['mean']
}).round(4)

print(interaction_summary.to_string())

# Best model
best_row = df_075.loc[df_075['test_fbeta'].idxmax()]
print(f'\n--- BEST MODEL (threshold=0.75) ---')
print(f'  Interactions: {best_row["interactions"]}')
print(f'  Max leaves: {best_row["max_leaves"]}')
print(f'  Test F-beta: {best_row["test_fbeta"]:.4f}')
print(f'  Test Precision: {best_row["test_precision"]:.4f}')
print(f'  Test Recall: {best_row["test_recall"]:.4f}')
print(f'  Train time: {best_row["train_time_sec"]:.1f}s')

# Compare to baseline (interactions=20)
baseline = df_075[df_075['interactions'] == 20]['test_fbeta'].max()
best_fbeta = df_075['test_fbeta'].max()
improvement = (best_fbeta - baseline) / baseline * 100

print(f'\n--- COMPARISON TO BASELINE ---')
print(f'  Baseline (interactions=20): F-beta = {baseline:.4f}')
print(f'  Best (interactions={best_row["interactions"]}): F-beta = {best_fbeta:.4f}')
print(f'  Improvement: {improvement:+.2f}%')

if improvement > 0.5:
    print(f'\n  [SUCCESS] Higher interactions provide meaningful improvement!')
elif improvement > 0:
    print(f'\n  [MARGINAL] Slight improvement with higher interactions')
else:
    print(f'\n  [NO BENEFIT] Higher interactions do not improve performance')

# Save best model
best_config_key = max(config_models.keys(),
                      key=lambda k: np.mean(config_models[k]['fbeta_scores']))
best_config_data = config_models[best_config_key]
best_mean_fbeta = np.mean(best_config_data['fbeta_scores'])

fbeta_scores = best_config_data['fbeta_scores']
median_idx = np.argsort(fbeta_scores)[len(fbeta_scores) // 2]
best_model = best_config_data['models'][median_idx]
best_params = best_config_data['params']

best_model_path = output_dir / 'ebm_best.pkl'
with open(best_model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f'\nBest model saved: {best_model_path}')
print(f'  Config: bins={best_params["max_bins"]}, inter={best_params["interactions"]}, '
      f'greedy={best_params["greedy_ratio"]}, leaves={best_params["max_leaves"]}')
print(f'  Test F-beta (threshold=0.75): {best_mean_fbeta:.4f}')

# Full threshold comparison for best interaction count
best_interactions = best_row['interactions']
df_best_inter = results_df[results_df['interactions'] == best_interactions]

print(f'\n--- PRECISION-RECALL TRADE-OFF (interactions={best_interactions}) ---')
tradeoff = df_best_inter.groupby('threshold').agg({
    'test_precision': 'mean',
    'test_recall': 'mean',
    'test_fbeta': 'mean'
}).reset_index()
print(tradeoff.to_string(index=False))

print('\n' + '='*80)
print('GRID SEARCH COMPLETE')
print('='*80)
