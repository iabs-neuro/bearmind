"""
Fair comparison: evaluate each model on its own dataset.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_fbeta(precision, recall, beta=0.5773502691896257):
    """Compute F-beta score (beta=0.577 favors precision)."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def evaluate_model(model, dataset_path, n_points=100):
    """Evaluate a model on its dataset."""
    # Load dataset
    df = pd.read_csv(dataset_path)

    # Prepare features
    exclude_cols = {'ground_truth', 'session', 'experiment', 'component_idx', 'center',
                    'distance_to_gt', 'is_corner_artifact', 'corr_groups'}

    # Use only features the model was trained on
    model_features = list(model.feature_names_in_)
    feature_cols = [c for c in df.columns if c not in exclude_cols and c in model_features]

    X = df[feature_cols].copy()
    y = df['ground_truth'].values

    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]

    # Scan thresholds
    thresholds = np.linspace(0.3, 0.9, n_points)
    precisions = []
    recalls = []
    fbeta_scores = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        fbeta = compute_fbeta(precision, recall)

        precisions.append(precision)
        recalls.append(recall)
        fbeta_scores.append(fbeta)

    return {
        'thresholds': thresholds,
        'precisions': np.array(precisions),
        'recalls': np.array(recalls),
        'fbeta_scores': np.array(fbeta_scores),
    }

print('='*80)
print('FAIR COMPARISON: Each model evaluated on its own dataset')
print('='*80)

# Load models and datasets
models = [
    ('v6_no3dm (Wavelet n=2)', 'ml/ebm_grid_search_v6_no3dm/ebm_best.pkl', 'ml/results/training_dataset_v6_no3dm.csv'),
    ('v7 (Threshold n=2)', 'ml/ebm_grid_search_v7/ebm_best.pkl', 'ml/results/training_dataset_v7.csv'),
    ('v8 (Hybrid + Wavelet n=3)', 'ml/ebm_grid_search_v8/ebm_best.pkl', 'ml/results/training_dataset_v8.csv'),
]

results = []
for name, model_path, dataset_path in models:
    print(f'\nEvaluating {name}...')
    print(f'  Model: {model_path}')
    print(f'  Dataset: {dataset_path}')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    res = evaluate_model(model, dataset_path)
    res['name'] = name

    # Find best F-beta
    best_idx = np.argmax(res['fbeta_scores'])
    best_fbeta = res['fbeta_scores'][best_idx]
    best_thresh = res['thresholds'][best_idx]
    best_prec = res['precisions'][best_idx]
    best_rec = res['recalls'][best_idx]

    print(f'  Best F-beta: {best_fbeta:.4f} at threshold {best_thresh:.2f}')
    print(f'    Precision: {best_prec:.4f}')
    print(f'    Recall: {best_rec:.4f}')

    results.append(res)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green

# Plot 1: Precision-Recall curves
ax1 = axes[0, 0]
for i, res in enumerate(results):
    best_idx = np.argmax(res['fbeta_scores'])
    best_fbeta = res['fbeta_scores'][best_idx]

    ax1.plot(res['recalls'], res['precisions'], color=colors[i], linewidth=2,
             label=f"{res['name']} (F={best_fbeta:.3f})")
    ax1.scatter([res['recalls'][best_idx]], [res['precisions'][best_idx]],
               c=colors[i], s=100, zorder=5, marker='*', edgecolors='black')

ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Precision-Recall Trade-off (Each on own dataset)', fontsize=14)
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.82, 1.01])
ax1.set_ylim([0.88, 0.96])

# Plot 2: F-beta vs Threshold
ax2 = axes[0, 1]
for i, res in enumerate(results):
    ax2.plot(res['thresholds'], res['fbeta_scores'], color=colors[i], linewidth=2,
             label=res['name'])

ax2.set_xlabel('Threshold', fontsize=12)
ax2.set_ylabel('F-beta (beta=0.577)', fontsize=12)
ax2.set_title('F-beta Score vs Threshold', fontsize=14)
ax2.legend(loc='lower left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.3, 0.9])
ax2.set_ylim([0.88, 0.93])

# Plot 3: Precision vs Threshold
ax3 = axes[1, 0]
for i, res in enumerate(results):
    ax3.plot(res['thresholds'], res['precisions'], color=colors[i], linewidth=2,
             label=res['name'])

ax3.set_xlabel('Threshold', fontsize=12)
ax3.set_ylabel('Precision', fontsize=12)
ax3.set_title('Precision vs Threshold', fontsize=14)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0.3, 0.9])
ax3.set_ylim([0.85, 0.96])

# Plot 4: Recall vs Threshold
ax4 = axes[1, 1]
for i, res in enumerate(results):
    ax4.plot(res['thresholds'], res['recalls'], color=colors[i], linewidth=2,
             label=res['name'])

ax4.set_xlabel('Threshold', fontsize=12)
ax4.set_ylabel('Recall', fontsize=12)
ax4.set_title('Recall vs Threshold', fontsize=14)
ax4.legend(loc='upper right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0.3, 0.9])
ax4.set_ylim([0.80, 1.01])

plt.tight_layout()
plt.savefig('ml/results/pr_comparison_v6_v7_v8_fair.png', dpi=150, bbox_inches='tight')
print(f'\nPlot saved to: ml/results/pr_comparison_v6_v7_v8_fair.png')

# Summary table
print('\n' + '='*80)
print('SUMMARY TABLE')
print('='*80)
print(f"\n{'Model':<35} {'Best F-beta':<12} {'@Threshold':<12} {'Precision':<12} {'Recall'}")
print('-'*80)
for res in results:
    best_idx = np.argmax(res['fbeta_scores'])
    print(f"{res['name']:<35} "
          f"{res['fbeta_scores'][best_idx]:<12.4f} "
          f"{res['thresholds'][best_idx]:<12.2f} "
          f"{res['precisions'][best_idx]:<12.4f} "
          f"{res['recalls'][best_idx]:.4f}")

print('\n' + '='*80)
print('AT THRESHOLD 0.75 (Your preferred)')
print('='*80)
print(f"\n{'Model':<35} {'F-beta':<12} {'Precision':<12} {'Recall'}")
print('-'*65)
for res in results:
    idx_075 = np.argmin(np.abs(res['thresholds'] - 0.75))
    print(f"{res['name']:<35} "
          f"{res['fbeta_scores'][idx_075]:<12.4f} "
          f"{res['precisions'][idx_075]:<12.4f} "
          f"{res['recalls'][idx_075]:.4f}")

print('\n' + '='*80)
print('INTERPRETATION')
print('='*80)
print('\nThis is a FAIR comparison:')
print('  - v6 model evaluated on v6 dataset')
print('  - v7 model evaluated on v7 dataset')
print('  - v8 model evaluated on v8 dataset')
print('\nEach model is assessed on the data distribution it was designed for.')
print('Differences reflect both model quality AND dataset characteristics.')
