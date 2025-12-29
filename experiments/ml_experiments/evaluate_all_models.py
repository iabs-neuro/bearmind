"""Comprehensive evaluation of all models on both datasets."""
import sys
sys.path.insert(0, 'ml')
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_utils import FEATURE_COLS, compute_fbeta, FBETA_BETA

print('='*80)
print('COMPREHENSIVE MODEL EVALUATION')
print('='*80)
print(f'F-beta weight: {FBETA_BETA:.4f} (precision {1/FBETA_BETA**2:.1f}x more important than recall)')
print()

# Load both datasets
df_full = pd.read_csv('ml/results/training_dataset_v6.csv')
df_no3dm = pd.read_csv('ml/results/training_dataset_v6_no3dm.csv')

print(f'Full dataset: {len(df_full)} samples (3DM + NOF + RFC + FOF)')
print(f'No-3DM dataset: {len(df_no3dm)} samples (NOF + RFC + FOF only)')
print()

# Models to evaluate
models_info = {
    'v5_full': {'path': 'ml/ebm_grid_search_v5/ebm_best.pkl', 'trained_on': 'v5 full'},
    'v5_no3dm': {'path': 'ml/ebm_grid_search_v5_no3dm/ebm_best.pkl', 'trained_on': 'v5 no-3DM'},
    'v6_full': {'path': 'ml/ebm_grid_search_v6/ebm_best.pkl', 'trained_on': 'v6 full'},
    'v6_no3dm': {'path': 'ml/ebm_grid_search_v6_no3dm/ebm_best.pkl', 'trained_on': 'v6 no-3DM'},
}

MODEL_COLORS = {'v5_full': '#1f77b4', 'v5_no3dm': '#ff7f0e',
                'v6_full': '#2ca02c', 'v6_no3dm': '#d62728'}

def evaluate_on_dataset(df, dataset_name):
    label_col = 'ground_truth' if 'ground_truth' in df.columns else 'label'
    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    y = df[label_col].values

    print(f'\n{"="*80}')
    print(f'EVALUATION ON: {dataset_name} ({len(X)} samples)')
    print(f'{"="*80}')

    results = []

    for name, info in models_info.items():
        with open(info['path'], 'rb') as f:
            model = pickle.load(f)

        y_proba = model.predict_proba(X)[:, 1]

        # Scan all thresholds
        thresholds = np.linspace(0.3, 0.9, 61)
        precisions = []
        recalls = []
        fbetas = []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            tp = ((y_pred == 1) & (y == 1)).sum()
            fp = ((y_pred == 1) & (y == 0)).sum()
            fn = ((y_pred == 0) & (y == 1)).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            fbeta = compute_fbeta(prec, rec)

            precisions.append(prec)
            recalls.append(rec)
            fbetas.append(fbeta)

        precisions = np.array(precisions)
        recalls = np.array(recalls)
        fbetas = np.array(fbetas)

        # Find best
        best_idx = np.argmax(fbetas)

        # Get t=0.5 metrics
        idx_05 = np.argmin(np.abs(thresholds - 0.5))

        results.append({
            'name': name,
            'trained_on': info['trained_on'],
            'thresholds': thresholds,
            'precisions': precisions,
            'recalls': recalls,
            'fbetas': fbetas,
            'fbeta_05': fbetas[idx_05],
            'prec_05': precisions[idx_05],
            'rec_05': recalls[idx_05],
            'best_fbeta': fbetas[best_idx],
            'best_thresh': thresholds[best_idx],
            'best_prec': precisions[best_idx],
            'best_rec': recalls[best_idx],
        })

    # Print results sorted by best F-beta
    results_sorted = sorted(results, key=lambda x: x['best_fbeta'], reverse=True)

    print(f'\n--- Results at threshold=0.5 ---')
    print(f'{"Model":<12} {"Trained On":<15} {"F-beta":>8} {"Precision":>10} {"Recall":>8}')
    print('-'*58)
    for r in results_sorted:
        print(f'{r["name"]:<12} {r["trained_on"]:<15} {r["fbeta_05"]:>8.4f} {r["prec_05"]:>10.4f} {r["rec_05"]:>8.4f}')

    print(f'\n--- Results at BEST threshold per model ---')
    print(f'{"Model":<12} {"Best F-beta":>10} {"@Thresh":>8} {"Precision":>10} {"Recall":>8}')
    print('-'*52)
    for r in results_sorted:
        print(f'{r["name"]:<12} {r["best_fbeta"]:>10.4f} {r["best_thresh"]:>8.2f} {r["best_prec"]:>10.4f} {r["best_rec"]:>8.4f}')

    return results_sorted

# Evaluate on both datasets
results_full = evaluate_on_dataset(df_full, 'FULL V6 DATASET (includes 3DM)')
results_no3dm = evaluate_on_dataset(df_no3dm, 'NO-3DM DATASET (NOF+RFC+FOF only)')

# Create comprehensive plot
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Row 1: Full dataset
ax1, ax2, ax3 = axes[0]

# PR curve - Full dataset
for r in results_full:
    ax1.plot(r['recalls'], r['precisions'], color=MODEL_COLORS[r['name']],
             linewidth=2, label=f"{r['name']} (best={r['best_fbeta']:.3f})")
    best_idx = np.argmax(r['fbetas'])
    ax1.scatter([r['recalls'][best_idx]], [r['precisions'][best_idx]],
               color=MODEL_COLORS[r['name']], s=100, marker='*', edgecolors='black', zorder=5)
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.set_title('PR Curve - FULL Dataset (60K samples)')
ax1.legend(loc='lower left', fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0.7, 1.0])
ax1.set_ylim([0.75, 1.0])

# F-beta vs threshold - Full dataset
for r in results_full:
    ax2.plot(r['thresholds'], r['fbetas'], color=MODEL_COLORS[r['name']],
             linewidth=2, label=r['name'])
ax2.set_xlabel('Threshold')
ax2.set_ylabel('F-beta')
ax2.set_title('F-beta vs Threshold - FULL Dataset')
ax2.legend(loc='lower left', fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0.3, 0.9])

# Precision/Recall vs threshold - Full dataset (best model only)
best_full = max(results_full, key=lambda x: x['best_fbeta'])
ax3.plot(best_full['thresholds'], best_full['precisions'], 'b-', linewidth=2, label='Precision')
ax3.plot(best_full['thresholds'], best_full['recalls'], 'g-', linewidth=2, label='Recall')
ax3.plot(best_full['thresholds'], best_full['fbetas'], 'r--', linewidth=2, label='F-beta')
ax3.axvline(x=best_full['best_thresh'], color='gray', linestyle=':', alpha=0.7)
ax3.set_xlabel('Threshold')
ax3.set_ylabel('Score')
ax3.set_title(f'Best Model ({best_full["name"]}) - FULL Dataset')
ax3.legend(loc='center left', fontsize=8)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([0.3, 0.9])

# Row 2: No-3DM dataset
ax4, ax5, ax6 = axes[1]

# PR curve - No-3DM dataset
for r in results_no3dm:
    ax4.plot(r['recalls'], r['precisions'], color=MODEL_COLORS[r['name']],
             linewidth=2, label=f"{r['name']} (best={r['best_fbeta']:.3f})")
    best_idx = np.argmax(r['fbetas'])
    ax4.scatter([r['recalls'][best_idx]], [r['precisions'][best_idx]],
               color=MODEL_COLORS[r['name']], s=100, marker='*', edgecolors='black', zorder=5)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title('PR Curve - NO-3DM Dataset (42K samples)')
ax4.legend(loc='lower left', fontsize=8)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([0.7, 1.0])
ax4.set_ylim([0.75, 1.0])

# F-beta vs threshold - No-3DM dataset
for r in results_no3dm:
    ax5.plot(r['thresholds'], r['fbetas'], color=MODEL_COLORS[r['name']],
             linewidth=2, label=r['name'])
ax5.set_xlabel('Threshold')
ax5.set_ylabel('F-beta')
ax5.set_title('F-beta vs Threshold - NO-3DM Dataset')
ax5.legend(loc='lower left', fontsize=8)
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0.3, 0.9])

# Precision/Recall vs threshold - No-3DM dataset (best model only)
best_no3dm = max(results_no3dm, key=lambda x: x['best_fbeta'])
ax6.plot(best_no3dm['thresholds'], best_no3dm['precisions'], 'b-', linewidth=2, label='Precision')
ax6.plot(best_no3dm['thresholds'], best_no3dm['recalls'], 'g-', linewidth=2, label='Recall')
ax6.plot(best_no3dm['thresholds'], best_no3dm['fbetas'], 'r--', linewidth=2, label='F-beta')
ax6.axvline(x=best_no3dm['best_thresh'], color='gray', linestyle=':', alpha=0.7)
ax6.set_xlabel('Threshold')
ax6.set_ylabel('Score')
ax6.set_title(f'Best Model ({best_no3dm["name"]}) - NO-3DM Dataset')
ax6.legend(loc='center left', fontsize=8)
ax6.grid(True, alpha=0.3)
ax6.set_xlim([0.3, 0.9])

plt.tight_layout()
plt.savefig('ml/results/comprehensive_v5_v6_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved plot to: ml/results/comprehensive_v5_v6_comparison.png')

# Summary
print()
print('='*80)
print('SUMMARY & RECOMMENDATIONS')
print('='*80)
print()
print('For GENERAL USE (all experiments including 3DM):')
print(f'  Best model: {best_full["name"]} (F-beta={best_full["best_fbeta"]:.4f} @ t={best_full["best_thresh"]:.2f})')
print(f'  At t=0.5: F-beta={best_full["fbeta_05"]:.4f}, Prec={best_full["prec_05"]:.4f}, Rec={best_full["rec_05"]:.4f}')

print()
print('For NOF/RFC/FOF experiments (no 3DM):')
print(f'  Best model: {best_no3dm["name"]} (F-beta={best_no3dm["best_fbeta"]:.4f} @ t={best_no3dm["best_thresh"]:.2f})')
print(f'  At t=0.5: F-beta={best_no3dm["fbeta_05"]:.4f}, Prec={best_no3dm["prec_05"]:.4f}, Rec={best_no3dm["rec_05"]:.4f}')
