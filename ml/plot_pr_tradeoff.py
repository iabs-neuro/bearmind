"""Plot precision-recall tradeoff curve for one or more models by scanning thresholds.

Supports multi-model comparison for benchmarking different configurations.
Uses CSV datasets from ml/results/ instead of artifacts directories.
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from data_utils import FEATURE_COLS, compute_fbeta, FBETA_BETA

# Color palette for multiple models
MODEL_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]


def load_dataset_from_csv(dataset_path, experiments=None):
    """
    Load dataset from CSV file.

    Parameters
    ----------
    dataset_path : str or Path
        Path to CSV dataset (e.g., ml/results/training_dataset_v5.csv)
    experiments : list of str, optional
        Filter to specific experiments (e.g., ['NOF', 'RFC', 'FOF'])

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : np.ndarray
        Labels (1=KEEP, 0=DELETE)
    """
    df = pd.read_csv(dataset_path)

    # Filter by experiments if specified
    if experiments:
        df = df[df['session'].str.split('_').str[0].isin(experiments)].copy()

    # Extract all numeric columns as potential features
    # Each model will select its own features via model.feature_names_in_
    label_col = 'label' if 'label' in df.columns else 'ground_truth'
    exclude_cols = {label_col, 'session', 'experiment', 'component_idx', 'center',
                    'distance_to_gt', 'is_corner_artifact', 'corr_groups'}
    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Also add wavelet_snr as alias for event_snr (for v4 compatibility)
    if 'event_snr' in X.columns and 'wavelet_snr' not in X.columns:
        X['wavelet_snr'] = X['event_snr']

    y = df[label_col].values

    return X, y


def plot_pr_tradeoff(model_path, dataset_path, experiments=None, output_path=None, n_points=100):
    """Plot precision-recall tradeoff curve."""
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Loaded model: {model_path}")

    # Load data
    exp_list = experiments.split(',') if experiments else None
    X, y = load_dataset_from_csv(dataset_path, exp_list)
    print(f"Loaded {len(X)} samples ({y.sum()} KEEP, {len(y) - y.sum()} DELETE)")

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

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    fbeta_scores = np.array(fbeta_scores)

    # Find best F-beta threshold
    best_idx = np.argmax(fbeta_scores)
    best_thresh = thresholds[best_idx]
    best_fbeta = fbeta_scores[best_idx]
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Precision-Recall curve
    ax1 = axes[0]
    ax1.plot(recalls, precisions, 'b-', linewidth=2, label='P-R Curve')
    ax1.scatter([best_rec], [best_prec], c='red', s=100, zorder=5,
                label=f'Best F-beta={best_fbeta:.3f} @ t={best_thresh:.2f}')

    # Add threshold annotations
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        idx = np.argmin(np.abs(thresholds - t))
        ax1.annotate(f't={t}', (recalls[idx], precisions[idx]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Trade-off', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.7, 1.01])
    ax1.set_ylim([0.8, 1.01])

    # Plot 2: Metrics vs Threshold
    ax2 = axes[1]
    ax2.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision')
    ax2.plot(thresholds, recalls, 'g-', linewidth=2, label='Recall')
    ax2.plot(thresholds, fbeta_scores, 'r--', linewidth=2, label=f'F-beta (b={FBETA_BETA:.3f})')
    ax2.axvline(x=best_thresh, color='gray', linestyle=':', alpha=0.7,
                label=f'Best threshold={best_thresh:.2f}')

    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Metrics vs Classification Threshold', fontsize=14)
    ax2.legend(loc='center left')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.3, 0.9])
    ax2.set_ylim([0.7, 1.01])

    plt.tight_layout()

    # Save or show
    if output_path is None:
        model_name = Path(model_path).stem
        output_path = Path(model_path).parent / f"{model_name}_pr_tradeoff.png"

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close to free memory
    print(f"Saved plot to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"PRECISION-RECALL TRADE-OFF SUMMARY (F-beta b={FBETA_BETA:.3f})")
    print("=" * 60)
    print(f"\nBest F-beta: {best_fbeta:.4f} at threshold {best_thresh:.2f}")
    print(f"  Precision: {best_prec:.4f}")
    print(f"  Recall: {best_rec:.4f}")

    print("\nKey threshold points:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F-beta':>10}")
    print("-" * 45)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(thresholds - t))
        print(f"{t:>10.1f} {precisions[idx]:>10.4f} {recalls[idx]:>10.4f} {fbeta_scores[idx]:>10.4f}")

    return thresholds, precisions, recalls, fbeta_scores


def evaluate_model(model, X, y, n_points=100):
    """Evaluate a model and return metrics at different thresholds."""
    # Use only features the model was trained on
    model_features = list(model.feature_names_in_)
    X_model = X[model_features].copy()
    y_proba = model.predict_proba(X_model)[:, 1]

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


def plot_multi_model_comparison(
    model_paths,
    model_names=None,
    dataset_path="ml/results/training_dataset_v5.csv",
    experiments=None,
    output_path=None,
    n_points=100
):
    """
    Plot precision-recall curves for multiple models on same dataset.

    Parameters
    ----------
    model_paths : list of str
        Paths to model pickle files
    model_names : list of str, optional
        Names for each model in legend. If None, uses file stems.
    dataset_path : str
        Path to CSV dataset file
    experiments : str, optional
        Comma-separated experiment filter (e.g., "NOF,RFC,FOF")
    output_path : str, optional
        Output path for the plot
    n_points : int
        Number of threshold points to scan
    """
    # Load models
    models = []
    if model_names is None:
        model_names = []
    for i, path in enumerate(model_paths):
        with open(path, 'rb') as f:
            models.append(pickle.load(f))
        if len(model_names) <= i:
            model_names.append(Path(path).stem)

    print(f"Loaded {len(models)} models:")
    for name in model_names:
        print(f"  - {name}")

    # Load data
    exp_list = experiments.split(',') if experiments else None
    X, y = load_dataset_from_csv(dataset_path, exp_list)
    print(f"Loaded {len(X)} samples ({y.sum()} KEEP, {len(y) - y.sum()} DELETE)")

    # Evaluate all models
    results = []
    for model, name in zip(models, model_names):
        print(f"Evaluating {name}...")
        res = evaluate_model(model, X, y, n_points)
        res['name'] = name
        results.append(res)

    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Precision-Recall curves (top-left)
    ax1 = axes[0, 0]
    for i, res in enumerate(results):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        best_idx = np.argmax(res['fbeta_scores'])
        best_fbeta = res['fbeta_scores'][best_idx]

        ax1.plot(res['recalls'], res['precisions'], color=color, linewidth=2,
                 label=f"{res['name']} (F-beta={best_fbeta:.3f})")
        ax1.scatter([res['recalls'][best_idx]], [res['precisions'][best_idx]],
                   c=color, s=80, zorder=5, marker='*', edgecolors='black')

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title('Precision-Recall Trade-off Comparison', fontsize=14)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.7, 1.01])
    ax1.set_ylim([0.8, 1.01])

    # Plot 2: F-beta vs Threshold (top-right)
    ax2 = axes[0, 1]
    for i, res in enumerate(results):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax2.plot(res['thresholds'], res['fbeta_scores'], color=color, linewidth=2,
                 label=res['name'])

    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel(f'F-beta (b={FBETA_BETA:.3f})', fontsize=12)
    ax2.set_title(f'F-beta Score vs Threshold', fontsize=14)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.3, 0.9])
    ax2.set_ylim([0.85, 0.95])

    # Plot 3: Precision vs Threshold (bottom-left)
    ax3 = axes[1, 0]
    for i, res in enumerate(results):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax3.plot(res['thresholds'], res['precisions'], color=color, linewidth=2,
                 label=res['name'])

    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Precision', fontsize=12)
    ax3.set_title('Precision vs Threshold', fontsize=14)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0.3, 0.9])
    ax3.set_ylim([0.8, 1.01])

    # Plot 4: Recall vs Threshold (bottom-right)
    ax4 = axes[1, 1]
    for i, res in enumerate(results):
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax4.plot(res['thresholds'], res['recalls'], color=color, linewidth=2,
                 label=res['name'])

    ax4.set_xlabel('Threshold', fontsize=12)
    ax4.set_ylabel('Recall', fontsize=12)
    ax4.set_title('Recall vs Threshold', fontsize=14)
    ax4.legend(loc='lower left', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.3, 0.9])
    ax4.set_ylim([0.7, 1.01])

    plt.tight_layout()

    # Save
    if output_path is None:
        output_path = Path(dataset_path).parent / "model_comparison.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)  # Close to free memory and avoid display issues
    print(f"\nSaved comparison plot to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"MODEL COMPARISON SUMMARY (threshold=0.5, F-beta b={FBETA_BETA:.3f})")
    print("=" * 80)
    print(f"\n{'Model':<35} {'F-beta':>8} {'Prec':>8} {'Rec':>8} {'Best':>8} {'@Thresh':>8}")
    print("-" * 80)

    for res in results:
        # Find threshold=0.5 metrics
        idx_05 = np.argmin(np.abs(res['thresholds'] - 0.5))
        best_idx = np.argmax(res['fbeta_scores'])

        print(f"{res['name']:<35} "
              f"{res['fbeta_scores'][idx_05]:>8.4f} "
              f"{res['precisions'][idx_05]:>8.4f} "
              f"{res['recalls'][idx_05]:>8.4f} "
              f"{res['fbeta_scores'][best_idx]:>8.4f} "
              f"{res['thresholds'][best_idx]:>8.2f}")

    # Print detailed per-threshold table for all models
    print("\n" + "=" * 80)
    print("DETAILED THRESHOLD COMPARISON")
    print("=" * 80)

    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        print(f"\n--- Threshold = {thresh} ---")
        print(f"{'Model':<35} {'F-beta':>8} {'Prec':>8} {'Rec':>8}")
        print("-" * 60)
        for res in results:
            idx = np.argmin(np.abs(res['thresholds'] - thresh))
            print(f"{res['name']:<35} "
                  f"{res['fbeta_scores'][idx]:>8.4f} "
                  f"{res['precisions'][idx]:>8.4f} "
                  f"{res['recalls'][idx]:>8.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot precision-recall tradeoff curve for one or more models"
    )
    parser.add_argument("--model", type=str, default=None,
                       help="Path to single model pickle file")
    parser.add_argument("--models", type=str, nargs='+', default=None,
                       help="Paths to multiple model pickle files for comparison")
    parser.add_argument("--names", type=str, nargs='+', default=None,
                       help="Names for each model in comparison (optional)")
    parser.add_argument("--dataset", default="ml/results/training_dataset_v5.csv",
                       help="Path to CSV dataset file (default: ml/results/training_dataset_v5.csv)")
    parser.add_argument("--experiments", type=str, default=None,
                       help="Filter to specific experiments (comma-separated, e.g., NOF,RFC,FOF)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for the plot")
    parser.add_argument("--n-points", type=int, default=100,
                       help="Number of threshold points to scan (default: 100)")

    args = parser.parse_args()

    if args.models:
        # Multi-model comparison mode
        plot_multi_model_comparison(
            model_paths=args.models,
            model_names=args.names,
            dataset_path=args.dataset,
            experiments=args.experiments,
            output_path=args.output,
            n_points=args.n_points
        )
    elif args.model:
        # Single model mode
        plot_pr_tradeoff(
            model_path=args.model,
            dataset_path=args.dataset,
            experiments=args.experiments,
            output_path=args.output,
            n_points=args.n_points
        )
    else:
        parser.error("Either --model or --models must be specified")
