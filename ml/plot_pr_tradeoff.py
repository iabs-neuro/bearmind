"""Plot precision-recall tradeoff curve for a given model by scanning thresholds."""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_dataset(artifacts_dir, experiments=None):
    """Load dataset from capcan_artifacts directories."""
    artifacts_path = Path(artifacts_dir)
    session_dirs = sorted([d for d in artifacts_path.iterdir()
                          if d.is_dir() and d.name.startswith('capcan_artifacts_')])

    if experiments:
        filtered_dirs = []
        for d in session_dirs:
            session_name = d.name.replace('capcan_artifacts_', '')
            exp_id = session_name.split('_')[0]
            if exp_id in experiments:
                filtered_dirs.append(d)
        session_dirs = filtered_dirs

    feature_cols = [
        'area', 'circularity', 'max_edge', 'convexity', 'caiman_snr', 'caiman_r_score',
        'events_per_min', 'events_fraction', 't_rise', 't_off', 'wavelet_snr',
        'r2_score', 'event_r2_score', 'nmae', 'nrmse', 'snr_recon', 'noise_level',
        'baseline', 'tau_decay', 'trace_skewness', 'footprint_compactness',
        # New v3 metrics
        'trace_kurtosis', 'aspect_ratio', 'eccentricity', 'edge_distance', 'nn_distance_center'
    ]

    all_features = []
    all_labels = []

    for session_dir in session_dirs:
        try:
            raw_metrics = session_dir / "metrics_init.csv"
            gt_metrics = session_dir / "metrics_gt.csv"

            df_raw = pd.read_csv(raw_metrics)
            df_gt = pd.read_csv(gt_metrics)

            if df_raw['center'].dtype == 'object':
                df_raw['center'] = df_raw['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
            if df_gt['center'].dtype == 'object':
                df_gt['center'] = df_gt['center'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))

            # Filter corner artifacts
            if 'is_corner_artifact' in df_raw.columns:
                df_raw = df_raw[df_raw['is_corner_artifact'] == 0].copy()

            raw_centers = np.array(df_raw['center'].tolist())
            gt_centers = np.array(df_gt['center'].tolist())

            labels = np.zeros(len(df_raw), dtype=int)
            for i, raw_center in enumerate(raw_centers):
                distances = np.linalg.norm(gt_centers - raw_center, axis=1)
                if distances.min() <= 3:
                    labels[i] = 1

            features = df_raw[feature_cols].copy()
            features = features.replace([np.inf, -np.inf], np.nan)

            all_features.append(features)
            all_labels.append(labels)
        except Exception:
            continue

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    return features_df, labels


def plot_pr_tradeoff(model_path, artifacts_dir, experiments=None, output_path=None, n_points=100):
    """Plot precision-recall tradeoff curve."""
    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print(f"Loaded model: {model_path}")

    # Load data
    exp_list = experiments.split(',') if experiments else None
    X, y = load_dataset(artifacts_dir, exp_list)
    print(f"Loaded {len(X)} samples ({y.sum()} KEEP, {len(y) - y.sum()} DELETE)")

    # Get probabilities
    y_proba = model.predict_proba(X)[:, 1]

    # Scan thresholds
    thresholds = np.linspace(0.3, 0.9, n_points)
    precisions = []
    recalls = []
    f1_scores = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        tp = ((y_pred == 1) & (y == 1)).sum()
        fp = ((y_pred == 1) & (y == 0)).sum()
        fn = ((y_pred == 0) & (y == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)

    # Find best F1 threshold
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Precision-Recall curve
    ax1 = axes[0]
    ax1.plot(recalls, precisions, 'b-', linewidth=2, label='P-R Curve')
    ax1.scatter([best_rec], [best_prec], c='red', s=100, zorder=5,
                label=f'Best F1={best_f1:.3f} @ t={best_thresh:.2f}')

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
    ax2.plot(thresholds, f1_scores, 'r--', linewidth=2, label='F1 Score')
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
    print(f"Saved plot to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("PRECISION-RECALL TRADE-OFF SUMMARY")
    print("=" * 60)
    print(f"\nBest F1: {best_f1:.4f} at threshold {best_thresh:.2f}")
    print(f"  Precision: {best_prec:.4f}")
    print(f"  Recall: {best_rec:.4f}")

    print("\nKey threshold points:")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(thresholds - t))
        print(f"{t:>10.1f} {precisions[idx]:>10.4f} {recalls[idx]:>10.4f} {f1_scores[idx]:>10.4f}")

    plt.show()
    return thresholds, precisions, recalls, f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot precision-recall tradeoff curve")
    parser.add_argument("--model", required=True, help="Path to model pickle file")
    parser.add_argument("--artifacts-dir", default="data/capcan_validation_127_v2",
                       help="Directory containing capcan_artifacts_* subdirectories")
    parser.add_argument("--experiments", type=str, default=None,
                       help="Filter to specific experiments (comma-separated, e.g., NOF,RFC)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for the plot (default: model_pr_tradeoff.png)")
    parser.add_argument("--n-points", type=int, default=100,
                       help="Number of threshold points to scan (default: 100)")

    args = parser.parse_args()

    plot_pr_tradeoff(
        model_path=args.model,
        artifacts_dir=args.artifacts_dir,
        experiments=args.experiments,
        output_path=args.output,
        n_points=args.n_points
    )
