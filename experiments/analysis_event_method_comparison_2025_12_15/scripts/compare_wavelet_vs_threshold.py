"""
Compare event-based quality metrics between wavelet (v6) and threshold (v7) methods.

This analysis investigates which event detection method produces better
quality metrics for auto-inspection.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2]))

# Set up plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def load_datasets(project_root):
    """Load v6 and v7 merged datasets."""
    print("Loading datasets...")

    # v6: wavelet method
    v6_path = project_root / "ml" / "results" / "training_dataset_v6.csv"
    df_v6 = pd.read_csv(v6_path)
    print(f"  v6 (wavelet): {len(df_v6)} neurons")

    # v7: threshold method
    v7_path = project_root / "analysis_event_method_comparison_2025_12_15" / "data" / "training_dataset_v7_merged.csv"
    df_v7 = pd.read_csv(v7_path)
    print(f"  v7 (threshold): {len(df_v7)} neurons")

    return df_v6, df_v7


def compare_metrics(df_v6, df_v7, output_dir):
    """Compare key event-based metrics between v6 and v7."""

    # Event-based quality metrics
    event_metrics = [
        'events_per_min',
        'events_fraction',
        'event_snr',
        'event_r2_score',
        'r2_score',
        'nmae',
        'nrmse',
        'snr_recon',
        'kinetics_opt'
    ]

    print("\n" + "="*80)
    print("METRIC COMPARISON: WAVELET (v6) vs THRESHOLD (v7)")
    print("="*80)

    comparison_data = []

    for metric in event_metrics:
        if metric not in df_v6.columns or metric not in df_v7.columns:
            print(f"  [SKIP] {metric}: not in both datasets")
            continue

        # Filter out NaN and inf values
        v6_vals = df_v6[metric].replace([np.inf, -np.inf], np.nan).dropna()
        v7_vals = df_v7[metric].replace([np.inf, -np.inf], np.nan).dropna()

        # Calculate statistics
        v6_mean = v6_vals.mean()
        v6_median = v6_vals.median()
        v6_std = v6_vals.std()

        v7_mean = v7_vals.mean()
        v7_median = v7_vals.median()
        v7_std = v7_vals.std()

        # Percent change (threshold vs wavelet)
        mean_change_pct = ((v7_mean - v6_mean) / abs(v6_mean) * 100) if v6_mean != 0 else 0
        median_change_pct = ((v7_median - v6_median) / abs(v6_median) * 100) if v6_median != 0 else 0

        # Determine which is better (higher is better for most metrics except nmae/nrmse)
        if metric in ['nmae', 'nrmse']:
            better = 'v7' if v7_mean < v6_mean else 'v6'
            improvement_pct = abs(mean_change_pct) if better == 'v7' else -abs(mean_change_pct)
        else:
            better = 'v7' if v7_mean > v6_mean else 'v6'
            improvement_pct = mean_change_pct if better == 'v7' else -mean_change_pct

        comparison_data.append({
            'metric': metric,
            'v6_mean': v6_mean,
            'v6_median': v6_median,
            'v6_std': v6_std,
            'v7_mean': v7_mean,
            'v7_median': v7_median,
            'v7_std': v7_std,
            'mean_change_pct': mean_change_pct,
            'median_change_pct': median_change_pct,
            'better': better,
            'improvement_pct': improvement_pct
        })

        # Print summary
        print(f"\n{metric}:")
        print(f"  v6 (wavelet):   mean={v6_mean:.4f}, median={v6_median:.4f}, std={v6_std:.4f}")
        print(f"  v7 (threshold): mean={v7_mean:.4f}, median={v7_median:.4f}, std={v7_std:.4f}")
        print(f"  Change: {mean_change_pct:+.2f}% (mean), {median_change_pct:+.2f}% (median)")
        print(f"  BETTER: {better.upper()} ({improvement_pct:+.2f}% improvement)")

    # Save comparison table
    df_comparison = pd.DataFrame(comparison_data)
    output_path = output_dir / "metric_comparison_summary.csv"
    df_comparison.to_csv(output_path, index=False)
    print(f"\n[SAVED] Comparison summary: {output_path}")

    return df_comparison


def compare_by_experiment(df_v6, df_v7, output_dir):
    """Compare metrics broken down by experiment type."""

    print("\n" + "="*80)
    print("EXPERIMENT-WISE COMPARISON")
    print("="*80)

    # Focus on key metric: event_r2_score
    metric = 'event_r2_score'

    experiments = sorted(df_v6['experiment'].unique())
    exp_comparison = []

    for exp in experiments:
        v6_exp = df_v6[df_v6['experiment'] == exp][metric].replace([np.inf, -np.inf], np.nan).dropna()
        v7_exp = df_v7[df_v7['experiment'] == exp][metric].replace([np.inf, -np.inf], np.nan).dropna()

        v6_mean = v6_exp.mean()
        v7_mean = v7_exp.mean()
        change_pct = ((v7_mean - v6_mean) / abs(v6_mean) * 100) if v6_mean != 0 else 0

        exp_comparison.append({
            'experiment': exp,
            'v6_mean': v6_mean,
            'v7_mean': v7_mean,
            'change_pct': change_pct,
            'n_neurons': len(v6_exp)
        })

        print(f"{exp}: v6={v6_mean:.4f}, v7={v7_mean:.4f}, change={change_pct:+.2f}% (n={len(v6_exp)})")

    df_exp_comparison = pd.DataFrame(exp_comparison)
    output_path = output_dir / "experiment_comparison.csv"
    df_exp_comparison.to_csv(output_path, index=False)
    print(f"\n[SAVED] Experiment comparison: {output_path}")

    return df_exp_comparison


def compare_by_quality(df_v6, df_v7, output_dir):
    """Compare metrics for KEEP vs DELETE neurons."""

    print("\n" + "="*80)
    print("QUALITY COMPARISON: KEEP vs DELETE")
    print("="*80)

    metric = 'event_r2_score'

    quality_comparison = []

    for label_name, label_val in [('KEEP', 1), ('DELETE', 0)]:
        v6_subset = df_v6[df_v6['ground_truth'] == label_val][metric].replace([np.inf, -np.inf], np.nan).dropna()
        v7_subset = df_v7[df_v7['ground_truth'] == label_val][metric].replace([np.inf, -np.inf], np.nan).dropna()

        v6_mean = v6_subset.mean()
        v7_mean = v7_subset.mean()
        change_pct = ((v7_mean - v6_mean) / abs(v6_mean) * 100) if v6_mean != 0 else 0

        quality_comparison.append({
            'label': label_name,
            'v6_mean': v6_mean,
            'v6_median': v6_subset.median(),
            'v7_mean': v7_mean,
            'v7_median': v7_subset.median(),
            'change_pct': change_pct,
            'n_neurons': len(v6_subset)
        })

        print(f"{label_name}: v6={v6_mean:.4f}, v7={v7_mean:.4f}, change={change_pct:+.2f}% (n={len(v6_subset)})")

    df_quality_comparison = pd.DataFrame(quality_comparison)
    output_path = output_dir / "quality_comparison.csv"
    df_quality_comparison.to_csv(output_path, index=False)
    print(f"\n[SAVED] Quality comparison: {output_path}")

    return df_quality_comparison


def create_visualizations(df_v6, df_v7, df_comparison, output_dir):
    """Create comprehensive comparison visualizations."""

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 1. Overall metric comparison bar plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    metrics_to_plot = df_comparison.sort_values('improvement_pct', ascending=False)

    x = np.arange(len(metrics_to_plot))
    width = 0.35

    bars1 = ax.bar(x - width/2, metrics_to_plot['v6_mean'], width, label='v6 (wavelet)', alpha=0.8)
    bars2 = ax.bar(x + width/2, metrics_to_plot['v7_mean'], width, label='v7 (threshold)', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Mean Value', fontsize=12)
    ax.set_title('Event-Based Metrics: Wavelet vs Threshold', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot['metric'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "metric_comparison_barplot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()

    # 2. Improvement percentage plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    colors = ['green' if x > 0 else 'red' for x in metrics_to_plot['improvement_pct']]
    bars = ax.barh(metrics_to_plot['metric'], metrics_to_plot['improvement_pct'], color=colors, alpha=0.7)

    ax.set_xlabel('Improvement % (positive = threshold better)', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    ax.set_title('Threshold vs Wavelet: Improvement Analysis', fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (bar, val) in enumerate(zip(bars, metrics_to_plot['improvement_pct'])):
        ax.text(val + (1 if val > 0 else -1), i, f'{val:+.1f}%',
                va='center', ha='left' if val > 0 else 'right', fontsize=9)

    plt.tight_layout()
    output_path = output_dir / "improvement_percentages.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()

    # 3. Distribution comparison for key metrics
    key_metrics = ['event_r2_score', 'r2_score', 'nmae', 'events_per_min']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric in enumerate(key_metrics):
        if metric not in df_v6.columns or metric not in df_v7.columns:
            continue

        ax = axes[idx]

        v6_vals = df_v6[metric].replace([np.inf, -np.inf], np.nan).dropna()
        v7_vals = df_v7[metric].replace([np.inf, -np.inf], np.nan).dropna()

        # Plot distributions
        ax.hist(v6_vals, bins=50, alpha=0.5, label='v6 (wavelet)', density=True, color='blue')
        ax.hist(v7_vals, bins=50, alpha=0.5, label='v7 (threshold)', density=True, color='orange')

        # Add mean lines
        ax.axvline(v6_vals.mean(), color='blue', linestyle='--', linewidth=2, label=f'v6 mean={v6_vals.mean():.3f}')
        ax.axvline(v7_vals.mean(), color='orange', linestyle='--', linewidth=2, label=f'v7 mean={v7_vals.mean():.3f}')

        ax.set_xlabel(metric, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{metric} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / "distribution_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {output_path}")
    plt.close()

    # 4. Scatter plot: v6 vs v7 event_r2_score
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Match neurons by session
    df_v6_sorted = df_v6.sort_values(['session', 'component_idx']).reset_index(drop=True)
    df_v7_sorted = df_v7.sort_values(['session', 'component_idx']).reset_index(drop=True)

    # Only compare if they have the same structure
    if len(df_v6_sorted) == len(df_v7_sorted) and \
       (df_v6_sorted['session'] == df_v7_sorted['session']).all() and \
       (df_v6_sorted['component_idx'] == df_v7_sorted['component_idx']).all():

        v6_r2 = df_v6_sorted['event_r2_score'].replace([np.inf, -np.inf], np.nan)
        v7_r2 = df_v7_sorted['event_r2_score'].replace([np.inf, -np.inf], np.nan)

        mask = ~(v6_r2.isna() | v7_r2.isna())

        ax.scatter(v6_r2[mask], v7_r2[mask], alpha=0.3, s=10)
        ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x (no change)')

        ax.set_xlabel('v6 (wavelet) event_r2_score', fontsize=12)
        ax.set_ylabel('v7 (threshold) event_r2_score', fontsize=12)
        ax.set_title('Neuron-by-Neuron Comparison: event_r2_score', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        plt.tight_layout()
        output_path = output_dir / "neuron_by_neuron_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  [SAVED] {output_path}")
    else:
        print("  [SKIP] Neuron-by-neuron scatter: datasets don't match exactly")

    plt.close()


def main():
    """Main analysis workflow."""
    project_root = Path(__file__).parents[2]
    output_dir = project_root / "analysis_event_method_comparison_2025_12_15" / "data"
    plots_dir = project_root / "analysis_event_method_comparison_2025_12_15" / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load data
    df_v6, df_v7 = load_datasets(project_root)

    # Compare overall metrics
    df_comparison = compare_metrics(df_v6, df_v7, output_dir)

    # Compare by experiment
    df_exp_comparison = compare_by_experiment(df_v6, df_v7, output_dir)

    # Compare by quality
    df_quality_comparison = compare_by_quality(df_v6, df_v7, output_dir)

    # Create visualizations
    create_visualizations(df_v6, df_v7, df_comparison, plots_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
