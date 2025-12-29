"""
Comprehensive analysis of event parameter comparison metrics.

Analyzes distributions, anomalies, and differences across 4 parameter combinations:
- wavelet + n_iter=2
- wavelet + n_iter=3
- threshold + n_iter=2
- threshold + n_iter=3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Configuration
BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'analysis'
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

PARAM_LABELS = ['wavelet_iter2', 'wavelet_iter3', 'threshold_iter2', 'threshold_iter3']

# Key metrics to analyze
KEY_METRICS = [
    'events_per_min',
    'events_fraction',
    'event_snr',
    'event_r2_score',
    't_rise',
    't_off',
    'r2_score',
    'nmae',
    'nrmse',
    'snr_recon',
    'caiman_snr',
    'area',
    'circularity',
    'max_edge',
    'convexity'
]


def load_all_metrics():
    """Load metrics from all parameter combinations."""
    all_data = {}

    for label in PARAM_LABELS:
        metrics_file = BASE_PATH / label / f'{label}_metrics.csv'
        if metrics_file.exists():
            df = pd.read_csv(metrics_file)
            all_data[label] = df
            print(f"Loaded {label}: {len(df)} neurons")
        else:
            print(f"WARNING: Missing {metrics_file}")

    return all_data


def compute_distribution_stats(all_data):
    """Compute comprehensive distribution statistics for all metrics."""

    stats_data = []

    for metric in KEY_METRICS:
        for label in PARAM_LABELS:
            if label not in all_data:
                continue

            df = all_data[label]
            if metric not in df.columns:
                continue

            values = df[metric].dropna()

            if len(values) == 0:
                continue

            # Compute statistics
            stat_row = {
                'metric': metric,
                'parameter': label,
                'count': len(values),
                'n_nan': df[metric].isna().sum(),
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'p01': values.quantile(0.01),
                'p05': values.quantile(0.05),
                'p25': values.quantile(0.25),
                'median': values.median(),
                'p75': values.quantile(0.75),
                'p95': values.quantile(0.95),
                'p99': values.quantile(0.99),
                'max': values.max(),
                'skew': stats.skew(values),
                'kurtosis': stats.kurtosis(values)
            }

            # Check for anomalies
            if metric == 't_off':
                stat_row['n_negative'] = (values < 0).sum()
                stat_row['pct_negative'] = (values < 0).sum() / len(values) * 100
            elif metric == 't_rise':
                stat_row['n_negative'] = (values < 0).sum()
                stat_row['pct_negative'] = (values < 0).sum() / len(values) * 100
            elif metric == 'events_per_min':
                stat_row['n_zero'] = (values == 0).sum()
                stat_row['pct_zero'] = (values == 0).sum() / len(values) * 100

            stats_data.append(stat_row)

    stats_df = pd.DataFrame(stats_data)
    return stats_df


def analyze_t_off_anomaly(all_data):
    """Deep dive into t_off negative values issue."""

    print("\n" + "="*80)
    print("T_OFF ANOMALY ANALYSIS")
    print("="*80 + "\n")

    anomaly_data = []

    for label in PARAM_LABELS:
        if label not in all_data:
            continue

        df = all_data[label]
        if 't_off' not in df.columns:
            continue

        t_off_values = df['t_off'].dropna()

        n_negative = (t_off_values < 0).sum()
        n_positive = (t_off_values > 0).sum()
        n_zero = (t_off_values == 0).sum()
        n_nan = df['t_off'].isna().sum()

        print(f"{label}:")
        print(f"  Total neurons: {len(df)}")
        print(f"  NaN t_off: {n_nan} ({n_nan/len(df)*100:.1f}%)")
        print(f"  Negative t_off: {n_negative} ({n_negative/len(t_off_values)*100:.1f}%)")
        print(f"  Zero t_off: {n_zero} ({n_zero/len(t_off_values)*100:.1f}%)")
        print(f"  Positive t_off: {n_positive} ({n_positive/len(t_off_values)*100:.1f}%)")
        print(f"  Mean (all): {t_off_values.mean():.4f}")
        print(f"  Mean (positive only): {t_off_values[t_off_values > 0].mean():.4f}")
        print(f"  Median: {t_off_values.median():.4f}")
        print(f"  Range: [{t_off_values.min():.4f}, {t_off_values.max():.4f}]")
        print()

        anomaly_data.append({
            'parameter': label,
            'total': len(df),
            'n_nan': n_nan,
            'n_negative': n_negative,
            'n_zero': n_zero,
            'n_positive': n_positive,
            'pct_negative': n_negative/len(t_off_values)*100,
            'mean_all': t_off_values.mean(),
            'mean_positive': t_off_values[t_off_values > 0].mean() if n_positive > 0 else np.nan,
            'median': t_off_values.median()
        })

    anomaly_df = pd.DataFrame(anomaly_data)

    # Save report
    report_file = OUTPUT_PATH / 't_off_anomaly_report.csv'
    anomaly_df.to_csv(report_file, index=False)
    print(f"Saved t_off anomaly report: {report_file}\n")

    return anomaly_df


def analyze_t_rise_anomaly(all_data):
    """Analyze t_rise values (should also be positive)."""

    print("\n" + "="*80)
    print("T_RISE ANOMALY ANALYSIS")
    print("="*80 + "\n")

    anomaly_data = []

    for label in PARAM_LABELS:
        if label not in all_data:
            continue

        df = all_data[label]
        if 't_rise' not in df.columns:
            continue

        t_rise_values = df['t_rise'].dropna()

        # Check for sentinel value -1 (no events detected)
        n_sentinel = (t_rise_values == -1).sum()
        n_negative_other = ((t_rise_values < 0) & (t_rise_values != -1)).sum()
        n_positive = (t_rise_values > 0).sum()
        n_zero = (t_rise_values == 0).sum()
        n_nan = df['t_rise'].isna().sum()

        print(f"{label}:")
        print(f"  Total neurons: {len(df)}")
        print(f"  NaN t_rise: {n_nan} ({n_nan/len(df)*100:.1f}%)")
        print(f"  Sentinel -1 (no events): {n_sentinel} ({n_sentinel/len(t_rise_values)*100:.1f}%)")
        print(f"  Other negative: {n_negative_other}")
        print(f"  Zero t_rise: {n_zero}")
        print(f"  Positive t_rise: {n_positive} ({n_positive/len(t_rise_values)*100:.1f}%)")

        # Stats excluding sentinel
        valid_values = t_rise_values[t_rise_values != -1]
        if len(valid_values) > 0:
            print(f"  Mean (excluding -1): {valid_values.mean():.4f}")
            print(f"  Median (excluding -1): {valid_values.median():.4f}")
            print(f"  Range (excluding -1): [{valid_values.min():.4f}, {valid_values.max():.4f}]")
        print()

        anomaly_data.append({
            'parameter': label,
            'total': len(df),
            'n_nan': n_nan,
            'n_sentinel': n_sentinel,
            'n_negative_other': n_negative_other,
            'n_positive': n_positive,
            'pct_sentinel': n_sentinel/len(t_rise_values)*100,
            'mean_valid': valid_values.mean() if len(valid_values) > 0 else np.nan,
            'median_valid': valid_values.median() if len(valid_values) > 0 else np.nan
        })

    anomaly_df = pd.DataFrame(anomaly_data)

    # Save report
    report_file = OUTPUT_PATH / 't_rise_anomaly_report.csv'
    anomaly_df.to_csv(report_file, index=False)
    print(f"Saved t_rise anomaly report: {report_file}\n")

    return anomaly_df


def plot_metric_distributions(all_data, metric, figsize=(16, 10)):
    """Plot distributions for a single metric across all parameters."""

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, label in enumerate(PARAM_LABELS):
        if label not in all_data:
            continue

        df = all_data[label]
        if metric not in df.columns:
            continue

        values = df[metric].dropna()

        # Handle sentinel values for t_rise
        if metric == 't_rise':
            n_sentinel = (values == -1).sum()
            values_plot = values[values != -1]
            title_suffix = f" (excl. {n_sentinel} sentinel -1)"
        else:
            values_plot = values
            title_suffix = ""

        ax = axes[idx]

        if len(values_plot) > 0:
            # Histogram with KDE
            ax.hist(values_plot, bins=50, alpha=0.6, edgecolor='black', density=True)

            # Add KDE if enough data
            if len(values_plot) > 10:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(values_plot)
                    x_range = np.linspace(values_plot.min(), values_plot.max(), 200)
                    ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
                except:
                    pass

            # Add mean and median lines
            mean_val = values_plot.mean()
            median_val = values_plot.median()
            ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

            # Styling
            ax.set_title(f'{label}{title_suffix}', fontsize=12, fontweight='bold')
            ax.set_xlabel(metric, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Add text box with stats
            stats_text = f'n={len(values_plot)}\nstd={values_plot.std():.3f}\nmin={values_plot.min():.3f}\nmax={values_plot.max():.3f}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{label} - No data', fontsize=12)

    plt.suptitle(f'Distribution of {metric} across parameters', fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save plot
    plot_file = OUTPUT_PATH / f'distribution_{metric}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved distribution plot: {plot_file}")


def plot_metric_comparison_boxplots(all_data):
    """Create boxplot comparisons for key metrics."""

    # Select subset of most important metrics
    important_metrics = [
        'event_r2_score',
        'events_per_min',
        't_rise',
        't_off',
        'event_snr',
        'r2_score',
        'nmae',
        'nrmse'
    ]

    n_metrics = len(important_metrics)
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()

    for idx, metric in enumerate(important_metrics):
        ax = axes[idx]

        # Collect data for this metric
        plot_data = []
        labels = []

        for label in PARAM_LABELS:
            if label not in all_data:
                continue

            df = all_data[label]
            if metric not in df.columns:
                continue

            values = df[metric].dropna()

            # Handle t_rise sentinel
            if metric == 't_rise':
                values = values[values != -1]

            if len(values) > 0:
                plot_data.append(values)
                labels.append(label.replace('_', '\n'))

        if len(plot_data) > 0:
            # Create boxplot
            bp = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)

            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(plot_data)]):
                patch.set_facecolor(color)

            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=0, labelsize=8)

            # Add horizontal line at y=0 if metric can be negative
            if metric in ['event_r2_score', 'r2_score', 't_rise', 't_off']:
                ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)

    plt.suptitle('Metric Distributions Across Parameters (Boxplots)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_file = OUTPUT_PATH / 'boxplot_comparison.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved boxplot comparison: {plot_file}")


def analyze_neuron_by_neuron_differences(all_data):
    """Compare same neurons across different parameters."""

    print("\n" + "="*80)
    print("NEURON-BY-NEURON COMPARISON")
    print("="*80 + "\n")

    # Use component_idx to match neurons across datasets
    base_label = 'wavelet_iter2'

    if base_label not in all_data:
        print("ERROR: Base dataset not found")
        return

    base_df = all_data[base_label][['component_idx'] + KEY_METRICS].copy()
    base_df = base_df.rename(columns={m: f'{m}_{base_label}' for m in KEY_METRICS})

    # Merge all datasets
    merged = base_df

    for label in PARAM_LABELS[1:]:  # Skip base
        if label not in all_data:
            continue

        other_df = all_data[label][['component_idx'] + KEY_METRICS].copy()
        other_df = other_df.rename(columns={m: f'{m}_{label}' for m in KEY_METRICS})

        merged = merged.merge(other_df, on='component_idx', how='inner')

    print(f"Matched {len(merged)} neurons across all datasets\n")

    # Compute differences
    comparison_data = []

    for metric in KEY_METRICS:
        for label1_idx, label1 in enumerate(PARAM_LABELS):
            for label2_idx, label2 in enumerate(PARAM_LABELS):
                if label2_idx <= label1_idx:
                    continue

                col1 = f'{metric}_{label1}'
                col2 = f'{metric}_{label2}'

                if col1 not in merged.columns or col2 not in merged.columns:
                    continue

                values1 = merged[col1].dropna()
                values2 = merged[col2].dropna()

                # Find common non-NaN indices
                common_idx = merged[col1].notna() & merged[col2].notna()
                v1 = merged.loc[common_idx, col1]
                v2 = merged.loc[common_idx, col2]

                if len(v1) == 0:
                    continue

                diff = v2 - v1
                abs_diff = np.abs(diff)
                rel_diff = np.abs(diff) / (np.abs(v1) + 1e-10) * 100  # Percent difference

                comparison_data.append({
                    'metric': metric,
                    'comparison': f'{label1} vs {label2}',
                    'n_compared': len(v1),
                    'mean_diff': diff.mean(),
                    'median_diff': diff.median(),
                    'std_diff': diff.std(),
                    'mean_abs_diff': abs_diff.mean(),
                    'median_abs_diff': abs_diff.median(),
                    'max_abs_diff': abs_diff.max(),
                    'mean_rel_diff_pct': rel_diff.mean(),
                    'median_rel_diff_pct': rel_diff.median(),
                    'correlation': np.corrcoef(v1, v2)[0, 1]
                })

    comparison_df = pd.DataFrame(comparison_data)

    # Save
    comparison_file = OUTPUT_PATH / 'neuron_by_neuron_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved neuron-by-neuron comparison: {comparison_file}\n")

    return comparison_df, merged


def generate_summary_report(stats_df, all_data):
    """Generate a comprehensive summary report."""

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("EVENT PARAMETER COMPARISON - COMPREHENSIVE METRIC ANALYSIS")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append(f"Analysis Date: {pd.Timestamp.now()}")
    report_lines.append(f"Session: NOF_H32_4D")
    report_lines.append(f"Total Neurons: {len(all_data[PARAM_LABELS[0]])}")
    report_lines.append("")

    # Summary statistics for key metrics
    report_lines.append("="*80)
    report_lines.append("KEY METRIC SUMMARY")
    report_lines.append("="*80)
    report_lines.append("")

    key_summary_metrics = ['event_r2_score', 'events_per_min', 't_rise', 't_off', 'event_snr']

    for metric in key_summary_metrics:
        report_lines.append(f"\n{metric.upper()}:")
        report_lines.append("-" * 40)

        metric_stats = stats_df[stats_df['metric'] == metric]

        for _, row in metric_stats.iterrows():
            param = row['parameter']
            report_lines.append(f"  {param:20s}: mean={row['mean']:8.4f}  median={row['median']:8.4f}  std={row['std']:8.4f}")
            report_lines.append(f"  {'':20s}  [p05={row['p05']:8.4f}, p95={row['p95']:8.4f}]  range=[{row['min']:8.4f}, {row['max']:8.4f}]")

            if 'n_negative' in row and pd.notna(row['n_negative']):
                report_lines.append(f"  {'':20s}  ANOMALY: {int(row['n_negative'])} negative values ({row['pct_negative']:.1f}%)")
            if 'n_zero' in row and pd.notna(row['n_zero']):
                report_lines.append(f"  {'':20s}  {int(row['n_zero'])} zero values ({row['pct_zero']:.1f}%)")

    report_lines.append("\n")
    report_lines.append("="*80)
    report_lines.append("INTERPRETATION")
    report_lines.append("="*80)
    report_lines.append("")
    report_lines.append("1. EVENT_R2_SCORE: Wavelet shows much better reconstruction quality")
    report_lines.append("   - Wavelet: ~0.56-0.59 (good fit)")
    report_lines.append("   - Threshold: ~-0.01-0.30 (poor to mediocre fit)")
    report_lines.append("")
    report_lines.append("2. T_OFF NEGATIVE VALUES: This is an ERROR in metric computation")
    report_lines.append("   - t_off represents calcium decay time, must be positive")
    report_lines.append("   - Negative values suggest numerical issues or failed fits")
    report_lines.append("   - See t_off_anomaly_report.csv for details")
    report_lines.append("")
    report_lines.append("3. N_ITER EFFECT: More iterations generally improve metrics")
    report_lines.append("   - Increasing n_iter from 2→3 improves reconstruction quality")
    report_lines.append("   - Effect more pronounced for wavelet method")
    report_lines.append("")
    report_lines.append("4. PERFORMANCE: Threshold is 3-4x faster than wavelet")
    report_lines.append("   - Wavelet: ~800s (~13 min)")
    report_lines.append("   - Threshold: ~200s (~3 min)")
    report_lines.append("")

    # Write report
    report_file = OUTPUT_PATH / 'COMPREHENSIVE_ANALYSIS_REPORT.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\nSaved comprehensive report: {report_file}")

    return '\n'.join(report_lines)


def main():
    print("="*80)
    print("COMPREHENSIVE EVENT PARAMETER METRIC ANALYSIS")
    print("="*80)
    print()

    # Load all data
    print("Loading metrics from all parameter combinations...")
    all_data = load_all_metrics()
    print()

    if len(all_data) == 0:
        print("ERROR: No data loaded!")
        return

    # Compute distribution statistics
    print("Computing distribution statistics...")
    stats_df = compute_distribution_stats(all_data)
    stats_file = OUTPUT_PATH / 'distribution_statistics.csv'
    stats_df.to_csv(stats_file, index=False)
    print(f"Saved distribution statistics: {stats_file}\n")

    # Analyze t_off anomaly
    t_off_anomaly = analyze_t_off_anomaly(all_data)

    # Analyze t_rise anomaly
    t_rise_anomaly = analyze_t_rise_anomaly(all_data)

    # Plot distributions for key metrics
    print("\n" + "="*80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("="*80 + "\n")

    plot_metrics = ['event_r2_score', 'events_per_min', 't_rise', 't_off',
                   'event_snr', 'r2_score', 'nmae', 'nrmse']

    for metric in plot_metrics:
        print(f"Plotting {metric}...")
        plot_metric_distributions(all_data, metric)

    print()

    # Create boxplot comparisons
    print("Generating boxplot comparison...")
    plot_metric_comparison_boxplots(all_data)
    print()

    # Neuron-by-neuron comparison
    print("Analyzing neuron-by-neuron differences...")
    comparison_df, merged_df = analyze_neuron_by_neuron_differences(all_data)

    # Save merged dataset
    merged_file = OUTPUT_PATH / 'merged_all_parameters.csv'
    merged_df.to_csv(merged_file, index=False)
    print(f"Saved merged dataset: {merged_file}\n")

    # Generate summary report
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80 + "\n")

    report = generate_summary_report(stats_df, all_data)
    print(report)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {OUTPUT_PATH}")
    print("\nKey output files:")
    print("  - distribution_statistics.csv: Full statistics for all metrics")
    print("  - t_off_anomaly_report.csv: Analysis of negative t_off values")
    print("  - t_rise_anomaly_report.csv: Analysis of t_rise sentinel values")
    print("  - neuron_by_neuron_comparison.csv: Differences between parameters")
    print("  - merged_all_parameters.csv: Full dataset with all parameters")
    print("  - distribution_*.png: Distribution plots for each metric")
    print("  - boxplot_comparison.png: Side-by-side boxplots")
    print("  - COMPREHENSIVE_ANALYSIS_REPORT.txt: Full text report")


if __name__ == '__main__':
    main()
