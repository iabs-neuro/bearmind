"""
Empirical Spearman Correlation Threshold Analysis for BEARMiND Merge Decisions.

Analyzes spatial correlation patterns in validation dataset to determine appropriate
Spearman correlation threshold equivalent to Pearson=0.6.

Usage:
    conda run -n bearmind python analyze_correlation_threshold.py --pilot
    conda run -n bearmind python analyze_correlation_threshold.py --full
"""

import numpy as np
import pandas as pd
import pickle
import glob
from pathlib import Path
from scipy.stats import spearmanr, linregress
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path('data/capcan_validation_99_v8')
RAW_DIR = Path('data/raw_compressed')
OUTPUT_DIR = Path('ml/results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Pilot sessions (representative sample, excluding 3DM - not in v8 dataset)
PILOT_SESSIONS = [
    'FOF_F05_1D',  # FOF experiment
    'NOF_H01_1D',  # NOF experiment
    'RFC_F01_1D',  # RFC experiment
    'NOF_H02_2D',  # Multi-day example
]


def find_pickle_file(session_name, raw_path=RAW_DIR):
    """
    Find pickle file for session handling naming variations.

    Args:
        session_name: Session name (e.g., 'FOF_F05_1D')
        raw_path: Path to raw_compressed directory

    Returns:
        Path object for pickle file

    Raises:
        FileNotFoundError: If no matching pickle file found
    """
    # Try glob pattern matching
    pattern = str(raw_path / f'{session_name}*estimates*.pickle')
    matches = glob.glob(pattern)

    if matches:
        return Path(matches[0])

    raise FileNotFoundError(
        f'No pickle file found for session {session_name}\n'
        f'Pattern: {pattern}'
    )


def find_close_pairs(fcd_matrix, max_distance=10):
    """
    Find neuron pairs within max_distance pixels.

    Args:
        fcd_matrix: N×N center distance matrix
        max_distance: Maximum distance threshold in pixels

    Returns:
        List of (idx1, idx2, distance) tuples (upper triangle only)
    """
    n = fcd_matrix.shape[0]
    pairs = []

    # Upper triangle only (avoid duplicates)
    for i in range(n):
        for j in range(i + 1, n):
            dist = fcd_matrix[i, j]
            if dist < max_distance:
                pairs.append((i, j, dist))

    return pairs


def compute_pair_correlations(traces, pairs, min_std=0.1):
    """
    Compute Pearson and Spearman correlations for neuron pairs.

    Args:
        traces: N × T array of calcium traces
        pairs: List of (idx1, idx2, distance) tuples
        min_std: Minimum standard deviation (filter low-variance traces)

    Returns:
        DataFrame with correlation results
    """
    results = []

    for idx1, idx2, distance in pairs:
        trace1 = traces[idx1, :]
        trace2 = traces[idx2, :]

        # Filter low-variance traces
        if np.std(trace1) < min_std or np.std(trace2) < min_std:
            continue

        # Compute correlations
        try:
            # Pearson
            pearson_corr = np.corrcoef(trace1, trace2)[0, 1]

            # Spearman
            spearman_corr, _ = spearmanr(trace1, trace2)

            # Store if valid
            if not (np.isnan(pearson_corr) or np.isnan(spearman_corr)):
                results.append({
                    'idx1': idx1,
                    'idx2': idx2,
                    'distance': distance,
                    'pearson_corr': pearson_corr,
                    'spearman_corr': spearman_corr
                })
        except Exception as e:
            # Skip problematic pairs
            continue

    return pd.DataFrame(results)


def process_single_session(session_name, max_distance=10):
    """
    Process one session: load data, find pairs, compute correlations.

    Args:
        session_name: Session name (e.g., 'FOF_F05_1D')
        max_distance: Maximum distance in pixels

    Returns:
        DataFrame with columns: session, experiment, idx1, idx2, distance,
                               pearson_corr, spearman_corr
    """
    # Extract experiment type
    experiment = session_name.split('_')[0]

    # Load FCD matrix (pre-computed distances)
    session_folder = DATA_DIR / f'capcan_artifacts_{session_name}'
    fcd_path = session_folder / 'FCD.npy'

    if not fcd_path.exists():
        raise FileNotFoundError(f'FCD.npy not found for {session_name}')

    fcd_matrix = np.load(fcd_path)

    # Find close pairs
    pairs = find_close_pairs(fcd_matrix, max_distance)

    if not pairs:
        # No pairs within distance
        return pd.DataFrame()

    # Load raw traces
    pickle_file = find_pickle_file(session_name)

    with open(pickle_file, 'rb') as f:
        estimates = pickle.load(f)

    # Get traces
    traces = estimates.C
    if hasattr(traces, 'toarray'):
        # Handle sparse matrices
        traces = traces.toarray()
    traces = traces.astype(np.float32)

    # Validate shape
    if fcd_matrix.shape[0] != traces.shape[0]:
        raise ValueError(
            f'Shape mismatch: FCD {fcd_matrix.shape[0]} vs traces {traces.shape[0]}'
        )

    # Compute correlations
    corr_df = compute_pair_correlations(traces, pairs)

    if len(corr_df) == 0:
        return pd.DataFrame()

    # Add metadata
    corr_df['session'] = session_name
    corr_df['experiment'] = experiment

    return corr_df


def find_equivalent_threshold(df, pearson_ref=0.6, window=0.02):
    """
    Find Spearman threshold equivalent to Pearson reference using three methods.

    Args:
        df: DataFrame with pearson_corr and spearman_corr columns
        pearson_ref: Reference Pearson threshold (default: 0.6)
        window: Window for empirical matching (default: 0.02)

    Returns:
        Dict with three threshold estimates and statistics
    """
    results = {}

    # Method 1: Linear Regression
    slope, intercept, r_value, _, _ = linregress(
        df['pearson_corr'], df['spearman_corr']
    )
    regression_threshold = slope * pearson_ref + intercept

    results['regression'] = {
        'threshold': regression_threshold,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2
    }

    # Method 2: Empirical Matching
    pearson_window = df[
        (df['pearson_corr'] >= pearson_ref - window) &
        (df['pearson_corr'] <= pearson_ref + window)
    ]

    if len(pearson_window) > 0:
        empirical_threshold = pearson_window['spearman_corr'].median()
        results['empirical'] = {
            'threshold': empirical_threshold,
            'n_pairs': len(pearson_window),
            'q25': pearson_window['spearman_corr'].quantile(0.25),
            'q75': pearson_window['spearman_corr'].quantile(0.75)
        }
    else:
        results['empirical'] = {'threshold': np.nan, 'n_pairs': 0}

    # Method 3: Quantile Matching
    pearson_percentile = (df['pearson_corr'] < pearson_ref).sum() / len(df) * 100
    quantile_threshold = df['spearman_corr'].quantile(pearson_percentile / 100)

    results['quantile'] = {
        'threshold': quantile_threshold,
        'percentile': pearson_percentile
    }

    # Consensus recommendation
    valid_thresholds = [
        v['threshold'] for v in results.values()
        if not np.isnan(v['threshold'])
    ]

    if valid_thresholds:
        results['recommendation'] = {
            'threshold': np.median(valid_thresholds),
            'min': np.min(valid_thresholds),
            'max': np.max(valid_thresholds),
            'range': np.max(valid_thresholds) - np.min(valid_thresholds)
        }

    return results


def plot_distance_vs_correlation(df, method, output_path):
    """
    Create distance vs correlation scatter plot.

    Args:
        df: DataFrame with distance and correlation columns
        method: 'pearson' or 'spearman'
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    corr_col = f'{method}_corr'

    # Hexbin for density
    hb = ax.hexbin(df['distance'], df[corr_col],
                   gridsize=30, cmap='Blues', mincnt=1, alpha=0.8)

    # Threshold line
    ax.axhline(y=0.6 if method == 'pearson' else df[corr_col].median(),
               color='red', linestyle='--', linewidth=2,
               label=f'Threshold ({method.capitalize()})')

    ax.set_xlabel('Distance (pixels)', fontsize=12)
    ax.set_ylabel(f'{method.capitalize()} Correlation', fontsize=12)
    ax.set_title(f'Distance vs {method.capitalize()} Correlation\n'
                 f'N = {len(df):,} pairs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.colorbar(hb, ax=ax, label='Pair count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_pearson_vs_spearman(df, threshold_results, output_path):
    """
    Create Pearson vs Spearman comparison plot.

    Args:
        df: DataFrame with both correlation columns
        threshold_results: Results from find_equivalent_threshold()
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 10), dpi=300)

    # Hexbin scatter
    hb = ax.hexbin(df['pearson_corr'], df['spearman_corr'],
                   gridsize=40, cmap='viridis', mincnt=1)

    # Diagonal reference line
    lims = [-0.2, 1.0]
    ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1.5, label='y=x (perfect agreement)')

    # Pearson threshold line
    ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2,
               label='Pearson threshold (0.6)')

    # Recommended Spearman threshold
    if 'recommendation' in threshold_results:
        spearman_thr = threshold_results['recommendation']['threshold']
        ax.axhline(y=spearman_thr, color='orange', linestyle='--', linewidth=2,
                   label=f'Recommended Spearman ({spearman_thr:.3f})')

    # Regression line
    if 'regression' in threshold_results:
        reg = threshold_results['regression']
        x_line = np.array(lims)
        y_line = reg['slope'] * x_line + reg['intercept']
        ax.plot(x_line, y_line, 'g-', linewidth=2, alpha=0.7,
                label=f"Regression (R²={reg['r_squared']:.3f})")

    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title(f'Pearson vs Spearman Correlation\nN = {len(df):,} pairs',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.colorbar(hb, ax=ax, label='Pair count')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_correlation_distributions(df, output_path):
    """
    Create histogram comparison of correlation distributions.

    Args:
        df: DataFrame with both correlation columns
        output_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # Pearson histogram
    axes[0].hist(df['pearson_corr'], bins=50, color='steelblue',
                 alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0.6, color='red', linestyle='--', linewidth=2,
                    label='Current threshold')
    axes[0].axvline(x=df['pearson_corr'].median(), color='green',
                    linestyle=':', linewidth=2, label='Median')
    axes[0].set_xlabel('Pearson Correlation', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title(f'Pearson Distribution\nMean: {df["pearson_corr"].mean():.3f}, '
                      f'Median: {df["pearson_corr"].median():.3f}',
                      fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spearman histogram
    axes[1].hist(df['spearman_corr'], bins=50, color='coral',
                 alpha=0.7, edgecolor='black')
    axes[1].axvline(x=df['spearman_corr'].median(), color='green',
                    linestyle=':', linewidth=2, label='Median')
    axes[1].set_xlabel('Spearman Correlation', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title(f'Spearman Distribution\nMean: {df["spearman_corr"].mean():.3f}, '
                      f'Median: {df["spearman_corr"].median():.3f}',
                      fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_threshold_equivalence(df, threshold_results, output_path):
    """
    Create threshold equivalence visualization.

    Args:
        df: DataFrame with both correlation columns
        threshold_results: Results from find_equivalent_threshold()
        output_path: Output file path
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    # Background scatter (faint)
    ax.scatter(df['pearson_corr'], df['spearman_corr'],
               alpha=0.1, s=10, c='gray', label='All pairs')

    # Highlight pairs near Pearson=0.6
    near_threshold = df[
        (df['pearson_corr'] >= 0.58) & (df['pearson_corr'] <= 0.62)
    ]
    ax.scatter(near_threshold['pearson_corr'], near_threshold['spearman_corr'],
               alpha=0.5, s=20, c='orange', label='Pearson ≈ 0.6 (±0.02)')

    # Regression line with confidence interval
    if 'regression' in threshold_results:
        reg = threshold_results['regression']
        x_line = np.linspace(df['pearson_corr'].min(), df['pearson_corr'].max(), 100)
        y_line = reg['slope'] * x_line + reg['intercept']
        ax.plot(x_line, y_line, 'b-', linewidth=3, alpha=0.8,
                label=f"Regression (R²={reg['r_squared']:.3f})")

    # Mark Pearson=0.6
    ax.axvline(x=0.6, color='red', linestyle='--', linewidth=2,
               label='Pearson 0.6')

    # Mark recommended Spearman threshold
    if 'recommendation' in threshold_results:
        spearman_thr = threshold_results['recommendation']['threshold']
        ax.axhline(y=spearman_thr, color='green', linestyle='--', linewidth=2,
                   label=f'Spearman {spearman_thr:.3f}')

        # Intersection point
        ax.plot([0.6], [spearman_thr], 'ro', markersize=12,
                markeredgewidth=2, markeredgecolor='darkred',
                label='Threshold mapping')

    ax.set_xlabel('Pearson Correlation', fontsize=12)
    ax.set_ylabel('Spearman Correlation', fontsize=12)
    ax.set_title('Mapping Pearson 0.6 → Spearman Threshold',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(df, threshold_results, output_path, is_pilot=True):
    """
    Generate statistical summary report.

    Args:
        df: DataFrame with correlation results
        threshold_results: Results from find_equivalent_threshold()
        output_path: Output file path
        is_pilot: Whether this is pilot run
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CORRELATION THRESHOLD ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis type: {'PILOT (5 sessions)' if is_pilot else 'FULL (99 sessions)'}\n")
        f.write(f"Distance filter: Neuron pairs within 10 pixels\n\n")

        f.write("1. DATASET SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total sessions analyzed: {df['session'].nunique()}\n")
        f.write(f"Total neuron pairs: {len(df):,}\n\n")

        # By experiment type
        exp_counts = df.groupby('experiment').size()
        for exp, count in exp_counts.items():
            pct = count / len(df) * 100
            f.write(f"  - {exp}: {count:,} pairs ({pct:.1f}%)\n")

        f.write("\n2. CORRELATION DISTRIBUTIONS\n")
        f.write("-" * 80 + "\n")

        f.write("PEARSON CORRELATION:\n")
        f.write(f"  Mean ± Std: {df['pearson_corr'].mean():.3f} ± {df['pearson_corr'].std():.3f}\n")
        f.write(f"  Median: {df['pearson_corr'].median():.3f}\n")
        f.write(f"  Range: [{df['pearson_corr'].min():.3f}, {df['pearson_corr'].max():.3f}]\n")
        f.write(f"  Quartiles: Q1={df['pearson_corr'].quantile(0.25):.3f}, "
                f"Q2={df['pearson_corr'].quantile(0.50):.3f}, "
                f"Q3={df['pearson_corr'].quantile(0.75):.3f}\n\n")

        f.write("SPEARMAN CORRELATION:\n")
        f.write(f"  Mean ± Std: {df['spearman_corr'].mean():.3f} ± {df['spearman_corr'].std():.3f}\n")
        f.write(f"  Median: {df['spearman_corr'].median():.3f}\n")
        f.write(f"  Range: [{df['spearman_corr'].min():.3f}, {df['spearman_corr'].max():.3f}]\n")
        f.write(f"  Quartiles: Q1={df['spearman_corr'].quantile(0.25):.3f}, "
                f"Q2={df['spearman_corr'].quantile(0.50):.3f}, "
                f"Q3={df['spearman_corr'].quantile(0.75):.3f}\n\n")

        f.write("\n3. THRESHOLD MAPPING: PEARSON 0.6 → SPEARMAN\n")
        f.write("-" * 80 + "\n")
        f.write("Current threshold: Pearson correlation >= 0.6\n\n")

        if 'regression' in threshold_results:
            reg = threshold_results['regression']
            f.write("METHOD 1: Regression-based prediction\n")
            f.write(f"  Spearman = {reg['slope']:.4f} × Pearson + {reg['intercept']:.4f}\n")
            f.write(f"  R² = {reg['r_squared']:.4f}\n")
            f.write(f"  Predicted Spearman threshold: {reg['threshold']:.3f}\n\n")

        if 'empirical' in threshold_results and threshold_results['empirical']['n_pairs'] > 0:
            emp = threshold_results['empirical']
            f.write("METHOD 2: Empirical matching (Pearson 0.58-0.62 window)\n")
            f.write(f"  Sample size: {emp['n_pairs']:,} pairs\n")
            f.write(f"  Median Spearman: {emp['threshold']:.3f}\n")
            f.write(f"  IQR: [{emp['q25']:.3f}, {emp['q75']:.3f}]\n\n")

        if 'quantile' in threshold_results:
            quant = threshold_results['quantile']
            f.write("METHOD 3: Quantile matching\n")
            f.write(f"  Pearson 0.6 represents {quant['percentile']:.1f}th percentile\n")
            f.write(f"  Equivalent Spearman percentile: {quant['threshold']:.3f}\n\n")

        if 'recommendation' in threshold_results:
            rec = threshold_results['recommendation']
            f.write("\n4. FINAL RECOMMENDATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"RECOMMENDED SPEARMAN THRESHOLD: {rec['threshold']:.3f}\n\n")
            f.write(f"Consensus range: [{rec['min']:.3f}, {rec['max']:.3f}]\n")
            f.write(f"Method agreement: ±{rec['range']:.3f}\n\n")

            # Expected impact
            above_pearson = (df['pearson_corr'] >= 0.6).sum()
            above_spearman = (df['spearman_corr'] >= rec['threshold']).sum()
            f.write("Expected impact:\n")
            f.write(f"  Pairs above Pearson 0.6: {above_pearson:,} ({above_pearson/len(df)*100:.1f}%)\n")
            f.write(f"  Pairs above Spearman {rec['threshold']:.3f}: {above_spearman:,} ({above_spearman/len(df)*100:.1f}%)\n")
            f.write(f"  Difference: {abs(above_spearman - above_pearson):,} pairs\n\n")

        f.write("=" * 80 + "\n")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze correlation thresholds for Pearson vs Spearman'
    )
    parser.add_argument('--pilot', action='store_true',
                        help='Run pilot analysis (5 sessions)')
    parser.add_argument('--full', action='store_true',
                        help='Run full analysis (99 sessions)')

    args = parser.parse_args()

    if not (args.pilot or args.full):
        print("Please specify --pilot or --full")
        return

    # Select sessions
    if args.pilot:
        sessions = PILOT_SESSIONS
        suffix = 'pilot'
    else:
        # Get all sessions from directory (excluding 3DM - not in v8 dataset)
        session_folders = sorted(DATA_DIR.glob('capcan_artifacts_*'))
        sessions = [f.name.replace('capcan_artifacts_', '') for f in session_folders
                   if not f.name.startswith('capcan_artifacts_3DM')]
        suffix = 'full'

    print(f"\n{'='*80}")
    print(f"CORRELATION THRESHOLD ANALYSIS - {'PILOT' if args.pilot else 'FULL'}")
    print(f"{'='*80}\n")
    print(f"Sessions to process: {len(sessions)}")
    print(f"Distance filter: <10 pixels\n")

    # Process sessions
    all_results = []
    failed_sessions = []

    for session_name in tqdm(sessions, desc='Processing sessions'):
        try:
            df = process_single_session(session_name, max_distance=10)
            if len(df) > 0:
                all_results.append(df)
        except Exception as e:
            failed_sessions.append((session_name, str(e)))
            print(f"\nWarning: Failed to process {session_name}: {e}")
            continue

    if not all_results:
        print("\nERROR: No sessions processed successfully!")
        return

    # Combine results
    combined_df = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {len(all_results)}/{len(sessions)} sessions")
    print(f"Total neuron pairs: {len(combined_df):,}")
    print(f"Failed sessions: {len(failed_sessions)}")

    if failed_sessions:
        print("\nFailed sessions:")
        for session, error in failed_sessions:
            print(f"  - {session}: {error}")

    # Save CSV
    csv_path = OUTPUT_DIR / f'correlation_pairs_{suffix}.csv'
    combined_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Calculate threshold recommendations
    print("\nCalculating threshold equivalence...")
    threshold_results = find_equivalent_threshold(combined_df, pearson_ref=0.6)

    # Generate visualizations
    print("Generating plots...")
    plot_distance_vs_correlation(combined_df, 'pearson',
                                  OUTPUT_DIR / f'distance_vs_pearson_{suffix}.png')
    plot_distance_vs_correlation(combined_df, 'spearman',
                                  OUTPUT_DIR / f'distance_vs_spearman_{suffix}.png')
    plot_pearson_vs_spearman(combined_df, threshold_results,
                             OUTPUT_DIR / f'pearson_vs_spearman_{suffix}.png')
    plot_correlation_distributions(combined_df,
                                    OUTPUT_DIR / f'correlation_distributions_{suffix}.png')
    plot_threshold_equivalence(combined_df, threshold_results,
                               OUTPUT_DIR / f'threshold_equivalence_{suffix}.png')

    # Generate report
    report_path = OUTPUT_DIR / f'threshold_recommendation_{suffix}.txt'
    generate_report(combined_df, threshold_results, report_path, is_pilot=args.pilot)
    print(f"Saved: {report_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("THRESHOLD RECOMMENDATION")
    print(f"{'='*80}")
    if 'recommendation' in threshold_results:
        rec = threshold_results['recommendation']
        print(f"Recommended Spearman threshold: {rec['threshold']:.3f}")
        print(f"Range: [{rec['min']:.3f}, {rec['max']:.3f}]")
    print(f"\nFull report: {report_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
