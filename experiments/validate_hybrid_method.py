"""
Hybrid Method Validation Script
================================
Automated validation of hybrid n=3 event detection method on specified session.

Runs all 5 methods (wavelet_iter2, wavelet_iter3, threshold_iter2, threshold_iter3, hybrid_iter3)
and generates comprehensive comparison report with hypothesis validation.

Usage:
    python validate_hybrid_method.py --session NOF_H09_4D --fps 30
    python validate_hybrid_method.py --session NOF_H09_4D --skip-computation
"""

import argparse
import pickle
import glob
import time
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from auto_inspector import estimates_to_metrics


# Method configurations
METHODS = [
    {
        'label': 'wavelet_iter2',
        'event_method': 'wavelet',
        'n_iter': 2,
        'hybrid_kinetics': False,
    },
    {
        'label': 'wavelet_iter3',
        'event_method': 'wavelet',
        'n_iter': 3,
        'hybrid_kinetics': False,
    },
    {
        'label': 'threshold_iter2',
        'event_method': 'threshold',
        'n_iter': 2,
        'hybrid_kinetics': False,
    },
    {
        'label': 'threshold_iter3',
        'event_method': 'threshold',
        'n_iter': 3,
        'hybrid_kinetics': False,
    },
    {
        'label': 'hybrid_iter3',
        'event_method': 'wavelet',
        'n_iter': 3,
        'hybrid_kinetics': True,
    },
]


def find_estimates_file(session_name: str) -> str:
    """Find estimates pickle file for given session."""
    pattern = f'data/raw_compressed/{session_name}*estimates.pickle'
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No estimates file found matching: {pattern}")

    if len(matches) > 1:
        raise ValueError(f"Multiple estimates files found: {matches}")

    return matches[0]


def load_estimates(filepath: str):
    """Load CaImAn estimates from pickle file."""
    print(f"Loading estimates from: {filepath}")
    with open(filepath, 'rb') as f:
        est = pickle.load(f)
    print(f"  Loaded {len(est.idx_components)} neurons")
    return est


def run_single_method(est, method_config: Dict, fps: int, output_dir: Path) -> Tuple[pd.DataFrame, str]:
    """
    Run a single event detection method.

    Returns:
        metrics_df: DataFrame with metrics for all neurons
        output_file: Path to saved CSV file
    """
    label = method_config['label']
    method_output_dir = output_dir / label
    method_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Running: {label}")
    print(f"{'='*80}")
    print(f"  event_method: {method_config['event_method']}")
    print(f"  n_iter: {method_config['n_iter']}")
    print(f"  hybrid_kinetics: {method_config['hybrid_kinetics']}")

    t_start = time.time()

    try:
        metrics_df, _, _, _, _, _ = estimates_to_metrics(
            est,
            fps=fps,
            include_event_based=True,
            include_heavy=True,
            event_method=method_config['event_method'],
            n_iter=method_config['n_iter'],
            correlation_method='spearman',
            hybrid_kinetics=method_config['hybrid_kinetics']
        )

        elapsed = time.time() - t_start

        # Save metrics
        output_file = method_output_dir / f"{label}_metrics.csv"
        metrics_df.to_csv(output_file, index=False)

        # Summary
        n_good = len(metrics_df[metrics_df['t_off'] > -1])
        success_rate = n_good / len(metrics_df) * 100

        print(f"\nCompleted in {elapsed:.1f}s ({len(metrics_df)/elapsed:.1f} neurons/sec)")
        print(f"  Good neurons: {n_good}/{len(metrics_df)} ({success_rate:.1f}%)")
        print(f"  Saved: {output_file}")

        return metrics_df, str(output_file)

    except Exception as e:
        print(f"\nERROR in {label}: {e}")
        raise


def run_all_methods(est, methods: List[Dict], fps: int, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Run all methods sequentially.

    Returns:
        Dictionary mapping method label to metrics DataFrame
    """
    results = {}

    print(f"\n{'='*80}")
    print(f"RUNNING ALL METHODS")
    print(f"{'='*80}")
    print(f"Session: {len(est.idx_components)} neurons")
    print(f"FPS: {fps}")
    print(f"Output directory: {output_dir}")
    print(f"Methods to run: {len(methods)}")

    overall_start = time.time()

    for i, method_config in enumerate(methods, 1):
        label = method_config['label']
        print(f"\n[{i}/{len(methods)}] Starting {label}...")

        try:
            metrics_df, _ = run_single_method(est, method_config, fps, output_dir)
            results[label] = metrics_df
        except Exception as e:
            print(f"FAILED: {label} - {e}")
            results[label] = None

    overall_elapsed = time.time() - overall_start

    print(f"\n{'='*80}")
    print(f"ALL METHODS COMPLETED")
    print(f"{'='*80}")
    print(f"Total time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    print(f"Successful: {sum(1 for v in results.values() if v is not None)}/{len(methods)}")

    return results


def load_all_metrics(output_dir: Path, methods: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Load all metrics CSVs from output directory."""
    results = {}

    for method_config in methods:
        label = method_config['label']
        csv_path = output_dir / label / f"{label}_metrics.csv"

        if csv_path.exists():
            results[label] = pd.read_csv(csv_path)
            print(f"Loaded: {label} ({len(results[label])} neurons)")
        else:
            print(f"WARNING: {csv_path} not found")
            results[label] = None

    return results


def analyze_good_neurons(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze good neurons (t_off > -1) for each method."""
    summary_data = []

    for label, df in dfs.items():
        if df is None:
            continue

        df_good = df[df['t_off'] > -1]

        summary_data.append({
            'Method': label,
            'Total': len(df),
            'Good': len(df_good),
            'Success_Rate_%': len(df_good) / len(df) * 100 if len(df) > 0 else 0,
            'R2_mean': df_good['r2_score'].mean() if len(df_good) > 0 else np.nan,
            'R2_median': df_good['r2_score'].median() if len(df_good) > 0 else np.nan,
            'Event_R2_mean': df_good['event_r2_score'].mean() if len(df_good) > 0 else np.nan,
            'Event_R2_median': df_good['event_r2_score'].median() if len(df_good) > 0 else np.nan,
            'Events_per_min_mean': df_good['events_per_min'].mean() if len(df_good) > 0 else np.nan,
            't_rise_mean': df_good['t_rise'].mean() if len(df_good) > 0 else np.nan,
            't_off_mean': df_good['t_off'].mean() if len(df_good) > 0 else np.nan,
        })

    return pd.DataFrame(summary_data)


def analyze_common_neurons(dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, int, Dict]:
    """
    Analyze neurons that passed all methods.

    Returns:
        summary_df: Summary statistics for common neurons
        n_common: Number of common neurons
        common_dfs: Dictionary of filtered DataFrames for common neurons
    """
    # Find common neurons (passed all methods)
    good_neuron_sets = {}
    for label, df in dfs.items():
        if df is not None:
            good_neuron_sets[label] = set(df[df['t_off'] > -1]['component_idx'].values)

    if len(good_neuron_sets) == 0:
        return pd.DataFrame(), 0, {}

    common_neurons = set.intersection(*good_neuron_sets.values())
    n_common = len(common_neurons)

    # Filter each dataset to common neurons
    common_dfs = {}
    for label, df in dfs.items():
        if df is not None:
            common_dfs[label] = df[df['component_idx'].isin(common_neurons)].sort_values('component_idx')

    # Summary statistics
    summary_data = []
    for label, df in common_dfs.items():
        summary_data.append({
            'Method': label,
            'N': len(df),
            'R2_mean': df['r2_score'].mean(),
            'R2_median': df['r2_score'].median(),
            'R2_std': df['r2_score'].std(),
            'Event_R2_mean': df['event_r2_score'].mean(),
            'Event_R2_median': df['event_r2_score'].median(),
            'Events_per_min': df['events_per_min'].mean(),
            't_rise': df['t_rise'].mean(),
            't_off': df['t_off'].mean(),
        })

    summary_df = pd.DataFrame(summary_data)

    return summary_df, n_common, common_dfs


def analyze_hybrid_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze hybrid method tier breakdown."""
    if 'kinetics_source' not in df.columns:
        return pd.DataFrame()

    # Filter to good neurons
    df_good = df[df['t_off'] > -1]

    tier_stats = df_good.groupby('kinetics_source').agg({
        'component_idx': 'count',
        'r2_score': ['mean', 'median', 'std'],
        'event_r2_score': ['mean', 'median'],
    }).round(4)

    tier_stats.columns = ['_'.join(col).strip() for col in tier_stats.columns.values]
    tier_stats = tier_stats.rename(columns={'component_idx_count': 'count'})
    tier_stats['percentage'] = tier_stats['count'] / len(df_good) * 100

    return tier_stats.reset_index()


def validate_hypotheses(good_summary: pd.DataFrame, common_dfs: Dict, n_common: int) -> Dict:
    """
    Validate the 3 hypotheses from NOF_H32_4D findings.

    Returns:
        Dictionary with hypothesis results (PASS/FAIL)
    """
    results = {}

    # Hypothesis 1: Detection Rate (Hybrid ~2x more neurons)
    if 'hybrid_iter3' in good_summary['Method'].values and 'wavelet_iter3' in good_summary['Method'].values:
        hybrid_rate = good_summary[good_summary['Method'] == 'hybrid_iter3']['Success_Rate_%'].values[0]
        wavelet_rate = good_summary[good_summary['Method'] == 'wavelet_iter3']['Success_Rate_%'].values[0]

        h1_pass = hybrid_rate > 55 and wavelet_rate < 40
        results['H1_detection_rate'] = {
            'pass': h1_pass,
            'hybrid_rate': hybrid_rate,
            'wavelet_rate': wavelet_rate,
            'description': f"Hybrid: {hybrid_rate:.1f}%, Wavelet: {wavelet_rate:.1f}%"
        }
    else:
        results['H1_detection_rate'] = {'pass': False, 'description': 'Missing data'}

    # Hypothesis 2: Common Neuron Quality (Hybrid R2 > Wavelet R2)
    if 'hybrid_iter3' in common_dfs and 'wavelet_iter3' in common_dfs:
        hybrid_r2 = common_dfs['hybrid_iter3']['r2_score'].mean()
        wavelet_r2 = common_dfs['wavelet_iter3']['r2_score'].mean()

        h2_pass = hybrid_r2 > wavelet_r2
        results['H2_common_quality'] = {
            'pass': h2_pass,
            'hybrid_r2': hybrid_r2,
            'wavelet_r2': wavelet_r2,
            'difference': hybrid_r2 - wavelet_r2,
            'description': f"Hybrid R2: {hybrid_r2:.4f}, Wavelet R2: {wavelet_r2:.4f}, Diff: {hybrid_r2-wavelet_r2:+.4f}"
        }
    else:
        results['H2_common_quality'] = {'pass': False, 'description': 'Missing data'}

    # Hypothesis 3: Tier Distribution
    if 'hybrid_iter3' in common_dfs:
        hybrid_df = common_dfs['hybrid_iter3']
        if 'kinetics_source' in hybrid_df.columns:
            tier_counts = hybrid_df['kinetics_source'].value_counts()
            total = len(hybrid_df)

            tier1_pct = tier_counts.get('wavelet_standard', 0) / total * 100 if total > 0 else 0
            tier2_pct = tier_counts.get('wavelet_relaxed', 0) / total * 100 if total > 0 else 0

            # More flexible range: tier1 20-70%, tier2 20-70%
            h3_pass = (20 <= tier1_pct <= 70) and (20 <= tier2_pct <= 70)
            results['H3_tier_distribution'] = {
                'pass': h3_pass,
                'tier1_pct': tier1_pct,
                'tier2_pct': tier2_pct,
                'description': f"Tier 1: {tier1_pct:.1f}%, Tier 2: {tier2_pct:.1f}%"
            }
        else:
            results['H3_tier_distribution'] = {'pass': False, 'description': 'No kinetics_source column'}
    else:
        results['H3_tier_distribution'] = {'pass': False, 'description': 'Missing data'}

    return results


def generate_markdown_report(
    session_name: str,
    fps: int,
    dfs: Dict[str, pd.DataFrame],
    good_summary: pd.DataFrame,
    common_summary: pd.DataFrame,
    n_common: int,
    common_dfs: Dict,
    tier_stats: pd.DataFrame,
    hypothesis_results: Dict,
    output_dir: Path
):
    """Generate comprehensive markdown validation report."""

    report_path = output_dir / f"VALIDATION_REPORT_{session_name}.md"

    with open(report_path, 'w') as f:
        f.write(f"# Hybrid Method Validation Report\n\n")
        f.write(f"**Session**: {session_name}\n")
        f.write(f"**FPS**: {fps}\n")
        f.write(f"**Total Neurons**: {len(dfs['hybrid_iter3']) if 'hybrid_iter3' in dfs and dfs['hybrid_iter3'] is not None else 'N/A'}\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("### Hypothesis Validation Results\n\n")

        for h_name, h_result in hypothesis_results.items():
            status = "[PASS]" if h_result.get('pass', False) else "[FAIL]"
            desc = h_result.get('description', 'No description')
            f.write(f"**{h_name}**: {status}\n")
            f.write(f"- {desc}\n\n")

        passes = sum(1 for h in hypothesis_results.values() if h.get('pass', False))
        total = len(hypothesis_results)
        f.write(f"**Overall**: {passes}/{total} hypotheses validated\n\n")

        # Good Neurons Analysis
        f.write("---\n\n")
        f.write("## Good Neurons Analysis (t_off > -1)\n\n")
        f.write("### Detection Rate by Method\n\n")
        f.write("```\n")
        f.write(good_summary.to_string(index=False))
        f.write("\n```\n\n")

        f.write("### Key Findings - Good Neurons\n\n")
        if 'hybrid_iter3' in good_summary['Method'].values:
            hybrid_row = good_summary[good_summary['Method'] == 'hybrid_iter3'].iloc[0]
            f.write(f"- **Hybrid n=3**: {hybrid_row['Good']}/{hybrid_row['Total']} neurons ({hybrid_row['Success_Rate_%']:.1f}%)\n")
            f.write(f"  - Mean R2: {hybrid_row['R2_mean']:.4f}\n")
            f.write(f"  - Mean Event R2: {hybrid_row['Event_R2_mean']:.4f}\n\n")

        if 'wavelet_iter3' in good_summary['Method'].values:
            wavelet_row = good_summary[good_summary['Method'] == 'wavelet_iter3'].iloc[0]
            f.write(f"- **Wavelet n=3**: {wavelet_row['Good']}/{wavelet_row['Total']} neurons ({wavelet_row['Success_Rate_%']:.1f}%)\n")
            f.write(f"  - Mean R2: {wavelet_row['R2_mean']:.4f}\n")
            f.write(f"  - Mean Event R2: {wavelet_row['Event_R2_mean']:.4f}\n\n")

        # Common Neurons Analysis
        f.write("---\n\n")
        f.write("## Common Neurons Analysis\n\n")
        f.write(f"**Neurons passing all 5 methods**: {n_common}\n\n")
        f.write("### Metrics Comparison (Common Neurons)\n\n")
        f.write("```\n")
        f.write(common_summary.to_string(index=False))
        f.write("\n```\n\n")

        # Pairwise Comparison
        if 'hybrid_iter3' in common_dfs and 'wavelet_iter3' in common_dfs:
            f.write("### Pairwise Comparison: Hybrid vs Wavelet n=3\n\n")

            hybrid_r2 = common_dfs['hybrid_iter3']['r2_score'].values
            wavelet_r2 = common_dfs['wavelet_iter3']['r2_score'].values

            diff_r2 = hybrid_r2 - wavelet_r2

            f.write("**R2 Score Comparison:**\n")
            f.write(f"- Mean difference: {diff_r2.mean():+.4f}\n")
            f.write(f"- Median difference: {np.median(diff_r2):+.4f}\n")
            f.write(f"- Hybrid better on: {(diff_r2 > 0).sum()} neurons ({(diff_r2 > 0).sum()/len(diff_r2)*100:.1f}%)\n")
            f.write(f"- Wavelet better on: {(diff_r2 < 0).sum()} neurons ({(diff_r2 < 0).sum()/len(diff_r2)*100:.1f}%)\n\n")

            hybrid_event_r2 = common_dfs['hybrid_iter3']['event_r2_score'].values
            wavelet_event_r2 = common_dfs['wavelet_iter3']['event_r2_score'].values
            diff_event_r2 = hybrid_event_r2 - wavelet_event_r2

            f.write("**Event R2 Score Comparison:**\n")
            f.write(f"- Mean difference: {diff_event_r2.mean():+.4f}\n")
            f.write(f"- Median difference: {np.median(diff_event_r2):+.4f}\n")
            f.write(f"- Hybrid better on: {(diff_event_r2 > 0).sum()} neurons ({(diff_event_r2 > 0).sum()/len(diff_event_r2)*100:.1f}%)\n")
            f.write(f"- Wavelet better on: {(diff_event_r2 < 0).sum()} neurons ({(diff_event_r2 < 0).sum()/len(diff_event_r2)*100:.1f}%)\n\n")

        # Hybrid Tier Breakdown
        f.write("---\n\n")
        f.write("## Hybrid Method Tier Breakdown\n\n")
        if not tier_stats.empty:
            f.write("```\n")
            f.write(tier_stats.to_string(index=False))
            f.write("\n```\n\n")
        else:
            f.write("No tier statistics available.\n\n")

        # Conclusion
        f.write("---\n\n")
        f.write("## Conclusion\n\n")

        if passes == total:
            f.write(f"[SUCCESS] All {total} hypotheses validated on {session_name}.\n\n")
            f.write("The hybrid n=3 method demonstrates:\n")
            f.write("1. Significantly higher detection rate (~2x standard methods)\n")
            f.write("2. Better reconstruction quality on common neurons\n")
            f.write("3. Expected tier distribution pattern\n\n")
            f.write("**Recommendation**: Hybrid n=3 method is validated for use as default.\n\n")
        elif passes >= 2:
            f.write(f"[PARTIAL SUCCESS] {passes}/{total} hypotheses validated on {session_name}.\n\n")
            f.write("The hybrid method shows strong performance but with some deviations from NOF_H32_4D findings.\n")
            f.write("Consider additional validation on more sessions before setting as default.\n\n")
        else:
            f.write(f"[WARNING] Only {passes}/{total} hypotheses validated on {session_name}.\n\n")
            f.write("Significant deviations from NOF_H32_4D findings detected.\n")
            f.write("Further investigation recommended before using hybrid as default.\n\n")

    print(f"\nValidation report saved: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Validate hybrid event detection method on specified session'
    )
    parser.add_argument(
        '--session',
        type=str,
        default='NOF_H09_4D',
        help='Session name (default: NOF_H09_4D)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    parser.add_argument(
        '--skip-computation',
        action='store_true',
        help='Skip method execution, only re-analyze existing results'
    )

    args = parser.parse_args()

    session_name = args.session
    fps = args.fps
    skip_computation = args.skip_computation

    # Setup output directory
    output_dir = Path(f'data/event_param_comparison_{session_name}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HYBRID METHOD VALIDATION")
    print("="*80)
    print(f"Session: {session_name}")
    print(f"FPS: {fps}")
    print(f"Output directory: {output_dir}")
    print(f"Skip computation: {skip_computation}")
    print()

    # Load or compute metrics
    if skip_computation:
        print("\nSkipping computation, loading existing results...")
        dfs = load_all_metrics(output_dir, METHODS)
    else:
        # Find and load estimates
        try:
            est_file = find_estimates_file(session_name)
            est = load_estimates(est_file)
        except Exception as e:
            print(f"ERROR: {e}")
            return 1

        # Run all methods
        dfs = run_all_methods(est, METHODS, fps, output_dir)

        # Save summary
        summary_file = output_dir / 'all_methods_summary.csv'
        summary_data = []
        for label, df in dfs.items():
            if df is not None:
                n_good = len(df[df['t_off'] > -1])
                summary_data.append({
                    'method': label,
                    'total_neurons': len(df),
                    'good_neurons': n_good,
                    'success_rate_%': n_good / len(df) * 100
                })
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False)
        print(f"\nSummary saved: {summary_file}")

    # Analysis phase
    print("\n" + "="*80)
    print("ANALYSIS PHASE")
    print("="*80)

    # Good neurons analysis
    print("\nAnalyzing good neurons...")
    good_summary = analyze_good_neurons(dfs)

    # Common neurons analysis
    print("Analyzing common neurons...")
    common_summary, n_common, common_dfs = analyze_common_neurons(dfs)

    # Hybrid tier analysis
    print("Analyzing hybrid tiers...")
    tier_stats = pd.DataFrame()
    if 'hybrid_iter3' in dfs and dfs['hybrid_iter3'] is not None:
        tier_stats = analyze_hybrid_tiers(dfs['hybrid_iter3'])

    # Validate hypotheses
    print("Validating hypotheses...")
    hypothesis_results = validate_hypotheses(good_summary, common_dfs, n_common)

    # Generate report
    print("\nGenerating validation report...")
    report_path = generate_markdown_report(
        session_name,
        fps,
        dfs,
        good_summary,
        common_summary,
        n_common,
        common_dfs,
        tier_stats,
        hypothesis_results,
        output_dir
    )

    # Print summary to console
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    for h_name, h_result in hypothesis_results.items():
        status = "[PASS]" if h_result.get('pass', False) else "[FAIL]"
        desc = h_result.get('description', 'No description')
        print(f"{h_name}: {status}")
        print(f"  {desc}")

    passes = sum(1 for h in hypothesis_results.values() if h.get('pass', False))
    total = len(hypothesis_results)
    print(f"\nOverall: {passes}/{total} hypotheses validated")
    print(f"\nFull report: {report_path}")
    print("="*80)

    return 0


if __name__ == '__main__':
    exit(main())
