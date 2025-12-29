"""
Compare impact of event detection method and n_iter on auto-inspection quality.

Tests 4 parameter combinations on a single NOF session:
- wavelet + n_iter=2
- wavelet + n_iter=3
- threshold + n_iter=2
- threshold + n_iter=3

Saves detailed results for each combination to investigate quality differences.
"""

import os
import sys
import pickle
import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')

from auto_inspector import (
    estimates_to_metrics,
    metrics_to_decision,
    implement_decision,
    compute_metrics,
    print_report
)
from ae_utils import save_validation_outputs

# Configuration
TEST_SESSION = 'NOF_H01_2D'  # Different session for validation
OUTPUT_BASE = Path('data/event_param_comparison_NOF_H01_2D')
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# Test parameters
PARAM_COMBINATIONS = [
    {'event_method': 'wavelet', 'n_iter': 2, 'label': 'wavelet_iter2'},
    {'event_method': 'wavelet', 'n_iter': 3, 'label': 'wavelet_iter3'},
    {'event_method': 'threshold', 'n_iter': 2, 'label': 'threshold_iter2'},
    {'event_method': 'threshold', 'n_iter': 3, 'label': 'threshold_iter3'},
]

# Validation deletion rules (same as capcan_validation)
VALIDATION_DELETION_RULES = [
    'area<=3',
    'circularity>1.7',
    'max_edge>1.45',
    'convexity>42'
]

# FPS for NOF sessions
FPS = 30


def load_estimates(filepath):
    """Load CaImAn estimates object from pickle file."""
    with open(filepath, 'rb') as f:
        estimates = pickle.load(f)
    return estimates


def run_auto_inspection(est, fps, event_method, n_iter, label):
    """
    Run full auto-inspection pipeline with specified parameters.

    Returns:
        dict: Results including metrics, decisions, and timing
    """
    print(f"\n{'='*80}")
    print(f"Running: {label}")
    print(f"  event_method: {event_method}")
    print(f"  n_iter: {n_iter}")
    print(f"{'='*80}")

    results = {
        'label': label,
        'event_method': event_method,
        'n_iter': n_iter
    }

    # Phase 1: Compute metrics
    print(f"\n[1/3] Computing metrics...")
    t_start = time.time()

    metrics_df, match_mtx, FCD, FBD, corner_info, _ = estimates_to_metrics(
        est,
        fps=fps,
        include_event_based=True,
        include_heavy=True,
        event_method=event_method,
        n_iter=n_iter,
        correlation_method='spearman'
    )

    t_metrics = time.time() - t_start
    results['time_metrics'] = t_metrics
    results['n_neurons'] = len(metrics_df)

    print(f"  Computed metrics for {len(metrics_df)} neurons in {t_metrics:.1f}s")

    # Phase 2: Make decisions
    print(f"\n[2/3] Running auto-inspection decisions...")
    t_start = time.time()

    decision_df = metrics_to_decision(
        metrics_df.copy(),
        match_mtx,
        FCD,
        FBD,
        deletion_rules=VALIDATION_DELETION_RULES,
        pxlthr_distance_boundary=5,
        d_snr_thr=10,
        enable_merge=True
    )

    t_decision = time.time() - t_start
    results['time_decision'] = t_decision

    # Count decisions
    n_delete = (decision_df['delete'] == 1).sum()
    n_keep = (decision_df['delete'] == 0).sum()
    n_merge_groups = len(decision_df[decision_df['merge'] != 0]['merge'].unique())

    results['n_delete'] = n_delete
    results['n_keep'] = n_keep
    results['n_merge_groups'] = n_merge_groups
    results['deletion_rate'] = n_delete / len(decision_df) if len(decision_df) > 0 else 0

    print(f"  Decisions: {n_keep} KEEP, {n_delete} DELETE ({results['deletion_rate']:.1%} deletion rate)")
    print(f"  Merge groups: {n_merge_groups}")

    # Phase 3: Implement decisions
    print(f"\n[3/3] Implementing decisions...")
    t_start = time.time()

    # Convert sparse S to dense if needed
    from scipy import sparse
    if sparse.issparse(est.S):
        est.S = est.S.toarray()

    decision_df['decision'] = decision_df['delete'].apply(
        lambda x: 'delete' if x == 1 else 'ok'
    )

    est_auto = implement_decision(est, decision_df)

    t_implement = time.time() - t_start
    results['time_implement'] = t_implement
    results['n_final'] = len(est_auto.idx_components)

    print(f"  Final neurons after implementation: {results['n_final']}")

    # Save results
    results['metrics_df'] = metrics_df
    results['decision_df'] = decision_df
    results['est_auto'] = est_auto
    results['corner_info'] = corner_info

    # Analyze rejection criteria
    rejection_criteria = {}
    for col in ['is_corner_artifact', 't_rise', 't_off', 'snr', 'r_score']:
        if col in decision_df.columns:
            deleted = decision_df[decision_df['delete'] == 1]
            if col == 'is_corner_artifact':
                rejection_criteria[col] = (deleted[col] == 1).sum()
            else:
                rejection_criteria[col] = deleted[f'{col}_reject'].sum() if f'{col}_reject' in deleted.columns else 0

    results['rejection_criteria'] = rejection_criteria

    print(f"\nRejection breakdown:")
    for criterion, count in rejection_criteria.items():
        if count > 0:
            print(f"  {criterion}: {count}")

    return results


def save_results(results, output_dir):
    """Save detailed results to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = results['label']

    # Save metrics
    metrics_file = output_dir / f'{label}_metrics.csv'
    results['metrics_df'].to_csv(metrics_file, index=False)
    print(f"  Saved metrics: {metrics_file}")

    # Save decisions
    decision_file = output_dir / f'{label}_decisions.csv'
    results['decision_df'].to_csv(decision_file, index=False)
    print(f"  Saved decisions: {decision_file}")

    # Save estimates
    est_file = output_dir / f'{label}_estimates.pkl'
    with open(est_file, 'wb') as f:
        pickle.dump(results['est_auto'], f)
    print(f"  Saved estimates: {est_file}")

    # Save summary
    summary = {
        'label': results['label'],
        'event_method': results['event_method'],
        'n_iter': results['n_iter'],
        'n_neurons': results['n_neurons'],
        'n_keep': results['n_keep'],
        'n_delete': results['n_delete'],
        'n_merge_groups': results['n_merge_groups'],
        'deletion_rate': results['deletion_rate'],
        'n_final': results['n_final'],
        'time_metrics': results['time_metrics'],
        'time_decision': results['time_decision'],
        'time_implement': results['time_implement'],
        **{f'reject_{k}': v for k, v in results['rejection_criteria'].items()}
    }

    summary_df = pd.DataFrame([summary])
    summary_file = output_dir / f'{label}_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"  Saved summary: {summary_file}")


def compare_metrics(all_results):
    """Compare key event-based metrics across all parameter combinations."""
    print(f"\n{'='*80}")
    print("METRIC COMPARISON ACROSS PARAMETERS")
    print(f"{'='*80}\n")

    # Key metrics to compare
    key_metrics = [
        'events_per_min',
        'events_fraction',
        'event_snr',
        'event_r2_score',
        't_rise',
        't_off',
        'r2_score',
        'nmae',
        'nrmse'
    ]

    comparison_data = []

    for metric in key_metrics:
        row = {'metric': metric}
        for results in all_results:
            label = results['label']
            metrics_df = results['metrics_df']
            if metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                row[label] = mean_val
            else:
                row[label] = np.nan
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)

    # Save comparison
    comparison_file = OUTPUT_BASE / 'metric_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved metric comparison: {comparison_file}")

    # Print comparison
    print("\nMean values across parameters:")
    print(comparison_df.to_string(index=False))

    return comparison_df


def compare_decisions(all_results):
    """Compare auto-inspection decisions across parameter combinations."""
    print(f"\n{'='*80}")
    print("DECISION COMPARISON ACROSS PARAMETERS")
    print(f"{'='*80}\n")

    summary_data = []

    for results in all_results:
        summary = {
            'label': results['label'],
            'event_method': results['event_method'],
            'n_iter': results['n_iter'],
            'n_neurons': results['n_neurons'],
            'n_keep': results['n_keep'],
            'n_delete': results['n_delete'],
            'deletion_rate': f"{results['deletion_rate']:.1%}",
            'n_merge_groups': results['n_merge_groups'],
            'n_final': results['n_final'],
            'time_total': results['time_metrics'] + results['time_decision'] + results['time_implement']
        }

        # Add rejection criteria
        for criterion, count in results['rejection_criteria'].items():
            summary[f'reject_{criterion}'] = count

        summary_data.append(summary)

    summary_df = pd.DataFrame(summary_data)

    # Save summary
    summary_file = OUTPUT_BASE / 'decision_comparison.csv'
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved decision comparison: {summary_file}")

    # Print summary
    print("\nDecision summary:")
    print(summary_df.to_string(index=False))

    return summary_df


def main():
    print(f"{'='*80}")
    print(f"EVENT PARAMETER COMPARISON")
    print(f"Session: {TEST_SESSION}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"{'='*80}\n")

    # Load session estimates
    print(f"Loading session {TEST_SESSION}...")

    # Find the init file (handles different naming patterns)
    import glob
    pattern = f'data/raw_compressed/{TEST_SESSION}*estimates.pickle'
    matches = glob.glob(pattern)

    if not matches:
        print(f"ERROR: No session file found matching: {pattern}")
        return

    init_file = matches[0]
    print(f"Found file: {init_file}")

    est = load_estimates(init_file)
    print(f"Loaded {len(est.idx_components)} neurons\n")

    # Run auto-inspection with each parameter combination
    all_results = []

    for params in PARAM_COMBINATIONS:
        try:
            # Use deepcopy to avoid modifying original estimates
            est_copy = copy.deepcopy(est)

            results = run_auto_inspection(
                est_copy,
                FPS,
                params['event_method'],
                params['n_iter'],
                params['label']
            )

            all_results.append(results)

            # Save individual results
            print(f"\nSaving results for {params['label']}...")
            save_results(results, OUTPUT_BASE / params['label'])

        except Exception as e:
            print(f"\nERROR running {params['label']}: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("GENERATING COMPARISONS")
        print(f"{'='*80}")

        metric_comparison = compare_metrics(all_results)
        decision_comparison = compare_decisions(all_results)

        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
        print(f"\nResults saved to: {OUTPUT_BASE}")
        print(f"\nKey files:")
        print(f"  - metric_comparison.csv: Mean values of key metrics")
        print(f"  - decision_comparison.csv: Decision statistics")
        print(f"  - [label]/ folders: Detailed results for each parameter combo")
    else:
        print("\nNo results collected!")


if __name__ == '__main__':
    main()
