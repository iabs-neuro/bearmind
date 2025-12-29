"""
Full Auto-Inspection: Smart Hybrid Comparison
==============================================
Uses same approach as compare_event_params.py - runs auto_inspector
with smart hybrid method.
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
)

# Configuration
BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'full_autoinspect'
FPS = 30

# Validation deletion rules (same as compare_event_params)
VALIDATION_DELETION_RULES = [
    'area<=3',
    'circularity>1.7',
    'max_edge>1.45',
    'convexity>42'
]


def load_estimates(filepath):
    """Load CaImAn estimates object from pickle file."""
    with open(filepath, 'rb') as f:
        estimates = pickle.load(f)
    return estimates


def run_auto_inspection(est, fps, event_method, n_iter, label):
    """
    Run full auto-inspection pipeline with specified parameters.
    Same as compare_event_params.py
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
    print(f"\n[1/2] Computing metrics...")
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
    print(f"\n[2/2] Running auto-inspection decisions...")
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

    results['n_delete'] = n_delete
    results['n_keep'] = n_keep
    results['deletion_rate'] = n_delete / len(decision_df) if len(decision_df) > 0 else 0

    print(f"  Decisions: {n_keep} KEEP, {n_delete} DELETE ({results['deletion_rate']:.1%} deletion rate)")

    # Save results
    results['metrics_df'] = metrics_df
    results['decision_df'] = decision_df

    return results


def main():
    print("=" * 80)
    print("FULL AUTO-INSPECTION: METHOD COMPARISON")
    print("=" * 80)

    # Find raw estimates file
    import glob
    pattern = 'data/raw_compressed/NOF_H32_4D*estimates.pickle'
    matches = glob.glob(pattern)

    if not matches:
        print(f"ERROR: No session file found matching: {pattern}")
        return

    init_file = matches[0]
    print(f"\nLoading: {init_file}")

    est = load_estimates(init_file)
    print(f"Loaded {len(est.idx_components)} neurons")

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Run both methods
    all_results = []

    for method, n_iter, label in [
        ('wavelet', 3, 'wavelet_n3'),
        ('threshold', 3, 'threshold_n3'),
    ]:
        try:
            est_copy = copy.deepcopy(est)
            results = run_auto_inspection(est_copy, FPS, method, n_iter, label)
            all_results.append(results)

            # Save metrics
            metrics_file = OUTPUT_PATH / f'{label}_metrics.csv'
            results['metrics_df'].to_csv(metrics_file, index=False)
            print(f"  Saved: {metrics_file}")

        except Exception as e:
            print(f"\nERROR running {label}: {e}")
            import traceback
            traceback.print_exc()

    # Compare results
    if len(all_results) == 2:
        compare_methods(all_results)


def compare_methods(all_results):
    """Compare wavelet vs threshold results."""
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)

    wvt = all_results[0]
    thr = all_results[1]

    wvt_df = wvt['metrics_df']
    thr_df = thr['metrics_df']

    # Merge on component_idx
    merged = wvt_df[['component_idx', 'r2_score', 'events_per_min', 't_rise', 't_off']].merge(
        thr_df[['component_idx', 'r2_score', 'events_per_min', 't_rise', 't_off']],
        on='component_idx',
        suffixes=('_wvt', '_thr')
    )

    print(f"\nMatched neurons: {len(merged)}")

    # Compare R2
    print(f"\nMean R2:")
    print(f"  Wavelet:   {merged['r2_score_wvt'].mean():.4f}")
    print(f"  Threshold: {merged['r2_score_thr'].mean():.4f}")

    # Winner counts
    merged['winner'] = merged.apply(
        lambda r: 'wavelet' if r['r2_score_wvt'] > r['r2_score_thr'] else
                  ('threshold' if r['r2_score_thr'] > r['r2_score_wvt'] else 'tie'),
        axis=1
    )

    n_wvt_wins = (merged['winner'] == 'wavelet').sum()
    n_thr_wins = (merged['winner'] == 'threshold').sum()
    n_ties = (merged['winner'] == 'tie').sum()

    print(f"\nWinner distribution:")
    print(f"  Wavelet wins:   {n_wvt_wins} ({100*n_wvt_wins/len(merged):.1f}%)")
    print(f"  Threshold wins: {n_thr_wins} ({100*n_thr_wins/len(merged):.1f}%)")
    print(f"  Ties:           {n_ties} ({100*n_ties/len(merged):.1f}%)")

    # Events comparison
    print(f"\nMean events/min:")
    print(f"  Wavelet:   {merged['events_per_min_wvt'].mean():.2f}")
    print(f"  Threshold: {merged['events_per_min_thr'].mean():.2f}")

    # Save comparison
    merged.to_csv(OUTPUT_PATH / 'method_comparison.csv', index=False)
    print(f"\nSaved: {OUTPUT_PATH / 'method_comparison.csv'}")

    # Summary
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    if n_wvt_wins > n_thr_wins:
        print(f"\nWavelet n_iter=3 recommended ({100*n_wvt_wins/len(merged):.1f}% win rate)")
    else:
        print(f"\nThreshold n_iter=3 recommended ({100*n_thr_wins/len(merged):.1f}% win rate)")


if __name__ == '__main__':
    main()
