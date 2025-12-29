"""
Run Hybrid Auto-Inspection
===========================
Uses estimates_to_metrics with hybrid_kinetics=True (cascading kinetics optimization)
"""

import pickle
import glob
from pathlib import Path

from auto_inspector import estimates_to_metrics

# Configuration
BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'hybrid_iter3'
FPS = 30

def main():
    print("=" * 80)
    print("HYBRID AUTO-INSPECTION (Cascading Kinetics)")
    print("=" * 80)

    # Find raw estimates file
    pattern = 'data/raw_compressed/NOF_H32_4D*estimates.pickle'
    matches = glob.glob(pattern)

    if not matches:
        print(f"ERROR: No session file found matching: {pattern}")
        return

    init_file = matches[0]
    print(f"\nLoading: {init_file}")

    with open(init_file, 'rb') as f:
        est = pickle.load(f)
    print(f"Loaded {len(est.idx_components)} neurons")

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Run auto-inspection with hybrid kinetics (default now True)
    print(f"\nRunning hybrid kinetics auto-inspection...")
    print("  event_method: wavelet")
    print("  n_iter: 3")
    print("  hybrid_kinetics: True (cascading: wavelet_std -> wavelet_relaxed -> thr_std -> thr_relaxed -> defaults)")

    metrics_df, match_mtx, FCD, FBD, corner_info, _ = estimates_to_metrics(
        est,
        fps=FPS,
        include_event_based=True,
        include_heavy=True,
        event_method='wavelet',
        n_iter=3,
        correlation_method='spearman',
        hybrid_kinetics=True
    )

    # Save results
    output_file = OUTPUT_PATH / 'hybrid_iter3_metrics.csv'
    metrics_df.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("KINETICS SOURCE BREAKDOWN")
    print("=" * 80)

    if 'kinetics_source' in metrics_df.columns:
        source_counts = metrics_df['kinetics_source'].value_counts()
        total = len(metrics_df)
        for source, count in source_counts.items():
            print(f"  {source}: {count} ({100*count/total:.1f}%)")
    else:
        print("  kinetics_source column not found")

    print("\n" + "=" * 80)
    print("R2 BY KINETICS SOURCE")
    print("=" * 80)

    if 'kinetics_source' in metrics_df.columns and 'r2_score' in metrics_df.columns:
        for source in ['wavelet_standard', 'wavelet_relaxed', 'threshold_standard', 'threshold_relaxed', 'defaults', 'unknown']:
            subset = metrics_df[metrics_df['kinetics_source'] == source]
            if len(subset) > 0:
                print(f"  {source}: mean R2 = {subset['r2_score'].mean():.4f} (n={len(subset)})")

        print(f"\nOverall mean R2: {metrics_df['r2_score'].mean():.4f}")
    else:
        print("  Required columns not found")


if __name__ == '__main__':
    main()
