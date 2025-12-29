"""
Smart Hybrid Approach Test
==========================
Strategy: Use wavelet n_iter=3 kinetics when optimization succeeds,
fall back to threshold kinetics only when wavelet optimization fails.

Optimization "failure" criteria:
- R2 < threshold (e.g., 0.5)
- Kinetics at default values (didn't converge)
- Very few events detected
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from driada.experiment.neuron import Neuron
from scipy import sparse


BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'smart_hybrid'
FPS = 30

# Thresholds for determining "failure"
R2_THRESHOLD = 0.5
MIN_EVENTS = 3
DEFAULT_T_RISE = 0.25  # seconds
DEFAULT_T_OFF = 2.0    # seconds


def load_estimates(session_folder='wavelet_iter2'):
    """Load estimates."""
    est_file = BASE_PATH / session_folder / f'{session_folder}_estimates.pkl'
    with open(est_file, 'rb') as f:
        est = pickle.load(f)
    return est


def optimize_kinetics_safe(neuron, fps, method, n_iter):
    """Run kinetics optimization with fallback to relaxed parameters."""

    opt_result = neuron.optimize_kinetics(
        method='direct',
        fps=fps,
        update_reconstruction=True,
        detection_method=method,
        n_mad=4.0 if method == 'threshold' else None,
        iterative=True,
        n_iter=n_iter,
        adaptive_thresholds=True
    )

    kinetics_optimized = opt_result.get('optimized', False)
    kinetics_source = 'optimized' if kinetics_optimized else 'defaults'

    # Try relaxed if failed
    if not kinetics_optimized:
        opt_result = neuron.optimize_kinetics(
            method='direct',
            fps=fps,
            update_reconstruction=True,
            detection_method=method,
            n_mad=4.0 if method == 'threshold' else None,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True,
            min_events=3,
            min_r2=0.6
        )

        kinetics_optimized = opt_result.get('optimized', False)
        if kinetics_optimized:
            kinetics_source = 'relaxed'

    return opt_result, kinetics_source


def check_optimization_success(n_events, r2, t_rise_sec, t_off_sec):
    """
    Check if wavelet optimization was successful.

    Returns:
        (success: bool, reasons: list of failure reasons)
    """
    reasons = []

    # Check number of events
    if n_events < MIN_EVENTS:
        reasons.append(f'few_events ({n_events}<{MIN_EVENTS})')

    # Check R2
    if r2 < R2_THRESHOLD:
        reasons.append(f'low_r2 ({r2:.3f}<{R2_THRESHOLD})')

    # Check if kinetics are at defaults (didn't optimize)
    if abs(t_rise_sec - DEFAULT_T_RISE) < 0.02 and abs(t_off_sec - DEFAULT_T_OFF) < 0.1:
        reasons.append('default_kinetics')

    success = len(reasons) == 0
    return success, reasons


def process_neuron_smart_hybrid(calcium_trace, fps):
    """
    Smart hybrid approach:
    1. Run wavelet n_iter=3
    2. If optimization succeeds, use wavelet result
    3. If optimization fails, use threshold kinetics with wavelet events
    """
    calcium_trace = np.ascontiguousarray(calcium_trace, dtype=np.float64)

    # STEP 1: Run wavelet n_iter=3
    neuron_wavelet = Neuron(
        cell_id='wavelet_n3',
        ca=calcium_trace,
        sp=None,
        fps=fps
    )

    neuron_wavelet.reconstruct_spikes(
        method='wavelet',
        create_event_regions=True,
        iterative=True,
        n_iter=3,
        adaptive_thresholds=True
    )

    n_events_wavelet = len(neuron_wavelet.wvt_ridges) if neuron_wavelet.wvt_ridges else 0

    # Optimize kinetics for wavelet
    opt_result_wvt, kinetics_source_wvt = optimize_kinetics_safe(neuron_wavelet, fps, 'wavelet', 3)

    # Get wavelet kinetics
    t_rise_wvt = neuron_wavelet.t_rise if neuron_wavelet.t_rise else neuron_wavelet.default_t_rise
    t_off_wvt = neuron_wavelet.t_off if neuron_wavelet.t_off else neuron_wavelet.default_t_off

    # Calculate wavelet R2
    spike_wavelet = neuron_wavelet.asp.data if neuron_wavelet.asp else neuron_wavelet.sp.data
    reconstruction_wvt = Neuron.get_restored_calcium(spike_wavelet, t_rise_wvt, t_off_wvt)[:len(calcium_trace)]

    residuals = calcium_trace - reconstruction_wvt
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((calcium_trace - np.mean(calcium_trace)) ** 2)
    r2_wavelet = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    # Check if wavelet optimization succeeded
    t_rise_sec = t_rise_wvt / fps
    t_off_sec = t_off_wvt / fps

    wavelet_success, failure_reasons = check_optimization_success(
        n_events_wavelet, r2_wavelet, t_rise_sec, t_off_sec
    )

    result = {
        'wavelet_events': n_events_wavelet,
        'wavelet_r2': r2_wavelet,
        'wavelet_t_rise': t_rise_sec,
        'wavelet_t_off': t_off_sec,
        'wavelet_kinetics_source': kinetics_source_wvt,
    }

    if wavelet_success:
        # Use wavelet result as-is
        result['decision'] = 'KEEP_WAVELET'
        result['failure_reasons'] = []
        result['smart_hybrid_r2'] = r2_wavelet
        result['smart_hybrid_t_rise'] = t_rise_sec
        result['smart_hybrid_t_off'] = t_off_sec
        result['threshold_events'] = None
        result['threshold_r2'] = None
        result['threshold_t_rise'] = None
        result['threshold_t_off'] = None
    else:
        # STEP 2: Get threshold kinetics
        neuron_threshold = Neuron(
            cell_id='threshold_n3',
            ca=calcium_trace,
            sp=None,
            fps=fps
        )

        neuron_threshold.reconstruct_spikes(
            method='threshold',
            n_mad=4.0,
            min_duration_frames=2,
            create_event_regions=True,
            iterative=True,
            n_iter=3,
            adaptive_thresholds=True
        )

        n_events_thresh = len(neuron_threshold.threshold_events) if neuron_threshold.threshold_events else 0

        # Optimize kinetics for threshold
        opt_result_thr, kinetics_source_thr = optimize_kinetics_safe(neuron_threshold, fps, 'threshold', 3)

        # Get threshold kinetics
        t_rise_thr = neuron_threshold.t_rise if neuron_threshold.t_rise else neuron_threshold.default_t_rise
        t_off_thr = neuron_threshold.t_off if neuron_threshold.t_off else neuron_threshold.default_t_off

        # Calculate threshold R2 for reference
        spike_thresh = neuron_threshold.asp.data if neuron_threshold.asp else neuron_threshold.sp.data
        reconstruction_thr = Neuron.get_restored_calcium(spike_thresh, t_rise_thr, t_off_thr)[:len(calcium_trace)]

        residuals_thr = calcium_trace - reconstruction_thr
        ss_res_thr = np.sum(residuals_thr ** 2)
        r2_threshold = 1 - (ss_res_thr / ss_tot) if ss_tot > 0 else 0

        result['threshold_events'] = n_events_thresh
        result['threshold_r2'] = r2_threshold
        result['threshold_t_rise'] = t_rise_thr / fps
        result['threshold_t_off'] = t_off_thr / fps

        # STEP 3: Reconstruct wavelet events with threshold kinetics
        if n_events_wavelet > 0:
            reconstruction_hybrid = Neuron.get_restored_calcium(spike_wavelet, t_rise_thr, t_off_thr)[:len(calcium_trace)]

            residuals_hybrid = calcium_trace - reconstruction_hybrid
            ss_res_hybrid = np.sum(residuals_hybrid ** 2)
            r2_hybrid = 1 - (ss_res_hybrid / ss_tot) if ss_tot > 0 else 0
        else:
            r2_hybrid = 0.0

        result['decision'] = 'FALLBACK_HYBRID'
        result['failure_reasons'] = failure_reasons
        result['smart_hybrid_r2'] = r2_hybrid
        result['smart_hybrid_t_rise'] = t_rise_thr / fps
        result['smart_hybrid_t_off'] = t_off_thr / fps

    return result


def main():
    print("=" * 80)
    print("SMART HYBRID APPROACH TEST")
    print("=" * 80)
    print("\nStrategy:")
    print("  1. Try wavelet n_iter=3 optimization")
    print("  2. If optimization succeeds (R2 >= 0.5, events >= 3, kinetics not default), keep wavelet")
    print("  3. If optimization fails, use threshold n_iter=3 kinetics with wavelet events")
    print()

    # Load estimates
    print("Loading estimates...")
    est = load_estimates('wavelet_iter2')
    print(f"Loaded {len(est.idx_components)} neurons")

    # Load the 25 neurons from previous gallery
    summary_csv = BASE_PATH / 'reconstruction_gallery_25' / 'optimization_summary.csv'
    df_neurons = pd.read_csv(summary_csv)
    neurons_to_test = sorted(df_neurons['neuron_id'].unique())

    print(f"\nTesting on {len(neurons_to_test)} neurons")
    print("-" * 80)

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    results = []

    for i, comp_idx in enumerate(neurons_to_test):
        # Find neuron
        neuron_pos = np.where(est.idx_components == comp_idx)[0][0]

        # Extract calcium trace
        C_raw = est.C[neuron_pos, :]
        if sparse.issparse(C_raw):
            C_raw = C_raw.toarray().flatten()
        C_raw = np.asarray(C_raw, dtype=np.float64)

        print(f"\n[{i+1}/25] Neuron {comp_idx}:")

        result = process_neuron_smart_hybrid(C_raw, FPS)
        result['neuron_id'] = comp_idx

        print(f"  Wavelet: {result['wavelet_events']} events, R2={result['wavelet_r2']:.4f}")
        print(f"    Kinetics: t_rise={result['wavelet_t_rise']:.3f}s, t_off={result['wavelet_t_off']:.3f}s")

        if result['decision'] == 'KEEP_WAVELET':
            print(f"  [KEEP] Wavelet optimization succeeded")
            print(f"  Smart Hybrid R2: {result['smart_hybrid_r2']:.4f}")
        else:
            print(f"  [FALLBACK] Reasons: {', '.join(result['failure_reasons'])}")
            print(f"  Threshold: {result['threshold_events']} events, R2={result['threshold_r2']:.4f}")
            print(f"    Kinetics: t_rise={result['threshold_t_rise']:.3f}s, t_off={result['threshold_t_off']:.3f}s")
            print(f"  Smart Hybrid R2: {result['smart_hybrid_r2']:.4f}")

        results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SMART HYBRID RESULTS SUMMARY")
    print("=" * 80)

    n_kept = (df['decision'] == 'KEEP_WAVELET').sum()
    n_fallback = (df['decision'] == 'FALLBACK_HYBRID').sum()

    print(f"\nDecision breakdown:")
    print(f"  Kept wavelet kinetics: {n_kept} neurons ({100*n_kept/len(df):.1f}%)")
    print(f"  Fell back to hybrid:   {n_fallback} neurons ({100*n_fallback/len(df):.1f}%)")

    # Performance comparison
    print(f"\nPerformance comparison:")
    print(f"  Mean wavelet R2:       {df['wavelet_r2'].mean():.4f}")
    print(f"  Mean smart hybrid R2:  {df['smart_hybrid_r2'].mean():.4f}")

    improvement = df['smart_hybrid_r2'].mean() - df['wavelet_r2'].mean()
    print(f"  Mean improvement:      {improvement:+.4f}")

    # For fallback neurons only
    if n_fallback > 0:
        fallback_df = df[df['decision'] == 'FALLBACK_HYBRID']
        print(f"\nFallback neurons analysis ({n_fallback} neurons):")
        print(f"  Mean wavelet R2 (before): {fallback_df['wavelet_r2'].mean():.4f}")
        print(f"  Mean hybrid R2 (after):   {fallback_df['smart_hybrid_r2'].mean():.4f}")
        fallback_improvement = fallback_df['smart_hybrid_r2'].mean() - fallback_df['wavelet_r2'].mean()
        print(f"  Mean improvement:         {fallback_improvement:+.4f}")

    # Compare against best standard method (from previous analysis)
    # Load previous comparison data
    prev_csv = BASE_PATH / 'hybrid_comparison' / 'method_comparison_with_hybrid.csv'
    if prev_csv.exists():
        df_prev = pd.read_csv(prev_csv)

        # Merge to get best standard R2
        df = df.merge(df_prev[['neuron_id', 'wavelet_n3_r2', 'threshold_n3_r2']], on='neuron_id', how='left')
        df['best_standard_r2'] = df[['wavelet_n3_r2', 'threshold_n3_r2']].max(axis=1)
        df['improvement_vs_best'] = df['smart_hybrid_r2'] - df['best_standard_r2']

        print(f"\nComparison vs Best Standard Method:")
        print(f"  Mean best standard R2:   {df['best_standard_r2'].mean():.4f}")
        print(f"  Mean smart hybrid R2:    {df['smart_hybrid_r2'].mean():.4f}")
        print(f"  Mean improvement:        {df['improvement_vs_best'].mean():+.4f}")

        wins = (df['improvement_vs_best'] > 0.001).sum()
        ties = ((df['improvement_vs_best'] >= -0.001) & (df['improvement_vs_best'] <= 0.001)).sum()
        losses = (df['improvement_vs_best'] < -0.001).sum()

        print(f"\nWin/Loss vs Best Standard:")
        print(f"  Smart hybrid wins:  {wins} ({100*wins/len(df):.1f}%)")
        print(f"  Smart hybrid ties:  {ties} ({100*ties/len(df):.1f}%)")
        print(f"  Smart hybrid loses: {losses} ({100*losses/len(df):.1f}%)")

    # Detailed results
    print("\n" + "-" * 80)
    print("DETAILED RESULTS BY NEURON")
    print("-" * 80)
    print(f"{'Neuron':<8} {'Wavelet':<10} {'SmartHybrid':<12} {'Decision':<15} {'vs Best':<10}")
    print("-" * 80)

    for _, row in df.iterrows():
        vs_best = row.get('improvement_vs_best', 0)
        print(f"{row['neuron_id']:<8} {row['wavelet_r2']:<10.4f} {row['smart_hybrid_r2']:<12.4f} "
              f"{row['decision']:<15} {vs_best:+.4f}")

    # Save results
    df.to_csv(OUTPUT_PATH / 'smart_hybrid_results.csv', index=False)

    # Create visualization
    create_visualization(df, n_kept, n_fallback)

    print(f"\nResults saved to {OUTPUT_PATH}")
    print("=" * 80)


def create_visualization(df, n_kept, n_fallback):
    """Create summary visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: R2 comparison by neuron
    ax1 = axes[0, 0]
    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width/2, df['wavelet_r2'], width, label='Wavelet n_iter=3', alpha=0.8, color='steelblue')
    ax1.bar(x + width/2, df['smart_hybrid_r2'], width, label='Smart Hybrid', alpha=0.8, color='gold')

    # Mark fallback neurons
    for i, (_, row) in enumerate(df.iterrows()):
        if row['decision'] == 'FALLBACK_HYBRID':
            ax1.axvline(i, color='red', linestyle='--', alpha=0.3, linewidth=1)

    ax1.set_xlabel('Neuron')
    ax1.set_ylabel('R2')
    ax1.set_title('R2 Comparison: Wavelet vs Smart Hybrid\n(Red lines = fallback neurons)')
    ax1.legend()
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels([str(n) for n in df['neuron_id'].values[::2]], rotation=45)

    # Plot 2: Improvement distribution
    ax2 = axes[0, 1]
    improvement = df['smart_hybrid_r2'] - df['wavelet_r2']
    colors = ['green' if x > 0 else 'red' for x in improvement]
    ax2.bar(range(len(df)), improvement, color=colors, alpha=0.7)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(improvement.mean(), color='blue', linestyle='--',
                label=f'Mean: {improvement.mean():+.4f}')
    ax2.set_xlabel('Neuron')
    ax2.set_ylabel('R2 Improvement (Smart Hybrid - Wavelet)')
    ax2.set_title('Smart Hybrid Improvement Over Wavelet')
    ax2.legend()

    # Plot 3: Decision breakdown pie
    ax3 = axes[1, 0]
    decision_counts = [n_kept, n_fallback]
    decision_labels = [f'Keep Wavelet\n({n_kept})', f'Fallback to Hybrid\n({n_fallback})']
    colors_pie = ['steelblue', 'gold']
    ax3.pie(decision_counts, labels=decision_labels, colors=colors_pie, autopct='%1.1f%%',
            startangle=90, explode=(0, 0.05))
    ax3.set_title('Smart Hybrid Decision Distribution')

    # Plot 4: Scatter - wavelet vs smart hybrid
    ax4 = axes[1, 1]
    kept_mask = df['decision'] == 'KEEP_WAVELET'
    fallback_mask = df['decision'] == 'FALLBACK_HYBRID'

    ax4.scatter(df.loc[kept_mask, 'wavelet_r2'], df.loc[kept_mask, 'smart_hybrid_r2'],
                c='steelblue', label=f'Kept wavelet ({kept_mask.sum()})', alpha=0.7, s=80)
    ax4.scatter(df.loc[fallback_mask, 'wavelet_r2'], df.loc[fallback_mask, 'smart_hybrid_r2'],
                c='gold', label=f'Fallback ({fallback_mask.sum()})', alpha=0.7, s=80, marker='s')

    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax4.set_xlabel('Wavelet n_iter=3 R2')
    ax4.set_ylabel('Smart Hybrid R2')
    ax4.set_title('Wavelet vs Smart Hybrid\n(Points above line = improvement)')
    ax4.legend()
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / 'smart_hybrid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved visualization: {OUTPUT_PATH / 'smart_hybrid_comparison.png'}")


if __name__ == '__main__':
    main()
