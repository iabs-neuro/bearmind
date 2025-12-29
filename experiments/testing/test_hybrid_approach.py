"""
Test Hybrid Approach: Threshold Kinetics + Wavelet Events
==========================================================

Strategy:
1. Use threshold to detect many events and optimize kinetics
2. Use those optimized kinetics with wavelet-detected events
3. Compare R² to see if we get best of both worlds
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
import pickle

from driada.experiment.neuron import Neuron


BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'neuron_471_investigation'
FPS = 30


def load_estimates(session_folder='wavelet_iter2'):
    """Load estimates."""
    est_file = BASE_PATH / session_folder / f'{session_folder}_estimates.pkl'
    with open(est_file, 'rb') as f:
        est = pickle.load(f)
    return est


def test_hybrid_approach(comp_idx, est):
    """Test using threshold kinetics with wavelet events."""

    # Find neuron
    neuron_pos = np.where(est.idx_components == comp_idx)[0][0]

    # Extract calcium trace
    C_raw = est.C[neuron_pos, :]
    if sparse.issparse(C_raw):
        C_raw = C_raw.toarray().flatten()
    C_raw = np.ascontiguousarray(C_raw, dtype=np.float64)

    print(f"\n{'='*80}")
    print(f"HYBRID APPROACH TEST - NEURON {comp_idx}")
    print(f"{'='*80}\n")

    results = {}

    # ============================================================================
    # STEP 1: Threshold detection + kinetics optimization
    # ============================================================================
    print("STEP 1: Threshold Detection + Kinetics Optimization")
    print("-" * 80)

    neuron_thresh = Neuron(
        cell_id='threshold_for_kinetics',
        ca=C_raw,
        sp=None,
        fps=FPS
    )

    neuron_thresh.reconstruct_spikes(
        method='threshold',
        n_mad=4.0,
        min_duration_frames=2,
        create_event_regions=True,
        iterative=True,
        n_iter=3,
        adaptive_thresholds=True
    )

    opt_result = neuron_thresh.optimize_kinetics(
        method='direct',
        fps=FPS,
        update_reconstruction=True,
        detection_method='threshold',
        n_mad=4.0,
        iterative=True,
        n_iter=3,
        adaptive_thresholds=True
    )

    kinetics_optimized = opt_result.get('optimized', False)

    if not kinetics_optimized:
        print("  Trying relaxed parameters...")
        opt_result = neuron_thresh.optimize_kinetics(
            method='direct',
            fps=FPS,
            update_reconstruction=True,
            detection_method='threshold',
            n_mad=4.0,
            iterative=True,
            n_iter=3,
            adaptive_thresholds=True,
            min_events=3,
            min_r2=0.6
        )
        kinetics_optimized = opt_result.get('optimized', False)

    # Get optimized kinetics (in frames)
    t_rise_opt = neuron_thresh.t_rise if neuron_thresh.t_rise else neuron_thresh.default_t_rise
    t_off_opt = neuron_thresh.t_off if neuron_thresh.t_off else neuron_thresh.default_t_off

    t_rise_sec = t_rise_opt / FPS
    t_off_sec = t_off_opt / FPS

    print(f"\n  Threshold detected: {len(neuron_thresh.threshold_events)} events")
    print(f"  Kinetics optimization: {'SUCCESS' if kinetics_optimized else 'FAILED'}")
    print(f"  Optimized kinetics: t_rise={t_rise_sec:.3f}s, t_off={t_off_sec:.2f}s")

    # Store threshold result
    spike_thresh = neuron_thresh.asp.data if neuron_thresh.asp else neuron_thresh.sp.data
    recon_thresh = Neuron.get_restored_calcium(spike_thresh, t_rise_opt, t_off_opt)[:len(C_raw)]

    residuals = C_raw - recon_thresh
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((C_raw - np.mean(C_raw)) ** 2)
    r2_thresh = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    results['threshold_standard'] = {
        'n_events': len(neuron_thresh.threshold_events),
        'kinetics': (t_rise_sec, t_off_sec),
        'r2': r2_thresh,
        'reconstruction': recon_thresh,
        'spikes': spike_thresh
    }

    print(f"  Threshold R²: {r2_thresh:.4f}")

    # ============================================================================
    # STEP 2: Wavelet detection (standard approach)
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 2: Wavelet Detection (Standard - for comparison)")
    print("-" * 80)

    neuron_wavelet = Neuron(
        cell_id='wavelet_standard',
        ca=C_raw,
        sp=None,
        fps=FPS
    )

    neuron_wavelet.reconstruct_spikes(
        method='wavelet',
        create_event_regions=True,
        iterative=True,
        n_iter=3,
        adaptive_thresholds=True
    )

    # Try to optimize (we know it will fail)
    opt_result_wvt = neuron_wavelet.optimize_kinetics(
        method='direct',
        fps=FPS,
        update_reconstruction=True,
        detection_method='wavelet',
        iterative=True,
        n_iter=3,
        adaptive_thresholds=True,
        min_events=3,
        min_r2=0.6
    )

    wvt_kinetics_optimized = opt_result_wvt.get('optimized', False)

    print(f"\n  Wavelet detected: {len(neuron_wavelet.wvt_ridges)} events")
    print(f"  Kinetics optimization: {'SUCCESS' if wvt_kinetics_optimized else 'FAILED'}")

    # Get wavelet kinetics (likely defaults)
    t_rise_wvt = neuron_wavelet.t_rise if neuron_wavelet.t_rise else neuron_wavelet.default_t_rise
    t_off_wvt = neuron_wavelet.t_off if neuron_wavelet.t_off else neuron_wavelet.default_t_off

    spike_wavelet = neuron_wavelet.asp.data if neuron_wavelet.asp else neuron_wavelet.sp.data
    recon_wavelet = Neuron.get_restored_calcium(spike_wavelet, t_rise_wvt, t_off_wvt)[:len(C_raw)]

    residuals = C_raw - recon_wavelet
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((C_raw - np.mean(C_raw)) ** 2)
    r2_wavelet = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    results['wavelet_standard'] = {
        'n_events': len(neuron_wavelet.wvt_ridges),
        'kinetics': (t_rise_wvt / FPS, t_off_wvt / FPS),
        'r2': r2_wavelet,
        'reconstruction': recon_wavelet,
        'spikes': spike_wavelet
    }

    print(f"  Wavelet kinetics: t_rise={t_rise_wvt/FPS:.3f}s, t_off={t_off_wvt/FPS:.2f}s")
    print(f"  Wavelet R²: {r2_wavelet:.4f}")

    # ============================================================================
    # STEP 3: HYBRID - Wavelet events + Threshold kinetics
    # ============================================================================
    print(f"\n{'='*80}")
    print("STEP 3: HYBRID APPROACH - Wavelet Events + Threshold Kinetics")
    print("-" * 80)

    # Use wavelet-detected spikes with threshold-optimized kinetics
    recon_hybrid = Neuron.get_restored_calcium(spike_wavelet, t_rise_opt, t_off_opt)[:len(C_raw)]

    residuals = C_raw - recon_hybrid
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((C_raw - np.mean(C_raw)) ** 2)
    r2_hybrid = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    results['hybrid'] = {
        'n_events': len(neuron_wavelet.wvt_ridges),
        'kinetics': (t_rise_sec, t_off_sec),
        'r2': r2_hybrid,
        'reconstruction': recon_hybrid,
        'spikes': spike_wavelet
    }

    print(f"\n  Using:")
    print(f"    - Wavelet events: {len(neuron_wavelet.wvt_ridges)} events")
    print(f"    - Threshold kinetics: t_rise={t_rise_sec:.3f}s, t_off={t_off_sec:.2f}s")
    print(f"  Hybrid R²: {r2_hybrid:.4f}")

    # ============================================================================
    # COMPARISON
    # ============================================================================
    print(f"\n{'='*80}")
    print("RESULTS COMPARISON")
    print("=" * 80)
    print(f"\n{'Method':<25} {'Events':<10} {'t_rise (s)':<12} {'t_off (s)':<12} {'R²':<10}")
    print("-" * 80)
    print(f"{'Threshold (standard)':<25} {results['threshold_standard']['n_events']:<10} "
          f"{results['threshold_standard']['kinetics'][0]:<12.3f} "
          f"{results['threshold_standard']['kinetics'][1]:<12.2f} "
          f"{results['threshold_standard']['r2']:<10.4f}")
    print(f"{'Wavelet (standard)':<25} {results['wavelet_standard']['n_events']:<10} "
          f"{results['wavelet_standard']['kinetics'][0]:<12.3f} "
          f"{results['wavelet_standard']['kinetics'][1]:<12.2f} "
          f"{results['wavelet_standard']['r2']:<10.4f}")
    print(f"{'HYBRID (Wvt + Thr kin)':<25} {results['hybrid']['n_events']:<10} "
          f"{results['hybrid']['kinetics'][0]:<12.3f} "
          f"{results['hybrid']['kinetics'][1]:<12.2f} "
          f"{results['hybrid']['r2']:<10.4f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Compare hybrid to alternatives
    r2_improvement_vs_wavelet = results['hybrid']['r2'] - results['wavelet_standard']['r2']
    r2_vs_threshold = results['hybrid']['r2'] - results['threshold_standard']['r2']

    print(f"\nHybrid R² change vs Wavelet standard: {r2_improvement_vs_wavelet:+.4f}")
    print(f"Hybrid R² change vs Threshold standard: {r2_vs_threshold:+.4f}")

    if r2_improvement_vs_wavelet > 0.01:
        print("\n[SUCCESS] Hybrid approach improves over standard wavelet!")
    elif abs(r2_improvement_vs_wavelet) < 0.01:
        print("\n[NEUTRAL] Hybrid approach has minimal impact on wavelet performance")
    else:
        print("\n[WORSE] Hybrid approach degrades wavelet performance")

    if results['hybrid']['r2'] > results['threshold_standard']['r2']:
        print("[BREAKTHROUGH] Hybrid beats threshold! Use this approach!")
    elif abs(r2_vs_threshold) < 0.01:
        print("[NEUTRAL] Hybrid matches threshold performance")
    else:
        print(f"[EXPECTED] Threshold still wins (detects {results['threshold_standard']['n_events']}x more events)")

    # Plot comparison
    plot_hybrid_comparison(C_raw, results, comp_idx)

    return results


def plot_hybrid_comparison(calcium, results, comp_idx):
    """Plot 3-way comparison."""

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    time_axis = np.arange(len(calcium)) / FPS

    configs = [
        ('threshold_standard', 'Threshold (standard)', 'darkgreen'),
        ('wavelet_standard', 'Wavelet (standard)', 'darkred'),
        ('hybrid', 'HYBRID (Wavelet events + Threshold kinetics)', 'darkblue')
    ]

    for idx, (key, title, color) in enumerate(configs):
        ax = axes[idx]
        result = results[key]

        # Normalize
        ca_norm = (calcium - calcium.min()) / (calcium.max() - calcium.min())
        recon_norm = (result['reconstruction'] - result['reconstruction'].min()) / \
                     (result['reconstruction'].max() - result['reconstruction'].min())

        # Plot
        ax.plot(time_axis, ca_norm, 'k-', linewidth=1.5, alpha=0.7, label='Calcium')
        ax.plot(time_axis, recon_norm, 'r-', linewidth=2, alpha=0.8, label='Reconstruction')

        # Mark events
        spike_times = time_axis[result['spikes'] > 0]
        if len(spike_times) > 0:
            ax.scatter(spike_times, np.ones(len(spike_times)) * 1.05,
                      marker='v', s=80, color='blue', label=f'Events (n={len(spike_times)})',
                      zorder=10, alpha=0.6)

        # Title
        t_rise, t_off = result['kinetics']
        full_title = f"{title}\n"
        full_title += f"Events: {result['n_events']} | "
        full_title += f"t_rise: {t_rise:.3f}s, t_off: {t_off:.2f}s | "
        full_title += f"R²: {result['r2']:.4f}"

        ax.set_title(full_title, fontsize=11, fontweight='bold', color=color)
        ax.set_ylabel('Normalized Fluorescence', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(-0.1, 1.15)

        if idx == 2:
            ax.set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle(f'Neuron {comp_idx} - Hybrid Approach Test',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_file = OUTPUT_PATH / f'neuron_{comp_idx}_hybrid_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved comparison plot: {output_file}")


def main():
    est = load_estimates('wavelet_iter2')
    results = test_hybrid_approach(471, est)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if results['hybrid']['r2'] > results['threshold_standard']['r2']:
        print("\nHybrid approach is SUPERIOR! Use threshold kinetics with wavelet events.")
    elif results['hybrid']['r2'] > results['wavelet_standard']['r2'] + 0.01:
        print("\nHybrid approach IMPROVES wavelet but doesn't beat threshold.")
        print("Conclusion: Threshold kinetics help, but wavelet's low event count is limiting.")
    else:
        print("\nHybrid approach provides NO BENEFIT.")
        print("Reason: Wavelet only detects 2 events - kinetics don't matter much.")
        print("Conclusion: Use standard threshold approach.")


if __name__ == '__main__':
    main()
