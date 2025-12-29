"""
Investigate Neuron 471 - Why Wavelet Fails While Threshold Succeeds
====================================================================

This neuron shows extreme divergence:
- Threshold: 34-54 events detected, kinetics optimized successfully
- Wavelet: Only 2 events detected, kinetics optimization fails

Goal: Understand the signal characteristics causing this divergence.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from driada.experiment.neuron import Neuron
from scipy import sparse


BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'neuron_471_investigation'
FPS = 30


def load_estimates(session_folder='wavelet_iter2'):
    """Load estimates from a specific session folder."""
    est_file = BASE_PATH / session_folder / f'{session_folder}_estimates.pkl'

    if est_file.exists():
        with open(est_file, 'rb') as f:
            est = pickle.load(f)
        return est
    else:
        print(f"ERROR: File not found: {est_file}")
        return None


def analyze_signal_characteristics(calcium_trace, fps):
    """Analyze signal properties that might affect detection methods."""

    # Basic statistics
    mean_val = np.mean(calcium_trace)
    std_val = np.std(calcium_trace)
    snr = mean_val / std_val if std_val > 0 else 0

    # Noise characteristics
    diff = np.diff(calcium_trace)
    noise_estimate = np.median(np.abs(diff)) / 0.6745  # MAD estimator

    # Event amplitude characteristics (simple peak detection)
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(calcium_trace, height=mean_val + 2*std_val, distance=int(fps*0.5))

    if len(peaks) > 0:
        peak_heights = properties['peak_heights']
        mean_amplitude = np.mean(peak_heights - mean_val)
        amplitude_std = np.std(peak_heights - mean_val)
    else:
        mean_amplitude = 0
        amplitude_std = 0

    # Baseline drift
    from scipy.ndimage import median_filter
    baseline = median_filter(calcium_trace, size=int(fps*10))
    drift_range = np.max(baseline) - np.min(baseline)

    return {
        'mean': mean_val,
        'std': std_val,
        'snr': snr,
        'noise_mad': noise_estimate,
        'n_peaks_simple': len(peaks),
        'mean_amplitude': mean_amplitude,
        'amplitude_std': amplitude_std,
        'baseline_drift_range': drift_range
    }


def reconstruct_with_diagnostics(calcium_trace, fps, method, n_iter):
    """Reconstruct with detailed diagnostic output."""

    calcium_trace = np.ascontiguousarray(calcium_trace, dtype=np.float64)

    neuron = Neuron(
        cell_id=f'{method}_n{n_iter}',
        ca=calcium_trace,
        sp=None,
        fps=fps
    )

    # Reconstruct spikes
    if method == 'threshold':
        neuron.reconstruct_spikes(
            method='threshold',
            n_mad=4.0,
            min_duration_frames=2,
            create_event_regions=True,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True
        )
        events = neuron.threshold_events if neuron.threshold_events else []
        n_events = len(events)
    else:  # wavelet
        neuron.reconstruct_spikes(
            method='wavelet',
            create_event_regions=True,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True
        )
        events = neuron.wvt_ridges if neuron.wvt_ridges else []
        n_events = len(events)

    # Get spike data for visualization
    spike_data = neuron.asp.data if neuron.asp else neuron.sp.data

    # Get event details
    event_details = []
    if events:
        for evt in events:
            if hasattr(evt, 'start') and hasattr(evt, 'end'):
                event_details.append({
                    'start_frame': evt.start,
                    'end_frame': evt.end,
                    'duration_frames': evt.end - evt.start,
                    'peak_frame': evt.peak if hasattr(evt, 'peak') else None,
                    'amplitude': evt.amplitude if hasattr(evt, 'amplitude') else None
                })

    return {
        'neuron': neuron,
        'n_events': n_events,
        'spike_data': spike_data,
        'event_details': event_details
    }


def plot_comprehensive_analysis(comp_idx, est):
    """Create comprehensive diagnostic plot."""

    # Find neuron
    neuron_pos = np.where(est.idx_components == comp_idx)[0][0]

    # Extract calcium trace
    C_raw = est.C[neuron_pos, :]
    if sparse.issparse(C_raw):
        C_raw = C_raw.toarray().flatten()
    C_raw = np.asarray(C_raw, dtype=np.float64)

    print(f"Analyzing neuron {comp_idx}...")
    print(f"Trace length: {len(C_raw)} frames ({len(C_raw)/FPS:.1f} seconds)")

    # Analyze signal characteristics
    signal_stats = analyze_signal_characteristics(C_raw, FPS)

    print("\nSignal Characteristics:")
    for key, value in signal_stats.items():
        print(f"  {key}: {value:.4f}")

    # Run all 4 reconstructions with diagnostics
    configs = [
        ('threshold', 2),
        ('threshold', 3),
        ('wavelet', 2),
        ('wavelet', 3)
    ]

    results = []
    for method, n_iter in configs:
        print(f"\n{method.upper()} n_iter={n_iter}:")
        result = reconstruct_with_diagnostics(C_raw, FPS, method, n_iter)
        results.append(result)

        print(f"  Events detected: {result['n_events']}")
        if result['event_details']:
            durations = [evt['duration_frames'] for evt in result['event_details']]
            print(f"  Event durations: min={min(durations)}, max={max(durations)}, mean={np.mean(durations):.1f} frames")
            if result['event_details'][0]['amplitude'] is not None:
                amplitudes = [evt['amplitude'] for evt in result['event_details'] if evt['amplitude'] is not None]
                if amplitudes:
                    print(f"  Event amplitudes: min={min(amplitudes):.2f}, max={max(amplitudes):.2f}, mean={np.mean(amplitudes):.2f}")

    # Create comprehensive plot
    fig = plt.figure(figsize=(20, 16))

    # Top panel: Full trace overview with all detections
    ax_overview = plt.subplot(5, 1, 1)
    time_axis = np.arange(len(C_raw)) / FPS

    ax_overview.plot(time_axis, C_raw, 'k-', linewidth=1, alpha=0.7, label='Calcium')

    # Mark threshold detections
    threshold_2_spikes = results[0]['spike_data'] > 0
    threshold_3_spikes = results[1]['spike_data'] > 0

    if np.any(threshold_2_spikes):
        ax_overview.scatter(time_axis[threshold_2_spikes],
                          np.ones(np.sum(threshold_2_spikes)) * np.max(C_raw) * 0.95,
                          marker='v', s=50, color='red', alpha=0.6, label=f'Threshold n=2 ({results[0]["n_events"]} events)')

    if np.any(threshold_3_spikes):
        ax_overview.scatter(time_axis[threshold_3_spikes],
                          np.ones(np.sum(threshold_3_spikes)) * np.max(C_raw) * 0.90,
                          marker='v', s=50, color='darkred', alpha=0.6, label=f'Threshold n=3 ({results[1]["n_events"]} events)')

    # Mark wavelet detections
    wavelet_2_spikes = results[2]['spike_data'] > 0
    wavelet_3_spikes = results[3]['spike_data'] > 0

    if np.any(wavelet_2_spikes):
        ax_overview.scatter(time_axis[wavelet_2_spikes],
                          np.ones(np.sum(wavelet_2_spikes)) * np.max(C_raw) * 0.85,
                          marker='o', s=100, color='blue', alpha=0.8, label=f'Wavelet n=2 ({results[2]["n_events"]} events)')

    if np.any(wavelet_3_spikes):
        ax_overview.scatter(time_axis[wavelet_3_spikes],
                          np.ones(np.sum(wavelet_3_spikes)) * np.max(C_raw) * 0.80,
                          marker='o', s=100, color='darkblue', alpha=0.8, label=f'Wavelet n=3 ({results[3]["n_events"]} events)')

    ax_overview.set_title(f'Neuron {comp_idx} - Full Trace with Event Detections', fontsize=14, fontweight='bold')
    ax_overview.set_ylabel('Fluorescence (a.u.)', fontsize=11)
    ax_overview.legend(loc='upper right', fontsize=9)
    ax_overview.grid(True, alpha=0.3)

    # Bottom 4 panels: Individual reconstructions
    for idx, (result, (method, n_iter)) in enumerate(zip(results, configs)):
        ax = plt.subplot(5, 1, idx + 2)

        # Get reconstruction
        neuron = result['neuron']
        t_rise = neuron.t_rise if neuron.t_rise else neuron.default_t_rise
        t_off = neuron.t_off if neuron.t_off else neuron.default_t_off

        reconstruction = Neuron.get_restored_calcium(
            result['spike_data'],
            t_rise,
            t_off
        )

        # Normalize
        calcium_norm = (C_raw - C_raw.min()) / (C_raw.max() - C_raw.min())
        recon_norm = (reconstruction[:len(C_raw)] - reconstruction[:len(C_raw)].min()) / (reconstruction[:len(C_raw)].max() - reconstruction[:len(C_raw)].min())

        # Plot
        ax.plot(time_axis, calcium_norm, 'k-', linewidth=1.5, alpha=0.7, label='Calcium')
        ax.plot(time_axis, recon_norm, 'r-', linewidth=2, alpha=0.8, label='Reconstruction')

        # Mark events
        spike_times = time_axis[result['spike_data'] > 0]
        if len(spike_times) > 0:
            ax.scatter(spike_times, np.ones(len(spike_times)) * 1.05,
                      marker='v', s=80, color='blue', label=f'Events (n={len(spike_times)})',
                      zorder=10, alpha=0.6)

        # Calculate R²
        residuals = calcium_norm - recon_norm
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((calcium_norm - np.mean(calcium_norm)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Title with kinetics info
        t_rise_sec = t_rise / FPS
        t_off_sec = t_off / FPS

        title = f"{method.capitalize()} n_iter={n_iter} | Events: {result['n_events']} | R²: {r2:.4f}"
        title += f"\nt_rise: {t_rise_sec:.3f}s, t_off: {t_off_sec:.2f}s"

        # Color code based on success
        if result['n_events'] >= 5:
            title_color = 'darkgreen'
        elif result['n_events'] >= 2:
            title_color = 'darkorange'
        else:
            title_color = 'darkred'

        ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)
        ax.set_ylabel('Normalized Fluorescence', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(-0.1, 1.15)

        if idx == 3:
            ax.set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle(f'Neuron {comp_idx} Investigation: Why Wavelet Fails to Detect Events\n' +
                f'Signal: mean={signal_stats["mean"]:.2f}, std={signal_stats["std"]:.2f}, ' +
                f'SNR={signal_stats["snr"]:.2f}, noise(MAD)={signal_stats["noise_mad"]:.4f}',
                fontsize=14, fontweight='bold', y=0.997)

    plt.tight_layout()

    return fig, signal_stats, results


def main():
    print("="*80)
    print("NEURON 471 INVESTIGATION")
    print("="*80)

    try:
        # Load estimates
        print("\nLoading estimates...")
        est = load_estimates('wavelet_iter2')
        if est is None:
            print("ERROR: Failed to load estimates")
            return

        print(f"Successfully loaded estimates with {len(est.idx_components)} neurons")
    except Exception as e:
        print(f"ERROR loading estimates: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # Create output directory
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Created output directory: {OUTPUT_PATH}")

        # Analyze neuron 471
        print("\nAnalyzing neuron 471...")
        fig, signal_stats, results = plot_comprehensive_analysis(471, est)

        # Save plot
        output_file = OUTPUT_PATH / 'neuron_471_comprehensive_analysis.png'
        print(f"Saving plot to: {output_file}")
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSaved comprehensive analysis: {output_file}")
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return

    try:
        # Generate detailed report
        report_file = OUTPUT_PATH / 'neuron_471_analysis_report.txt'
        print(f"\nGenerating report: {report_file}")

        with open(report_file, 'w') as f:
            f.write("NEURON 471 INVESTIGATION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write("PROBLEM STATEMENT:\n")
            f.write("-" * 80 + "\n")
            f.write("Threshold methods successfully optimize kinetics (34-54 events detected)\n")
            f.write("Wavelet methods fail to optimize kinetics (only 2 events detected)\n")
            f.write("This is counterintuitive since wavelet is typically more sensitive\n\n")

            f.write("SIGNAL CHARACTERISTICS:\n")
            f.write("-" * 80 + "\n")
            for key, value in signal_stats.items():
                f.write(f"{key:25s}: {value:.4f}\n")
            f.write("\n")

            f.write("DETECTION RESULTS:\n")
            f.write("-" * 80 + "\n")
            configs = [('threshold', 2), ('threshold', 3), ('wavelet', 2), ('wavelet', 3)]
            for (method, n_iter), result in zip(configs, results):
                f.write(f"\n{method.upper()} n_iter={n_iter}:\n")
                f.write(f"  Events detected: {result['n_events']}\n")

                if result['event_details']:
                    durations = [evt['duration_frames'] for evt in result['event_details']]
                    f.write(f"  Event duration (frames): min={min(durations)}, max={max(durations)}, mean={np.mean(durations):.1f}\n")

                    if result['event_details'][0]['amplitude'] is not None:
                        amplitudes = [evt['amplitude'] for evt in result['event_details'] if evt['amplitude'] is not None]
                        if amplitudes:
                            f.write(f"  Event amplitude: min={min(amplitudes):.2f}, max={max(amplitudes):.2f}, mean={np.mean(amplitudes):.2f}\n")

            f.write("\n\nHYPOTHESIS:\n")
            f.write("-" * 80 + "\n")
            f.write("Wavelet detection is TOO SELECTIVE for this neuron's signal characteristics.\n")
            f.write("Possible reasons:\n")
            f.write("1. Signal has many small, brief transients that threshold captures but wavelet filters\n")
            f.write("2. Wavelet may be looking for specific temporal patterns that don't match this neuron\n")
            f.write("3. Baseline characteristics or noise properties confuse wavelet ridge detection\n")
            f.write("4. Event shapes don't match typical calcium transient templates used by wavelet\n\n")

            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 80 + "\n")
            f.write("1. Examine first 60 seconds of trace in detail to see what wavelet is missing\n")
            f.write("2. Check wavelet parameters (scales, threshold criteria)\n")
            f.write("3. Compare with other neurons that wavelet succeeds on\n")
            f.write("4. Consider signal preprocessing (detrending, smoothing) effects\n")

        print(f"Saved detailed report: {report_file}")
    except Exception as e:
        print(f"ERROR generating report: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("INVESTIGATION COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
