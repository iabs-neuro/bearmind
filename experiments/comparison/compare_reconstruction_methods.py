"""
Compare Threshold vs Wavelet Reconstruction on Real Calcium Data
=================================================================

Adapted from driada example to work with CaImAn estimates.
Compares 4 parameter combinations:
- threshold n_iter=2
- threshold n_iter=3
- wavelet n_iter=2
- wavelet n_iter=3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from driada.experiment.neuron import Neuron
from scipy import sparse


# Configuration
BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'reconstruction_gallery_25'  # Separate folder for batch run
FPS = 30  # NOF session


def load_estimates(session_folder='wavelet_iter2'):
    """Load estimates from a specific session folder."""
    est_file = BASE_PATH / session_folder / f'{session_folder}_estimates.pkl'

    if est_file.exists():
        with open(est_file, 'rb') as f:
            est = pickle.load(f)
        print(f"Loaded {len(est.idx_components)} neurons from {session_folder}")
        return est
    else:
        print(f"WARNING: File not found: {est_file}")
        return None


def select_neurons(est, n_neurons=5, seed=42):
    """Select random neurons with activity for visualization."""
    np.random.seed(seed)

    # Find all neurons with activity
    active_neurons = []

    for i in range(len(est.idx_components)):
        comp_idx = est.idx_components[i]

        # Get calcium trace
        C = est.C[i, :]
        if sparse.issparse(C):
            C = C.toarray().flatten()

        # Check if has any activity (not flat line)
        if np.std(C) > 1.0:
            active_neurons.append(comp_idx)

    print(f"Found {len(active_neurons)} neurons with activity")

    # Randomly select n_neurons
    if len(active_neurons) > n_neurons:
        selected_indices = np.random.choice(active_neurons, size=n_neurons, replace=False).tolist()
    else:
        selected_indices = active_neurons[:n_neurons]

    print(f"Randomly selected {len(selected_indices)} neurons: {selected_indices}")
    return selected_indices


def reconstruct_with_params(calcium_trace, fps, method, n_iter):
    """Reconstruct spikes with specified method and n_iter.

    Returns reconstruction results dict.
    """
    # Ensure proper dtype and make copy
    calcium_trace = np.ascontiguousarray(calcium_trace, dtype=np.float64)

    # Create Neuron object (pass array directly, not TimeSeries)
    neuron = Neuron(
        cell_id=f'{method}_n{n_iter}',
        ca=calcium_trace,  # Pass numpy array directly
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
    else:  # wavelet
        neuron.reconstruct_spikes(
            method='wavelet',
            create_event_regions=True,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True
        )

    # Optimize kinetics with proper parameters - CAPTURE RETURN VALUE
    opt_result = neuron.optimize_kinetics(
        method='direct',
        fps=fps,
        update_reconstruction=True,
        detection_method=method,
        n_mad=4.0,
        iterative=True,
        n_iter=n_iter,
        adaptive_thresholds=True
    )

    # Check optimization success and log details
    kinetics_optimized = opt_result.get('optimized', False)
    kinetics_source = 'optimized' if kinetics_optimized else 'defaults'

    # Try relaxed parameters if standard optimization failed
    if not kinetics_optimized:
        print(f"    Initial optimization FAILED: {opt_result.get('error', 'Unknown error')}")
        print(f"      Measurements: {opt_result.get('n_events_used_rise', 0)} rise, "
              f"{opt_result.get('n_events_used_off', 0)} off (need >=5 each)")

        # Retry with relaxed parameters
        print(f"    Retrying with relaxed parameters...")
        opt_result = neuron.optimize_kinetics(
            method='direct',
            fps=fps,
            update_reconstruction=True,
            detection_method=method,
            n_mad=4.0,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True,
            min_events=3,        # Relaxed from 5
            min_r2=0.6          # Relaxed from 0.8
        )

        kinetics_optimized = opt_result.get('optimized', False)
        if kinetics_optimized:
            kinetics_source = 'relaxed'
            print(f"    Relaxed optimization SUCCEEDED!")
        else:
            print(f"    Relaxed optimization also FAILED")
            print(f"      Error: {opt_result.get('error', 'Unknown error')}")
            kinetics_source = 'defaults'

    # Log optimization results
    if kinetics_optimized:
        t_rise_std = opt_result.get('t_rise_std', 0.0)
        t_off_std = opt_result.get('t_off_std', 0.0)
        print(f"    Kinetics: t_rise={opt_result['t_rise']:.4f}s ± {t_rise_std:.4f}s, "
              f"t_off={opt_result['t_off']:.4f}s ± {t_off_std:.4f}s")
        print(f"    Measurements: {opt_result.get('n_events_used_rise', 0)} rise, "
              f"{opt_result.get('n_events_used_off', 0)} off")

    # Get results
    t_rise = neuron.t_rise / fps if neuron.t_rise else neuron.default_t_rise / fps
    t_off = neuron.t_off / fps if neuron.t_off else neuron.default_t_off / fps

    # Get reconstruction
    spike_data = neuron.asp.data if neuron.asp else neuron.sp.data
    reconstruction = Neuron.get_restored_calcium(
        spike_data,
        neuron.t_rise if neuron.t_rise else neuron.default_t_rise,
        neuron.t_off if neuron.t_off else neuron.default_t_off
    )

    # Count events
    if method == 'threshold':
        n_events = len(neuron.threshold_events) if neuron.threshold_events else 0
    else:
        n_events = len(neuron.wvt_ridges) if neuron.wvt_ridges else 0

    # Calculate R²
    signal = neuron.ca.data
    residuals = signal - reconstruction[:len(signal)]
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((signal - np.mean(signal)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'method': method,
        'n_iter': n_iter,
        'reconstruction': reconstruction[:len(signal)],
        'spikes': spike_data,
        'calcium': signal,
        't_rise': t_rise,
        't_off': t_off,
        'n_events': n_events,
        'r2': r2,
        # Kinetics optimization info
        'kinetics_optimized': kinetics_optimized,
        'kinetics_source': kinetics_source,
        'kinetics_error': opt_result.get('error', None),
        'n_events_used_rise': opt_result.get('n_events_used_rise', 0),
        'n_events_used_off': opt_result.get('n_events_used_off', 0),
        't_rise_std': opt_result.get('t_rise_std', 0.0),
        't_off_std': opt_result.get('t_off_std', 0.0)
    }


def plot_neuron_comparison(comp_idx, est, output_file):
    """Create 4-panel comparison plot for one neuron."""

    # Find neuron position
    neuron_pos = np.where(est.idx_components == comp_idx)[0][0]

    # Extract calcium trace
    C_raw = est.C[neuron_pos, :]
    if sparse.issparse(C_raw):
        C_raw = C_raw.toarray().flatten()
    C_raw = np.asarray(C_raw, dtype=np.float64)

    print(f"\nProcessing neuron {comp_idx}...")

    # Run all 4 reconstructions
    configs = [
        ('threshold', 2),
        ('threshold', 3),
        ('wavelet', 2),
        ('wavelet', 3)
    ]

    results = []
    for method, n_iter in configs:
        print(f"  {method} n_iter={n_iter}...")
        result = reconstruct_with_params(C_raw, FPS, method, n_iter)
        results.append(result)

    # Create plot
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))

    time_axis = np.arange(len(C_raw)) / FPS

    for idx, (result, ax) in enumerate(zip(results, axes)):
        # Normalize to [0, 1] for comparison
        calcium_norm = (result['calcium'] - result['calcium'].min()) / (result['calcium'].max() - result['calcium'].min())
        recon_norm = (result['reconstruction'] - result['reconstruction'].min()) / (result['reconstruction'].max() - result['reconstruction'].min())

        # Plot
        ax.plot(time_axis, calcium_norm, 'k-', linewidth=1.5, alpha=0.7, label='Calcium')
        ax.plot(time_axis, recon_norm, 'r-', linewidth=2, alpha=0.8, label='Reconstruction')

        # Mark events
        spike_times = time_axis[result['spikes'] > 0]
        if len(spike_times) > 0:
            ax.scatter(spike_times, np.ones(len(spike_times)) * 1.05,
                      marker='v', s=80, color='blue', label=f'Events (n={len(spike_times)})',
                      zorder=10, alpha=0.6)

        # Styling with kinetics source annotation
        title = f"{result['method'].capitalize()} n_iter={result['n_iter']}"
        title += f" | Events: {result['n_events']} | R²: {result['r2']:.4f}"
        title += f"\nt_rise: {result['t_rise']:.3f}s, t_off: {result['t_off']:.2f}s"

        # Add kinetics source indicator with color
        kinetics_source = result['kinetics_source']
        if kinetics_source == 'optimized':
            source_text = " [OPTIMIZED]"
            title_color = 'darkgreen'
        elif kinetics_source == 'relaxed':
            source_text = " [RELAXED]"
            title_color = 'darkorange'
        else:  # defaults
            source_text = " [DEFAULTS - FAILED]"
            title_color = 'darkred'

        title += source_text

        ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)
        ax.set_ylabel('Normalized Fluorescence', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_axis[0], time_axis[-1])
        ax.set_ylim(-0.1, 1.15)

        if idx == 3:
            ax.set_xlabel('Time (seconds)', fontsize=11)

    plt.suptitle(f'Neuron {comp_idx} - Reconstruction Method Comparison\nNOF_H32_4D Session',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file}")

    return results


def generate_optimization_summary(all_results, output_file):
    """Generate CSV summary of optimization results across all neurons."""

    summary_rows = []

    for comp_idx, results in all_results.items():
        for result in results:
            summary_rows.append({
                'neuron_id': comp_idx,
                'method': result['method'],
                'n_iter': result['n_iter'],
                'n_events_detected': result['n_events'],
                'kinetics_optimized': result['kinetics_optimized'],
                'kinetics_source': result['kinetics_source'],
                't_rise': result['t_rise'],
                't_off': result['t_off'],
                't_rise_std': result['t_rise_std'],
                't_off_std': result['t_off_std'],
                'n_measurements_rise': result['n_events_used_rise'],
                'n_measurements_off': result['n_events_used_off'],
                'optimization_error': result['kinetics_error'] if result['kinetics_error'] else '',
                'reconstruction_r2': result['r2']
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_file, index=False)

    print(f"\nSaved optimization summary: {output_file}")

    # Print statistics
    print("\n" + "="*80)
    print("OPTIMIZATION SUCCESS STATISTICS")
    print("="*80)

    for method in ['threshold', 'wavelet']:
        method_data = summary_df[summary_df['method'] == method]

        total = len(method_data)
        optimized = (method_data['kinetics_source'] == 'optimized').sum()
        relaxed = (method_data['kinetics_source'] == 'relaxed').sum()
        defaults = (method_data['kinetics_source'] == 'defaults').sum()

        print(f"\n{method.upper()}:")
        print(f"  Total: {total}")
        print(f"  Optimized: {optimized} ({optimized/total*100:.1f}%)")
        print(f"  Relaxed: {relaxed} ({relaxed/total*100:.1f}%)")
        print(f"  Defaults (failed): {defaults} ({defaults/total*100:.1f}%)")

        # R² comparison by kinetics source
        for source in ['optimized', 'relaxed', 'defaults']:
            source_data = method_data[method_data['kinetics_source'] == source]
            if len(source_data) > 0:
                mean_r2 = source_data['reconstruction_r2'].mean()
                print(f"    Mean R² ({source}): {mean_r2:.4f}")


def main():
    print("="*80)
    print("RECONSTRUCTION METHOD COMPARISON ON REAL DATA")
    print("="*80)

    # Load estimates (we just need one to get the calcium traces)
    print("\nLoading estimates...")
    est = load_estimates('wavelet_iter2')

    if est is None:
        print("ERROR: Could not load estimates!")
        return

    # Select neurons
    print("\nSelecting neurons...")
    selected_neurons = select_neurons(est, n_neurons=25, seed=789)  # Generate 25 random neurons

    # Process each neuron
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for comp_idx in selected_neurons:
        output_file = OUTPUT_PATH / f'neuron_{comp_idx}_method_comparison.png'
        results = plot_neuron_comparison(comp_idx, est, output_file)
        all_results[comp_idx] = results

    # Generate optimization summary CSV
    summary_file = OUTPUT_PATH / 'optimization_summary.csv'
    generate_optimization_summary(all_results, summary_file)

    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"\nGenerated {len(selected_neurons)} comparison plots")
    print("Each plot shows:")
    print("  - Threshold n_iter=2")
    print("  - Threshold n_iter=3")
    print("  - Wavelet n_iter=2")
    print("  - Wavelet n_iter=3")
    print("\nFiles saved:")
    print(f"  - {len(selected_neurons)} neuron comparison PNGs")
    print(f"  - optimization_summary.csv (detailed metrics)")
    print("\nKinetics optimization with fallback strategies:")
    print("  [GREEN] Optimized - standard optimization succeeded")
    print("  [ORANGE] Relaxed - succeeded with relaxed parameters")
    print("  [RED] Defaults - optimization failed, using defaults")


if __name__ == '__main__':
    main()
