"""
Comprehensive Method Comparison Including Hybrid Approach
==========================================================

Compares 5 approaches:
1. Threshold n_iter=2 (standard)
2. Threshold n_iter=3 (standard)
3. Wavelet n_iter=2 (standard)
4. Wavelet n_iter=3 (standard)
5. HYBRID: Threshold n_iter=3 kinetics + Wavelet n_iter=3 events

Tests on all 25 neurons to see if hybrid is systematically better.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from driada.experiment.neuron import Neuron
from scipy import sparse


BASE_PATH = Path('data/event_param_comparison')
OUTPUT_PATH = BASE_PATH / 'hybrid_comparison'
FPS = 30


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


def reconstruct_standard(calcium_trace, fps, method, n_iter):
    """Standard reconstruction (same method for detection and reconstruction)."""

    calcium_trace = np.ascontiguousarray(calcium_trace, dtype=np.float64)

    neuron = Neuron(
        cell_id=f'{method}_n{n_iter}',
        ca=calcium_trace,
        sp=None,
        fps=fps
    )

    # Detect spikes
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
        n_events = len(neuron.threshold_events) if neuron.threshold_events else 0
    else:  # wavelet
        neuron.reconstruct_spikes(
            method='wavelet',
            create_event_regions=True,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True
        )
        n_events = len(neuron.wvt_ridges) if neuron.wvt_ridges else 0

    # Optimize kinetics
    opt_result, kinetics_source = optimize_kinetics_safe(neuron, fps, method, n_iter)

    # Get kinetics
    t_rise = neuron.t_rise if neuron.t_rise else neuron.default_t_rise
    t_off = neuron.t_off if neuron.t_off else neuron.default_t_off

    # Get reconstruction
    spike_data = neuron.asp.data if neuron.asp else neuron.sp.data
    reconstruction = Neuron.get_restored_calcium(spike_data, t_rise, t_off)[:len(calcium_trace)]

    # Calculate R²
    residuals = calcium_trace - reconstruction
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((calcium_trace - np.mean(calcium_trace)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'method': f'{method}_n{n_iter}',
        'n_events': n_events,
        'kinetics_source': kinetics_source,
        't_rise': t_rise / fps,
        't_off': t_off / fps,
        'r2': r2,
        'reconstruction': reconstruction,
        'spikes': spike_data
    }


def reconstruct_hybrid(calcium_trace, fps):
    """Hybrid: Threshold n_iter=3 kinetics + Wavelet n_iter=3 events."""

    calcium_trace = np.ascontiguousarray(calcium_trace, dtype=np.float64)

    # STEP 1: Threshold n_iter=3 for kinetics optimization
    neuron_thresh = Neuron(
        cell_id='threshold_for_kinetics',
        ca=calcium_trace,
        sp=None,
        fps=fps
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

    n_events_thresh = len(neuron_thresh.threshold_events) if neuron_thresh.threshold_events else 0

    # Optimize kinetics on threshold events
    opt_result, kinetics_source = optimize_kinetics_safe(neuron_thresh, fps, 'threshold', 3)

    # Get optimized kinetics (in frames)
    t_rise_opt = neuron_thresh.t_rise if neuron_thresh.t_rise else neuron_thresh.default_t_rise
    t_off_opt = neuron_thresh.t_off if neuron_thresh.t_off else neuron_thresh.default_t_off

    # STEP 2: Wavelet n_iter=3 for event detection
    neuron_wavelet = Neuron(
        cell_id='wavelet_for_events',
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

    # STEP 3: Reconstruct using wavelet events with threshold kinetics
    spike_wavelet = neuron_wavelet.asp.data if neuron_wavelet.asp else neuron_wavelet.sp.data
    reconstruction = Neuron.get_restored_calcium(spike_wavelet, t_rise_opt, t_off_opt)[:len(calcium_trace)]

    # Calculate R²
    residuals = calcium_trace - reconstruction
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((calcium_trace - np.mean(calcium_trace)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return {
        'method': 'hybrid_thr3_wvt3',
        'n_events': n_events_wavelet,  # Events actually used for reconstruction
        'n_events_kinetics': n_events_thresh,  # Events used for kinetics optimization
        'kinetics_source': kinetics_source,
        't_rise': t_rise_opt / fps,
        't_off': t_off_opt / fps,
        'r2': r2,
        'reconstruction': reconstruction,
        'spikes': spike_wavelet
    }


def compare_all_methods(comp_idx, est):
    """Compare all 5 methods for one neuron."""

    # Find neuron
    neuron_pos = np.where(est.idx_components == comp_idx)[0][0]

    # Extract calcium trace
    C_raw = est.C[neuron_pos, :]
    if sparse.issparse(C_raw):
        C_raw = C_raw.toarray().flatten()
    C_raw = np.asarray(C_raw, dtype=np.float64)

    print(f"\nProcessing neuron {comp_idx}...")

    results = {}

    # Standard methods
    for method in ['threshold', 'wavelet']:
        for n_iter in [2, 3]:
            key = f'{method}_n{n_iter}'
            print(f"  {key}...", end=' ')
            result = reconstruct_standard(C_raw, FPS, method, n_iter)
            results[key] = result
            print(f"R²={result['r2']:.4f} ({result['n_events']} events)")

    # Hybrid method
    print(f"  hybrid...", end=' ')
    result = reconstruct_hybrid(C_raw, FPS)
    results['hybrid'] = result
    print(f"R²={result['r2']:.4f} ({result['n_events']} events from wavelet, {result['n_events_kinetics']} for kinetics)")

    # Find best R²
    best_r2 = max(r['r2'] for r in results.values())
    best_method = [k for k, v in results.items() if v['r2'] == best_r2][0]

    print(f"  BEST: {best_method} (R²={best_r2:.4f})")

    return results, best_method


def main():
    print("="*80)
    print("COMPREHENSIVE METHOD COMPARISON WITH HYBRID")
    print("="*80)

    # Load estimates
    print("\nLoading estimates...")
    est = load_estimates('wavelet_iter2')
    print(f"Loaded {len(est.idx_components)} neurons")

    # Load the 25 neurons from previous gallery
    summary_csv = BASE_PATH / 'reconstruction_gallery_25' / 'optimization_summary.csv'
    df = pd.read_csv(summary_csv)
    neurons_to_test = sorted(df['neuron_id'].unique())

    print(f"\nTesting on {len(neurons_to_test)} neurons from gallery_25")
    print(f"Neurons: {neurons_to_test}")

    # Create output directory
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Process all neurons
    all_results = {}
    best_methods_count = {
        'threshold_n2': 0,
        'threshold_n3': 0,
        'wavelet_n2': 0,
        'wavelet_n3': 0,
        'hybrid': 0
    }

    for comp_idx in neurons_to_test:
        results, best_method = compare_all_methods(comp_idx, est)
        all_results[comp_idx] = results
        best_methods_count[best_method] += 1

    # Generate summary statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\nBest method distribution (out of {len(neurons_to_test)} neurons):")
    for method, count in sorted(best_methods_count.items(), key=lambda x: -x[1]):
        percentage = count / len(neurons_to_test) * 100
        print(f"  {method:<20}: {count:>3} neurons ({percentage:>5.1f}%)")

    # Create detailed comparison CSV
    comparison_rows = []
    for neuron_id, results in all_results.items():
        row = {'neuron_id': neuron_id}

        for method_key, result in results.items():
            row[f'{method_key}_r2'] = result['r2']
            row[f'{method_key}_events'] = result['n_events']
            row[f'{method_key}_kinetics_source'] = result['kinetics_source']

        # Add hybrid kinetics events
        row['hybrid_kinetics_events'] = results['hybrid']['n_events_kinetics']

        # Find best method
        r2_values = {k: v['r2'] for k, v in results.items()}
        best_method = max(r2_values, key=r2_values.get)
        row['best_method'] = best_method
        row['best_r2'] = r2_values[best_method]

        # Calculate hybrid advantage
        row['hybrid_vs_best_standard'] = results['hybrid']['r2'] - max(
            results['threshold_n2']['r2'],
            results['threshold_n3']['r2'],
            results['wavelet_n2']['r2'],
            results['wavelet_n3']['r2']
        )

        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)

    # Save detailed CSV
    csv_file = OUTPUT_PATH / 'method_comparison_with_hybrid.csv'
    comparison_df.to_csv(csv_file, index=False)
    print(f"\nSaved detailed comparison: {csv_file}")

    # Statistical analysis
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS")
    print("="*80)

    # Mean R² by method
    print("\nMean R² by method:")
    r2_cols = [col for col in comparison_df.columns if col.endswith('_r2')]
    for col in r2_cols:
        method = col.replace('_r2', '')
        mean_r2 = comparison_df[col].mean()
        std_r2 = comparison_df[col].std()
        print(f"  {method:<20}: {mean_r2:.4f} ± {std_r2:.4f}")

    # Hybrid advantage statistics
    print("\nHybrid vs Best Standard Method:")
    hybrid_advantage = comparison_df['hybrid_vs_best_standard']
    print(f"  Mean advantage: {hybrid_advantage.mean():+.4f}")
    print(f"  Std: {hybrid_advantage.std():.4f}")
    print(f"  Median: {hybrid_advantage.median():+.4f}")
    print(f"  Min: {hybrid_advantage.min():+.4f}")
    print(f"  Max: {hybrid_advantage.max():+.4f}")

    n_wins = (hybrid_advantage > 0.001).sum()
    n_neutral = ((hybrid_advantage >= -0.001) & (hybrid_advantage <= 0.001)).sum()
    n_losses = (hybrid_advantage < -0.001).sum()

    print(f"\nHybrid performance breakdown:")
    print(f"  Wins (>0.1% better): {n_wins} ({n_wins/len(neurons_to_test)*100:.1f}%)")
    print(f"  Neutral (±0.1%): {n_neutral} ({n_neutral/len(neurons_to_test)*100:.1f}%)")
    print(f"  Losses (<-0.1%): {n_losses} ({n_losses/len(neurons_to_test)*100:.1f}%)")

    # Show top 5 neurons where hybrid wins
    print("\nTop 5 neurons where hybrid WINS:")
    top_hybrid = comparison_df.nlargest(5, 'hybrid_vs_best_standard')
    for idx, row in top_hybrid.iterrows():
        print(f"  Neuron {row['neuron_id']}: hybrid R²={row['hybrid_r2']:.4f}, "
              f"advantage={row['hybrid_vs_best_standard']:+.4f} "
              f"(vs {row['best_method']} R²={row['best_r2']:.4f})")

    # Show top 5 neurons where hybrid loses
    print("\nTop 5 neurons where hybrid LOSES:")
    worst_hybrid = comparison_df.nsmallest(5, 'hybrid_vs_best_standard')
    for idx, row in worst_hybrid.iterrows():
        print(f"  Neuron {row['neuron_id']}: hybrid R²={row['hybrid_r2']:.4f}, "
              f"disadvantage={row['hybrid_vs_best_standard']:+.4f} "
              f"(vs {row['best_method']} R²={row['best_r2']:.4f})")

    # Plot comparison
    plot_method_comparison(comparison_df)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if best_methods_count['hybrid'] > sum(best_methods_count.values()) / 2:
        print("\nHYBRID IS THE WINNER!")
        print("Hybrid approach achieves best R² for majority of neurons.")
        print("RECOMMENDATION: Use hybrid as default method.")
    elif hybrid_advantage.mean() > 0.005:
        print("\nHYBRID PROVIDES MODEST IMPROVEMENT")
        print(f"Mean advantage: {hybrid_advantage.mean():+.4f}")
        print("RECOMMENDATION: Consider hybrid for higher quality results.")
    elif hybrid_advantage.mean() > -0.005:
        print("\nHYBRID IS COMPETITIVE")
        print("Hybrid performs similarly to best standard methods.")
        print("RECOMMENDATION: Use hybrid when you want conservative event detection.")
    else:
        print("\nHYBRID DOES NOT IMPROVE OVER STANDARD METHODS")
        print("RECOMMENDATION: Continue using threshold n_iter=3 as default.")


def plot_method_comparison(df):
    """Create visualization comparing all methods."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: R² distribution by method
    ax = axes[0, 0]
    r2_data = []
    labels = []
    for method in ['threshold_n2', 'threshold_n3', 'wavelet_n2', 'wavelet_n3', 'hybrid']:
        r2_data.append(df[f'{method}_r2'].values)
        labels.append(method.replace('_', ' '))

    bp = ax.boxplot(r2_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['lightblue', 'blue', 'lightcoral', 'red', 'gold']):
        patch.set_facecolor(color)

    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('R² Distribution by Method', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # Plot 2: Hybrid advantage distribution
    ax = axes[0, 1]
    ax.hist(df['hybrid_vs_best_standard'], bins=30, color='gold', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Neutral')
    ax.axvline(df['hybrid_vs_best_standard'].mean(), color='blue', linestyle='-',
               linewidth=2, label=f'Mean: {df["hybrid_vs_best_standard"].mean():+.4f}')
    ax.set_xlabel('Hybrid R² - Best Standard R²', fontsize=12)
    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_title('Hybrid Advantage Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Best method frequency
    ax = axes[1, 0]
    method_counts = df['best_method'].value_counts()
    colors_map = {
        'threshold_n2': 'lightblue',
        'threshold_n3': 'blue',
        'wavelet_n2': 'lightcoral',
        'wavelet_n3': 'red',
        'hybrid': 'gold'
    }
    colors = [colors_map.get(m, 'gray') for m in method_counts.index]

    ax.bar(range(len(method_counts)), method_counts.values, color=colors)
    ax.set_xticks(range(len(method_counts)))
    ax.set_xticklabels([m.replace('_', ' ') for m in method_counts.index], rotation=45, ha='right')
    ax.set_ylabel('Number of Neurons', fontsize=12)
    ax.set_title('Best Method Frequency', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Mean R² comparison
    ax = axes[1, 1]
    methods = ['threshold_n2', 'threshold_n3', 'wavelet_n2', 'wavelet_n3', 'hybrid']
    mean_r2 = [df[f'{m}_r2'].mean() for m in methods]
    std_r2 = [df[f'{m}_r2'].std() for m in methods]
    colors_list = ['lightblue', 'blue', 'lightcoral', 'red', 'gold']

    x = range(len(methods))
    ax.bar(x, mean_r2, yerr=std_r2, color=colors_list, capsize=5, edgecolor='black', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in methods], rotation=45, ha='right')
    ax.set_ylabel('Mean R² ± Std', fontsize=12)
    ax.set_title('Mean R² Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_file = OUTPUT_PATH / 'method_comparison_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved summary plot: {output_file}")


if __name__ == '__main__':
    main()
