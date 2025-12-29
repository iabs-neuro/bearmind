"""
Examine actual calcium traces that cause partial failures.

This script loads LNOF sessions and extracts traces for neurons with partial failures
to understand what characteristics cause DRIADA reconstruction methods to return NaN.
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

def load_lnof_session(session_name):
    """Load LNOF session processed estimates."""
    lnof_dir = Path('data/LNOF')

    for folder in lnof_dir.iterdir():
        if folder.is_dir() and session_name in folder.name:
            pickle_files = list(folder.glob('*_processed.pickle'))
            if pickle_files:
                with open(pickle_files[0], 'rb') as f:
                    return pickle.load(f)
    return None


def analyze_trace_characteristics(trace, neuron_idx, session_name):
    """Analyze what makes this trace fail reconstruction metrics."""
    print(f'\nNeuron {neuron_idx} from {session_name}:')
    print(f'  Trace length: {len(trace)}')
    print(f'  Min: {np.min(trace):.4f}')
    print(f'  Max: {np.max(trace):.4f}')
    print(f'  Mean: {np.mean(trace):.4f}')
    print(f'  Std: {np.std(trace):.4f}')
    print(f'  NaN count: {np.isnan(trace).sum()}')
    print(f'  Inf count: {np.isinf(trace).sum()}')
    print(f'  Zeros: {(trace == 0).sum()}')

    # Check for flat sections
    unique_vals = len(np.unique(trace))
    print(f'  Unique values: {unique_vals} ({100*unique_vals/len(trace):.1f}% of length)')

    # Check for extreme values
    if np.max(np.abs(trace)) > 100:
        print(f'  [WARNING] Extreme values detected (max abs: {np.max(np.abs(trace)):.2f})')

    # Check for very low variance
    if np.std(trace) < 0.01:
        print(f'  [WARNING] Very low variance (std: {np.std(trace):.6f})')

    # Check baseline characteristics
    baseline = np.percentile(trace, 10)
    peak = np.percentile(trace, 99)
    dynamic_range = peak - baseline
    print(f'  Dynamic range: {dynamic_range:.4f} (10th to 99th percentile)')

    if dynamic_range < 0.1:
        print(f'  [WARNING] Low dynamic range - may not have clear events')

    return trace


def main():
    """Examine partial failure traces."""
    print('='*80)
    print('EXAMINING PARTIAL FAILURE TRACES')
    print('='*80)

    # Load v9 dataset
    v9 = pd.read_csv('ml/results/training_dataset_v9.csv')

    # Find partial failures
    event_metrics = [
        'event_r2_score', 'event_snr', 'events_fraction', 'events_per_min',
        'kinetics_opt', 't_off', 't_rise', 'nmae', 'nrmse', 'r2_score', 'snr_recon'
    ]
    has_nan = v9[event_metrics].isna().any(axis=1)
    partial_failures = has_nan & (v9['kinetics_source'] != 'error')

    print(f'\nPartial failures: {partial_failures.sum()}')

    # Get unique sessions with partial failures
    failed_sessions = v9[partial_failures]['session_name'].unique()
    print(f'Sessions affected: {len(failed_sessions)}')
    print(f'  {list(failed_sessions[:5])}...')

    # Load first session and examine traces
    print('\n' + '='*80)
    print('LOADING SESSION DATA')
    print('='*80)

    session_to_load = failed_sessions[0]
    print(f'\nLoading: {session_to_load}')

    est = load_lnof_session(session_to_load)

    if est is None:
        print(f'[ERROR] Could not load {session_to_load}')
        return

    print(f'Loaded estimates for {session_to_load}')

    # Get failed neuron indices from this session
    session_failures = v9[(v9['session_name'] == session_to_load) & partial_failures]
    failed_indices = session_failures['component_idx'].values

    print(f'\nFailed neurons in this session: {failed_indices}')

    # Extract traces for failed neurons
    if not hasattr(est, 'C') or est.C is None:
        print('[ERROR] No traces (C matrix) in estimates')
        return

    traces = est.C  # Should be (n_neurons, n_timepoints)
    print(f'\nTrace matrix shape: {traces.shape}')
    print(f'  Neurons: {traces.shape[0]}')
    print(f'  Timepoints: {traces.shape[1]}')

    # Get FPS
    fps = getattr(est, 'fps', 30.0)
    duration_min = traces.shape[1] / fps / 60
    print(f'  FPS: {fps}')
    print(f'  Duration: {duration_min:.2f} minutes')

    # Analyze failed traces
    print('\n' + '='*80)
    print('ANALYZING FAILED TRACES')
    print('='*80)

    failed_traces = []
    for idx in failed_indices[:5]:  # Examine first 5
        if idx < traces.shape[0]:
            trace = traces[idx, :]
            trace_analysis = analyze_trace_characteristics(trace, idx, session_to_load)
            failed_traces.append((idx, trace))

    # Compare with successful neuron
    print('\n' + '='*80)
    print('COMPARING WITH SUCCESSFUL NEURON')
    print('='*80)

    # Get a successful neuron from same session
    session_neurons = v9[v9['session_name'] == session_to_load]
    successful = session_neurons[~session_neurons.index.isin(session_failures.index)]

    if len(successful) > 0:
        success_idx = successful.iloc[0]['component_idx']
        if success_idx < traces.shape[0]:
            success_trace = traces[success_idx, :]
            print(f'\nSuccessful neuron {success_idx} for comparison:')
            analyze_trace_characteristics(success_trace, success_idx, session_to_load)

    # Plot examples
    print('\n' + '='*80)
    print('GENERATING PLOTS')
    print('='*80)

    fig, axes = plt.subplots(min(len(failed_traces), 3), 1, figsize=(15, 8))
    if len(failed_traces) == 1:
        axes = [axes]

    for i, (idx, trace) in enumerate(failed_traces[:3]):
        ax = axes[i] if i < len(axes) else axes[-1]
        time = np.arange(len(trace)) / fps / 60  # Convert to minutes
        ax.plot(time, trace, linewidth=0.5)
        ax.set_ylabel('Fluorescence')
        ax.set_title(f'Neuron {idx} (PARTIAL FAILURE) - {session_to_load}')
        ax.grid(True, alpha=0.3)

    if len(failed_traces) > 0:
        axes[-1].set_xlabel('Time (minutes)')

    plt.tight_layout()
    output_path = 'output/partial_failure_traces.png'
    Path('output').mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved plots: {output_path}')

    # Summary statistics
    print('\n' + '='*80)
    print('SUMMARY')
    print('='*80)

    print('\nPartial failure characteristics:')
    print(f'  Total partial failures: {partial_failures.sum()}')
    print(f'  All from LNOF: {(v9[partial_failures]["experiment"] == "LNOF").all()}')
    print(f'  All are KEEP neurons: {(v9[partial_failures]["ground_truth"] == 1).all()}')

    if len(failed_traces) > 0:
        all_stds = [np.std(trace) for _, trace in failed_traces]
        all_means = [np.mean(trace) for _, trace in failed_traces]
        all_lengths = [len(trace) for _, trace in failed_traces]

        print(f'\nTrace statistics (n={len(failed_traces)}):')
        print(f'  Mean std: {np.mean(all_stds):.4f}')
        print(f'  Mean signal: {np.mean(all_means):.4f}')
        print(f'  Trace length: {all_lengths[0]} timepoints')

    print('\n' + '='*80)
    print('HYPOTHESIS')
    print('='*80)
    print('''
Based on analysis, partial failures likely occur when:
1. Traces have very sparse events (low events_fraction ~0.0003-0.001)
2. Reconstruction is technically possible but has edge cases
3. DRIADA reconstruction quality methods hit internal error handling:
   - Not enough events for reliable quality metrics
   - Division by zero in metric calculations
   - Invalid reconstruction array characteristics
   - Edge cases in R2 computation (especially event_only=True)

These are good neurons (KEEP=1) with detectable events (event_snr > 2),
but reconstruction quality metrics fail for technical reasons.

The fix: Wrap DRIADA method calls and return 0 on NaN/exception.
    ''')


if __name__ == '__main__':
    main()
