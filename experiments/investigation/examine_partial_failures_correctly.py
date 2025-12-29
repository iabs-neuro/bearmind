"""
Correctly examine partial failure traces by loading from processed estimates.

Fix indexing issue - use metrics_df to find correct trace mapping.
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
                print(f'Loading: {pickle_files[0].name}')
                with open(pickle_files[0], 'rb') as f:
                    return pickle.load(f), folder
    return None, None


def analyze_trace_characteristics(trace, neuron_idx, session_name, metrics_row):
    """Analyze trace and show metrics from original processing."""
    print(f'\n{"="*80}')
    print(f'Neuron {neuron_idx} from {session_name}')
    print(f'{"="*80}')

    # Original metrics from processing
    print('\nORIGINAL METRICS (from metrics_df):')
    print(f'  caiman_snr: {metrics_row.get("caiman_snr", "N/A")}')
    print(f'  caiman_r_score: {metrics_row.get("caiman_r_score", "N/A")}')
    print(f'  Signal metrics (VALID):')
    print(f'    event_snr: {metrics_row.get("event_snr", "N/A")}')
    print(f'    events_fraction: {metrics_row.get("events_fraction", "N/A")}')
    print(f'    events_per_min: {metrics_row.get("events_per_min", "N/A")}')
    print(f'    kinetics_opt: {metrics_row.get("kinetics_opt", "N/A")}')
    print(f'    t_rise: {metrics_row.get("t_rise", "N/A")}')
    print(f'    t_off: {metrics_row.get("t_off", "N/A")}')
    print(f'  Reconstruction metrics (NaN):')
    print(f'    event_r2_score: {metrics_row.get("event_r2_score", "N/A")}')
    print(f'    r2_score: {metrics_row.get("r2_score", "N/A")}')
    print(f'    nmae: {metrics_row.get("nmae", "N/A")}')
    print(f'    nrmse: {metrics_row.get("nrmse", "N/A")}')
    print(f'    snr_recon: {metrics_row.get("snr_recon", "N/A")}')

    # Trace characteristics
    print('\nTRACE CHARACTERISTICS:')
    print(f'  Length: {len(trace)} timepoints')
    print(f'  Min: {np.min(trace):.4f}')
    print(f'  Max: {np.max(trace):.4f}')
    print(f'  Mean: {np.mean(trace):.4f}')
    print(f'  Std: {np.std(trace):.4f}')
    print(f'  Median: {np.median(trace):.4f}')

    # Data quality
    print('\nDATA QUALITY:')
    print(f'  NaN values: {np.isnan(trace).sum()}')
    print(f'  Inf values: {np.isinf(trace).sum()}')
    print(f'  Negative values: {(trace < 0).sum()}')
    print(f'  Zero values: {(trace == 0).sum()}')
    unique_vals = len(np.unique(trace))
    print(f'  Unique values: {unique_vals} ({100*unique_vals/len(trace):.1f}% of length)')

    # Signal characteristics
    print('\nSIGNAL CHARACTERISTICS:')
    baseline = np.percentile(trace, 10)
    peak = np.percentile(trace, 99)
    dynamic_range = peak - baseline
    print(f'  Baseline (10th percentile): {baseline:.4f}')
    print(f'  Peak (99th percentile): {peak:.4f}')
    print(f'  Dynamic range: {dynamic_range:.4f}')

    # SNR estimate
    noise = np.percentile(np.abs(np.diff(trace)), 95)
    signal = peak - baseline
    snr_estimate = signal / noise if noise > 0 else 0
    print(f'  Estimated SNR: {snr_estimate:.2f}')

    # Detect transients
    threshold = baseline + 2 * np.std(trace)
    above_threshold = trace > threshold
    n_frames_active = above_threshold.sum()
    activity_fraction = n_frames_active / len(trace)
    print(f'  Frames above threshold (baseline + 2*std): {n_frames_active} ({100*activity_fraction:.2f}%)')

    return trace


def main():
    """Examine partial failure traces with correct indexing."""
    print('='*80)
    print('EXAMINING PARTIAL FAILURE TRACES - CORRECT INDEXING')
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

    # Get first few unique sessions
    failed_sessions = v9[partial_failures]['session_name'].unique()[:3]
    print(f'\nExamining sessions: {list(failed_sessions)}')

    all_failed_traces = []

    for session_name in failed_sessions:
        print(f'\n{"="*80}')
        print(f'LOADING SESSION: {session_name}')
        print(f'{"="*80}')

        # Load estimates
        est, folder = load_lnof_session(session_name)
        if est is None:
            print(f'[ERROR] Could not load {session_name}')
            continue

        # Get metrics_df
        if not hasattr(est, 'metrics_df') or est.metrics_df is None:
            print(f'[ERROR] No metrics_df in estimates')
            continue

        metrics_df = est.metrics_df
        print(f'\nmetrics_df shape: {metrics_df.shape}')
        print(f'component_idx range: {metrics_df["component_idx"].min()} to {metrics_df["component_idx"].max()}')

        # Get failed neurons from v9 for this session
        session_failures = v9[(v9['session_name'] == session_name) & partial_failures]
        failed_indices = session_failures['component_idx'].values
        print(f'\nFailed component indices from v9: {failed_indices}')

        # Check traces
        if not hasattr(est, 'C') or est.C is None:
            print(f'[ERROR] No C matrix in estimates')
            continue

        traces = est.C
        print(f'\nC matrix shape: {traces.shape}')
        print(f'  Components: {traces.shape[0]}')
        print(f'  Timepoints: {traces.shape[1]}')

        # Get FPS
        fps = getattr(est, 'fps', 30.0)
        print(f'  FPS: {fps}')

        # For each failed neuron, find it in metrics_df and get trace
        for comp_idx in failed_indices[:2]:  # Limit to 2 per session
            # Find this component in metrics_df
            comp_row_mask = metrics_df['component_idx'] == comp_idx

            if not comp_row_mask.any():
                print(f'\n[WARNING] Component {comp_idx} not found in metrics_df!')
                continue

            comp_row = metrics_df[comp_row_mask].iloc[0]

            # The row index in metrics_df should match the row in C matrix
            # After component evaluation, metrics_df row i corresponds to C[i, :]
            row_idx = metrics_df[comp_row_mask].index[0]

            print(f'\nComponent {comp_idx}:')
            print(f'  Row in metrics_df: {row_idx}')
            print(f'  Checking if row_idx < C.shape[0]: {row_idx < traces.shape[0]}')

            if row_idx < traces.shape[0]:
                trace = traces[row_idx, :]
                trace_analysis = analyze_trace_characteristics(
                    trace, comp_idx, session_name, comp_row.to_dict()
                )
                all_failed_traces.append((session_name, comp_idx, trace, comp_row))
            else:
                print(f'  [ERROR] Row index {row_idx} out of bounds for C matrix')

    # Generate plots
    if len(all_failed_traces) > 0:
        print('\n' + '='*80)
        print('GENERATING PLOTS')
        print('='*80)

        n_plots = min(len(all_failed_traces), 6)
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 2.5*n_plots))
        if n_plots == 1:
            axes = [axes]

        fps = 30.0  # Default
        for i, (sess, idx, trace, metrics) in enumerate(all_failed_traces[:n_plots]):
            ax = axes[i]
            time = np.arange(len(trace)) / fps / 60  # Convert to minutes
            ax.plot(time, trace, linewidth=0.5, color='steelblue')

            # Mark events if available
            events_frac = metrics.get('events_fraction', 0)
            title = f'{sess} - Neuron {idx} (PARTIAL FAILURE)\n'
            title += f'event_snr={metrics.get("event_snr", "N/A"):.2f}, '
            title += f'events_frac={events_frac:.5f}, '
            title += f'caiman_snr={metrics.get("caiman_snr", "N/A"):.2f}'

            ax.set_ylabel('Fluorescence')
            ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time (minutes)')
        plt.tight_layout()

        output_path = 'output/partial_failure_traces_correct.png'
        Path('output').mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f'\nSaved: {output_path}')
        plt.close()

    # Summary
    print('\n' + '='*80)
    print('ANALYSIS COMPLETE')
    print('='*80)
    print(f'\nExamined {len(all_failed_traces)} partial failure traces')
    print('\nConclusion:')
    print('  If traces look normal with clear activity, then partial failures are')
    print('  due to DRIADA reconstruction quality methods returning NaN for edge')
    print('  cases (sparse events, reconstruction computation issues), NOT bad traces.')


if __name__ == '__main__':
    main()
