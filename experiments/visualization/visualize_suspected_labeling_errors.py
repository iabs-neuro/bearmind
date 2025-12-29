"""
Visualize traces and reconstructions for suspected labeling errors.
Load from processed estimates to show actual calcium signals.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

print('='*80)
print('VISUALIZING SUSPECTED LABELING ERRORS - TRACES AND RECONSTRUCTIONS')
print('='*80)

# Load suspected errors
suspected = pd.read_csv('ml/results/suspected_labeling_errors_r2.csv')
print(f'\nLoaded {len(suspected)} suspected labeling errors')

# Filter to LNOF only
suspected_lnof = suspected[suspected['experiment'] == 'LNOF'].copy()
print(f'LNOF only: {len(suspected_lnof)} suspected labeling errors')

# Focus on top 12 worst cases for visualization
n_show = 12
worst_cases = suspected_lnof.head(n_show)

print(f'\nVisualizing top {n_show} worst cases:')
for idx, row in worst_cases.iterrows():
    print(f'  {row["session_name"]:20s} neuron {row["component_idx"]:4.0f}: '
          f'r2={row["r2_score"]:7.3f}, exp={row["experiment"]}')

# Create figure with subplots
fig, axes = plt.subplots(n_show, 1, figsize=(16, 2.5*n_show))
if n_show == 1:
    axes = [axes]

for plot_idx, (_, row) in enumerate(worst_cases.iterrows()):
    session_name = row['session_name']
    component_idx = int(row['component_idx'])
    experiment = row['experiment']
    r2_score = row['r2_score']

    ax = axes[plot_idx]

    # Find processed estimates file (LNOF only)
    estimates_file = None
    metrics_file = None

    # LNOF files are in inspection_artifacts folders
    lnof_dir = Path('data/LNOF')
    # Find the inspection artifacts folder for this session
    matching_dirs = list(lnof_dir.glob(f'inspection_artifacts_{session_name}_*'))
    if matching_dirs:
        artifact_dir = matching_dirs[0]
        # Find the processed pickle file
        processed_files = list(artifact_dir.glob(f'{session_name}_*_processed.pickle'))
        if processed_files:
            estimates_file = processed_files[0]
        # Metrics file
        metrics_files = list(artifact_dir.glob(f'{session_name}_metrics_with_decisions.csv'))
        if metrics_files:
            metrics_file = metrics_files[0]

    if estimates_file is None or not estimates_file.exists():
        ax.text(0.5, 0.5, f'Estimates file not found:\n{estimates_file}',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'{session_name} neuron {component_idx} (r2={r2_score:.3f}) - FILE NOT FOUND')
        ax.axis('off')
        continue

    try:
        # Load estimates
        with open(estimates_file, 'rb') as f:
            estimates = pickle.load(f)

        # Get metrics to find the row corresponding to this component
        if metrics_file is None or not metrics_file.exists():
            ax.text(0.5, 0.5, f'Metrics file not found',
                    ha='center', va='center', fontsize=10, color='red')
            ax.set_title(f'{session_name} neuron {component_idx} (r2={r2_score:.3f}) - METRICS NOT FOUND')
            ax.axis('off')
            continue

        metrics_df = pd.read_csv(metrics_file)

        # Find row index for this component
        comp_mask = metrics_df['component_idx'] == component_idx
        if not comp_mask.any():
            ax.text(0.5, 0.5, f'Component {component_idx} not found in metrics',
                    ha='center', va='center', fontsize=10, color='red')
            ax.set_title(f'{session_name} neuron {component_idx} (r2={r2_score:.3f}) - NOT FOUND')
            ax.axis('off')
            continue

        row_idx = metrics_df[comp_mask].index[0]

        # Extract trace (C matrix contains calcium traces)
        if hasattr(estimates, 'C') and estimates.C is not None:
            if row_idx < estimates.C.shape[0]:
                trace = estimates.C[row_idx, :].copy()

                # Normalize trace for visualization
                trace_normalized = (trace - trace.min()) / (trace.max() - trace.min() + 1e-10)

                # Extract spike events (S matrix)
                events = None
                if hasattr(estimates, 'S') and estimates.S is not None:
                    if row_idx < estimates.S.shape[0]:
                        events = estimates.S[row_idx, :].copy()

                # Try to get reconstruction from estimates
                reconstruction = None
                if hasattr(estimates, 'F_dff') and estimates.F_dff is not None:
                    # F_dff might contain fitted reconstructions
                    if row_idx < estimates.F_dff.shape[0]:
                        reconstruction = estimates.F_dff[row_idx, :].copy()

                # Plot trace
                time = np.arange(len(trace))
                ax.plot(time, trace_normalized, 'k-', linewidth=1, alpha=0.7, label='Raw trace')

                # Plot events as vertical lines if available
                if events is not None:
                    event_times = np.where(events > 0)[0]
                    event_amplitudes = events[event_times]
                    # Normalize event amplitudes
                    if len(event_amplitudes) > 0:
                        event_amp_norm = event_amplitudes / event_amplitudes.max() * 0.3
                        for t, amp in zip(event_times, event_amp_norm):
                            ax.axvline(t, color='red', alpha=0.5, linewidth=0.5)

                        # Add text showing number of events
                        n_events = len(event_times)
                        ax.text(0.02, 0.95, f'{n_events} events',
                               transform=ax.transAxes, fontsize=9,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

                # Plot reconstruction if available
                if reconstruction is not None:
                    recon_normalized = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min() + 1e-10)
                    ax.plot(time, recon_normalized, 'b-', linewidth=1.5, alpha=0.5, label='Reconstruction')

                # Add metadata
                caiman_snr = row.get('caiman_snr', np.nan)
                event_snr = row.get('event_snr', np.nan)
                events_per_min = row.get('events_per_min', np.nan)
                nmae = row.get('nmae', np.nan)

                title_text = (f'{session_name} neuron {component_idx} ({experiment}) | '
                             f'r2={r2_score:.3f} | caiman_snr={caiman_snr:.2f} | '
                             f'event_snr={event_snr:.2f} | events/min={events_per_min:.2f} | '
                             f'nmae={nmae:.2f}')
                ax.set_title(title_text, fontsize=10, pad=5)

                ax.set_xlabel('Frame', fontsize=9)
                ax.set_ylabel('Normalized F', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.05, 1.1])

                if reconstruction is not None:
                    ax.legend(fontsize=8, loc='upper right')

            else:
                ax.text(0.5, 0.5, f'Row index {row_idx} out of bounds (C shape: {estimates.C.shape})',
                        ha='center', va='center', fontsize=10, color='red')
                ax.set_title(f'{session_name} neuron {component_idx} (r2={r2_score:.3f}) - INDEX ERROR')
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'C matrix not found in estimates',
                    ha='center', va='center', fontsize=10, color='red')
            ax.set_title(f'{session_name} neuron {component_idx} (r2={r2_score:.3f}) - NO C MATRIX')
            ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error loading data:\n{str(e)}',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'{session_name} neuron {component_idx} (r2={r2_score:.3f}) - ERROR')
        ax.axis('off')
        print(f'ERROR processing {session_name} neuron {component_idx}: {e}')

plt.tight_layout()
output_path = 'output/suspected_errors_lnof_traces_top12.png'
Path('output').mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {output_path}')
plt.close()

# Create a second figure with spatial footprints for the same neurons
print('\n' + '='*80)
print('VISUALIZING SPATIAL FOOTPRINTS (LNOF ONLY)')
print('='*80)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for plot_idx, (_, row) in enumerate(worst_cases.iterrows()):
    if plot_idx >= 12:
        break

    session_name = row['session_name']
    component_idx = int(row['component_idx'])
    experiment = row['experiment']
    r2_score = row['r2_score']

    ax = axes[plot_idx]

    # Find processed estimates file (LNOF only)
    estimates_file = None
    metrics_file = None

    # LNOF files are in inspection_artifacts folders
    lnof_dir = Path('data/LNOF')
    matching_dirs = list(lnof_dir.glob(f'inspection_artifacts_{session_name}_*'))
    if matching_dirs:
        artifact_dir = matching_dirs[0]
        processed_files = list(artifact_dir.glob(f'{session_name}_*_processed.pickle'))
        if processed_files:
            estimates_file = processed_files[0]
        metrics_files = list(artifact_dir.glob(f'{session_name}_metrics_with_decisions.csv'))
        if metrics_files:
            metrics_file = metrics_files[0]

    if estimates_file is None or not estimates_file.exists():
        ax.text(0.5, 0.5, 'File not found', ha='center', va='center', fontsize=10)
        ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
        ax.axis('off')
        continue

    try:
        # Load estimates
        with open(estimates_file, 'rb') as f:
            estimates = pickle.load(f)

        # Get metrics
        if metrics_file is None or not metrics_file.exists():
            ax.text(0.5, 0.5, 'Metrics not found', ha='center', va='center', fontsize=10)
            ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
            ax.axis('off')
            continue

        metrics_df = pd.read_csv(metrics_file)
        comp_mask = metrics_df['component_idx'] == component_idx
        if not comp_mask.any():
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', fontsize=10)
            ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
            ax.axis('off')
            continue

        row_idx = metrics_df[comp_mask].index[0]

        # Extract spatial footprint (A matrix)
        if hasattr(estimates, 'A') and estimates.A is not None:
            if row_idx < estimates.A.shape[1]:
                # Get spatial component
                spatial = estimates.A[:, row_idx].toarray().flatten()

                # Reshape to 2D (need dimensions)
                if hasattr(estimates, 'dims'):
                    dims = estimates.dims
                    spatial_2d = spatial.reshape(dims, order='F')

                    # Plot
                    im = ax.imshow(spatial_2d, cmap='viridis', interpolation='nearest')
                    ax.set_title(f'{session_name[:15]}\nn={component_idx}, r2={r2_score:.2f}', fontsize=8)
                    ax.axis('off')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'No dims', ha='center', va='center', fontsize=10)
                    ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Index error', ha='center', va='center', fontsize=10)
                ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No A matrix', ha='center', va='center', fontsize=10)
            ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
            ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=10)
        ax.set_title(f'{session_name[:15]}\nn={component_idx}', fontsize=8)
        ax.axis('off')
        print(f'ERROR processing footprint for {session_name} neuron {component_idx}: {e}')

plt.tight_layout()
output_path_footprints = 'output/suspected_errors_lnof_footprints_top12.png'
plt.savefig(output_path_footprints, dpi=150, bbox_inches='tight')
print(f'Saved: {output_path_footprints}')
plt.close()

print('\n' + '='*80)
print('COMPLETE')
print('='*80)
print(f'\nGenerated visualizations (LNOF only):')
print(f'  1. Traces: {output_path}')
print(f'  2. Spatial footprints: {output_path_footprints}')
print(f'\nThese show the top {n_show} worst LNOF suspected labeling errors.')
print(f'If these traces look like artifacts (noisy, few events, poor signal),')
print(f'it confirms they are mislabeled and should be marked DELETE.')
