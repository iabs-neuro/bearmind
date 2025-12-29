"""
Visualize traces of false negatives from event_r2_score < 0.15 rule.
These are neurons labeled KEEP but flagged by the rule.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

print('='*80)
print('VISUALIZING FALSE NEGATIVES: event_r2_score < 0.15 RULE')
print('='*80)

# Load v9 dataset
v9 = pd.read_csv('ml/results/training_dataset_v9.csv')
v9_valid = v9[v9['event_r2_score'].notna()].copy()

# Find false negatives: flagged by rule but labeled KEEP
flagged_by_rule = v9_valid['event_r2_score'] < 0.15
labeled_keep = v9_valid['ground_truth'] == 1

false_negatives = v9_valid[flagged_by_rule & labeled_keep].copy()
print(f'\nTotal false negatives: {len(false_negatives)}')

# Sort by event_r2_score to see worst cases first
false_negatives = false_negatives.sort_values('event_r2_score')

# Filter to LNOF only for visualization (we have the data)
fn_lnof = false_negatives[false_negatives['experiment'] == 'LNOF'].copy()
print(f'LNOF false negatives: {len(fn_lnof)}')

# Take worst 12 cases from LNOF
n_show = min(12, len(fn_lnof))
worst_fn = fn_lnof.head(n_show)

print(f'\nVisualizing worst {n_show} LNOF false negatives:')
print(f'\n{"Session":<20} {"Neuron":>6} {"event_r2":>10} {"r2_score":>10} {"event_snr":>10} {"events/min":>12}')
print('-' * 75)
for _, row in worst_fn.iterrows():
    print(f'{row["session_name"]:<20} {row["component_idx"]:>6.0f} {row["event_r2_score"]:>10.4f} '
          f'{row["r2_score"]:>10.4f} {row["event_snr"]:>10.2f} {row["events_per_min"]:>12.2f}')

# Visualize traces
fig, axes = plt.subplots(n_show, 1, figsize=(16, 2.5*n_show))
if n_show == 1:
    axes = [axes]

for plot_idx, (_, row) in enumerate(worst_fn.iterrows()):
    session_name = row['session_name']
    component_idx = int(row['component_idx'])
    experiment = row['experiment']
    event_r2 = row['event_r2_score']
    r2_score = row['r2_score']

    ax = axes[plot_idx]

    # Find processed estimates file (LNOF only)
    lnof_dir = Path('data/LNOF')
    matching_dirs = list(lnof_dir.glob(f'inspection_artifacts_{session_name}_*'))

    if not matching_dirs:
        ax.text(0.5, 0.5, f'Estimates not found', ha='center', va='center', fontsize=10)
        ax.set_title(f'{session_name} n={component_idx} (event_r2={event_r2:.3f})', fontsize=10)
        ax.axis('off')
        continue

    artifact_dir = matching_dirs[0]
    processed_files = list(artifact_dir.glob(f'{session_name}_*_processed.pickle'))
    metrics_files = list(artifact_dir.glob(f'{session_name}_metrics_with_decisions.csv'))

    if not processed_files or not metrics_files:
        ax.text(0.5, 0.5, f'Files not found', ha='center', va='center', fontsize=10)
        ax.set_title(f'{session_name} n={component_idx} (event_r2={event_r2:.3f})', fontsize=10)
        ax.axis('off')
        continue

    try:
        # Load estimates and metrics
        with open(processed_files[0], 'rb') as f:
            estimates = pickle.load(f)

        metrics_df = pd.read_csv(metrics_files[0])

        # Find row index
        comp_mask = metrics_df['component_idx'] == component_idx
        if not comp_mask.any():
            ax.text(0.5, 0.5, f'Component not found', ha='center', va='center', fontsize=10)
            ax.set_title(f'{session_name} n={component_idx} (event_r2={event_r2:.3f})', fontsize=10)
            ax.axis('off')
            continue

        row_idx = metrics_df[comp_mask].index[0]

        # Extract trace
        if hasattr(estimates, 'C') and estimates.C is not None:
            if row_idx < estimates.C.shape[0]:
                trace = estimates.C[row_idx, :].copy()

                # Normalize
                trace_norm = (trace - trace.min()) / (trace.max() - trace.min() + 1e-10)

                # Plot
                time = np.arange(len(trace))
                ax.plot(time, trace_norm, 'k-', linewidth=1, alpha=0.7)

                # Mark events if available
                if hasattr(estimates, 'S') and estimates.S is not None:
                    if row_idx < estimates.S.shape[0]:
                        events = estimates.S[row_idx, :].copy()
                        if events.size > 0:
                            event_times = np.where(events > 0)[0]
                            for t in event_times:
                                ax.axvline(t, color='red', alpha=0.5, linewidth=0.5)

                            # Add event count
                            ax.text(0.02, 0.95, f'{len(event_times)} events',
                                   transform=ax.transAxes, fontsize=9,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

                # Metadata
                caiman_snr = row.get('caiman_snr', np.nan)
                event_snr = row.get('event_snr', np.nan)
                events_per_min = row.get('events_per_min', np.nan)

                title_text = (f'{session_name} n={component_idx} | '
                             f'event_r2={event_r2:.4f} | r2={r2_score:.3f} | '
                             f'caiman_snr={caiman_snr:.2f} | event_snr={event_snr:.2f} | '
                             f'events/min={events_per_min:.2f}')
                ax.set_title(title_text, fontsize=9, pad=5)
                ax.set_xlabel('Frame', fontsize=9)
                ax.set_ylabel('Normalized F', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.set_ylim([-0.05, 1.1])
            else:
                ax.text(0.5, 0.5, f'Index error', ha='center', va='center', fontsize=10)
                ax.set_title(f'{session_name} n={component_idx} (event_r2={event_r2:.3f})', fontsize=10)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'No C matrix', ha='center', va='center', fontsize=10)
            ax.set_title(f'{session_name} n={component_idx} (event_r2={event_r2:.3f})', fontsize=10)
            ax.axis('off')

    except Exception as e:
        ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', ha='center', va='center', fontsize=8)
        ax.set_title(f'{session_name} n={component_idx} (event_r2={event_r2:.3f})', fontsize=10)
        ax.axis('off')
        print(f'ERROR: {session_name} n={component_idx}: {e}')

plt.tight_layout()
output_path = 'output/event_r2_015_false_negatives_traces.png'
Path('output').mkdir(exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f'\n\nSaved: {output_path}')
plt.close()

# Summary statistics
print('\n' + '='*80)
print('FALSE NEGATIVE STATISTICS')
print('='*80)

print(f'\nAll {len(false_negatives)} false negatives:')
print(f'  event_r2_score: mean={false_negatives["event_r2_score"].mean():.4f}, '
      f'median={false_negatives["event_r2_score"].median():.4f}, '
      f'min={false_negatives["event_r2_score"].min():.4f}')
print(f'  r2_score: mean={false_negatives["r2_score"].mean():.4f}, '
      f'median={false_negatives["r2_score"].median():.4f}')
print(f'  event_snr: mean={false_negatives["event_snr"].mean():.2f}, '
      f'median={false_negatives["event_snr"].median():.2f}')
print(f'  events_per_min: mean={false_negatives["events_per_min"].mean():.2f}, '
      f'median={false_negatives["events_per_min"].median():.2f}')

# Compare to all KEEP neurons
keep_all = v9_valid[v9_valid['ground_truth'] == 1]
print(f'\nAll KEEP neurons (for comparison):')
print(f'  event_r2_score: mean={keep_all["event_r2_score"].mean():.4f}, '
      f'median={keep_all["event_r2_score"].median():.4f}')
print(f'  r2_score: mean={keep_all["r2_score"].mean():.4f}, '
      f'median={keep_all["r2_score"].median():.4f}')
print(f'  event_snr: mean={keep_all["event_snr"].mean():.2f}, '
      f'median={keep_all["event_snr"].median():.2f}')
print(f'  events_per_min: mean={keep_all["events_per_min"].mean():.2f}, '
      f'median={keep_all["events_per_min"].median():.2f}')

print('\n' + '='*80)
print('INTERPRETATION')
print('='*80)
print('''
These false negatives are KEEP neurons flagged by event_r2_score < 0.15.
Visual inspection will show if they are:
1. Legitimate good neurons (true false negatives - rule is too strict)
2. Borderline/questionable neurons (acceptable collateral damage)
3. Potentially mislabeled neurons (should actually be DELETE)

Check the traces above to make the judgment call.
''')
print('='*80)
