"""
Visualize top errors from v8_iter5 model at threshold 0.80.
Generates PNG pages showing trace and footprint for manual review.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

THRESHOLD = 0.80
NEURONS_PER_PAGE = 5
N_FP_PAGES = 20
N_FN_PAGES = 20

print('='*80)
print(f'VISUALIZING v8_iter5 ERRORS AT THRESHOLD {THRESHOLD}')
print('='*80)

# Load dataset with predictions
df = pd.read_csv('ml/results/training_dataset_v8_iter5_with_predictions.csv')
print(f'\nDataset: {len(df):,} neurons')

# Identify errors at 0.80 threshold
y_true = df['ground_truth'].values
y_proba = df['y_proba'].values
y_pred = (y_proba >= THRESHOLD).astype(int)

# Confusion matrix
tp_mask = (y_pred == 1) & (y_true == 1)
fp_mask = (y_pred == 1) & (y_true == 0)
fn_mask = (y_pred == 0) & (y_true == 1)
tn_mask = (y_pred == 0) & (y_true == 0)

n_tp = tp_mask.sum()
n_fp = fp_mask.sum()
n_fn = fn_mask.sum()
n_tn = tn_mask.sum()

print(f'\nConfusion Matrix at threshold {THRESHOLD}:')
print(f'  True Positives:  {n_tp:,}')
print(f'  False Positives: {n_fp:,}')
print(f'  False Negatives: {n_fn:,}')
print(f'  True Negatives:  {n_tn:,}')

# Get top errors
df['error_type'] = 'CORRECT'
df.loc[fp_mask, 'error_type'] = 'FP'
df.loc[fn_mask, 'error_type'] = 'FN'

# Sort FP by probability (highest first - most confident mistakes)
df_fp = df[df['error_type'] == 'FP'].sort_values('y_proba', ascending=False).copy()
df_fp['fp_rank'] = range(1, len(df_fp) + 1)

# Sort FN by probability (lowest first - most confident mistakes)
df_fn = df[df['error_type'] == 'FN'].sort_values('y_proba', ascending=True).copy()
df_fn['fn_rank'] = range(1, len(df_fn) + 1)

print(f'\nTop errors to visualize:')
print(f'  False Positives: {min(N_FP_PAGES * NEURONS_PER_PAGE, len(df_fp))}')
print(f'  False Negatives: {min(N_FN_PAGES * NEURONS_PER_PAGE, len(df_fn))}')

# Output directory
output_dir = Path('ml/results/v8_iter5_080_visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Load raw estimates for trace visualization
def load_estimates(session_name, raw_dir='data/raw_compressed'):
    """Load CaImAn estimates from raw_compressed directory."""
    raw_path = Path(raw_dir)

    patterns = [
        f'{session_name}_estimates*.pickle',
        f'{session_name}_raw*.pickle',
        f'{session_name}_*.pickle',
        f'{session_name}.pickle'
    ]

    for pattern in patterns:
        files = list(raw_path.glob(pattern))
        if files:
            try:
                with open(files[0], 'rb') as f:
                    data = pickle.load(f)

                if isinstance(data, dict):
                    if 'estimates' in data:
                        return data['estimates']
                    elif 'est' in data:
                        return data['est']
                    for key in ['cnmf', 'cnm', 'results']:
                        if key in data and hasattr(data[key], 'estimates'):
                            return data[key].estimates

                if hasattr(data, 'A') and hasattr(data, 'C'):
                    return data
                if hasattr(data, 'estimates'):
                    return data.estimates

                return data
            except Exception as e:
                continue

    return None

def visualize_neuron_page(neurons_df, page_num, error_type, output_dir):
    """Visualize 5 neurons on one page."""
    n_neurons = len(neurons_df)

    fig, axes = plt.subplots(n_neurons, 2, figsize=(14, 3*n_neurons))
    if n_neurons == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(f'{error_type} Errors - Page {page_num} (threshold={THRESHOLD})',
                 fontsize=14, fontweight='bold')

    for idx, (_, neuron) in enumerate(neurons_df.iterrows()):
        session = neuron['session']
        comp_idx = int(neuron['component_idx'])
        prob = neuron['y_proba']
        gt = int(neuron['ground_truth'])
        rank = int(neuron[f'{error_type.lower()}_rank'])

        # Load estimates
        est = load_estimates(session)

        if est is None or not hasattr(est, 'C') or est.C is None:
            axes[idx, 0].text(0.5, 0.5, f'No trace data for {session}',
                            ha='center', va='center')
            axes[idx, 1].text(0.5, 0.5, f'No footprint data for {session}',
                            ha='center', va='center')
            continue

        # Plot trace
        if comp_idx < est.C.shape[0]:
            trace = est.C[comp_idx, :]
            trace_norm = (trace - trace.min()) / (trace.max() - trace.min() + 1e-10)

            axes[idx, 0].plot(trace_norm, 'b-', linewidth=0.8)
            axes[idx, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.3, linewidth=0.8)
            axes[idx, 0].set_ylabel('Normalized', fontsize=8)
            axes[idx, 0].set_xlim(0, len(trace_norm))
            axes[idx, 0].set_ylim(-0.05, 1.05)
            axes[idx, 0].grid(True, alpha=0.2)

            # Title with error info
            gt_label = 'KEEP' if gt == 1 else 'DELETE'
            pred_label = 'KEEP' if prob >= THRESHOLD else 'DELETE'
            title = f'{error_type} #{rank}: {session} comp={comp_idx}\n'
            title += f'GT={gt_label}, Pred={pred_label} (p={prob:.3f})'
            axes[idx, 0].set_title(title, fontsize=9, fontweight='bold')

        # Plot footprint
        if hasattr(est, 'A') and est.A is not None and comp_idx < est.A.shape[1]:
            footprint_flat = est.A[:, comp_idx].toarray().flatten()

            # Determine image dimensions
            if hasattr(est, 'dims'):
                dims = est.dims
            elif hasattr(est, 'd1') and hasattr(est, 'd2'):
                dims = (est.d1, est.d2)
            else:
                dims = (512, 512)

            try:
                footprint = footprint_flat.reshape(dims, order='F')
            except Exception:
                # Fallback: try to infer dims from footprint size
                size = len(footprint_flat)
                dim = int(np.sqrt(size))
                if dim * dim == size:
                    footprint = footprint_flat.reshape((dim, dim), order='F')
                else:
                    footprint = footprint_flat.reshape((512, 512), order='F')

            # Find bounding box
            threshold_val = footprint.max() * 0.01
            if footprint.ndim == 1:
                # Footprint is still 1D, skip visualization
                axes[idx, 1].text(0.5, 0.5, 'Cannot visualize footprint',
                                ha='center', va='center')
                axes[idx, 1].axis('off')
                continue

            rows, cols = np.where(footprint > threshold_val)
            if len(rows) > 0:
                row_min, row_max = rows.min(), rows.max()
                col_min, col_max = cols.min(), cols.max()

                # Add padding
                pad = 10
                row_min = max(0, row_min - pad)
                row_max = min(dims[0], row_max + pad)
                col_min = max(0, col_min - pad)
                col_max = min(dims[1], col_max + pad)

                footprint_crop = footprint[row_min:row_max, col_min:col_max]
            else:
                footprint_crop = footprint

            im = axes[idx, 1].imshow(footprint_crop, cmap='hot', aspect='auto')
            axes[idx, 1].set_title(f'Footprint (cropped)', fontsize=9)
            axes[idx, 1].axis('off')
            plt.colorbar(im, ax=axes[idx, 1], fraction=0.046, pad=0.04)

        # Remove x-axis for all but last row
        if idx < n_neurons - 1:
            axes[idx, 0].set_xticks([])

    # Set x-label on last row
    axes[-1, 0].set_xlabel('Time (frames)', fontsize=8)

    plt.tight_layout()

    # Save
    filename = f'{error_type.lower()}_page_{page_num:02d}.png'
    filepath = output_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath

# Visualize FP errors
print(f'\n{"="*80}')
print(f'VISUALIZING FALSE POSITIVES (predicted KEEP, actually DELETE)')
print('='*80)

n_fp_to_viz = min(N_FP_PAGES * NEURONS_PER_PAGE, len(df_fp))
df_fp_viz = df_fp.head(n_fp_to_viz)

for page in tqdm(range(N_FP_PAGES), desc='Generating FP pages'):
    start_idx = page * NEURONS_PER_PAGE
    end_idx = min(start_idx + NEURONS_PER_PAGE, len(df_fp_viz))

    if start_idx >= len(df_fp_viz):
        break

    neurons_page = df_fp_viz.iloc[start_idx:end_idx]
    filepath = visualize_neuron_page(neurons_page, page + 1, 'FP', output_dir)

print(f'\nFP visualizations saved to: {output_dir}/')

# Visualize FN errors
print(f'\n{"="*80}')
print(f'VISUALIZING FALSE NEGATIVES (predicted DELETE, actually KEEP)')
print('='*80)

n_fn_to_viz = min(N_FN_PAGES * NEURONS_PER_PAGE, len(df_fn))
df_fn_viz = df_fn.head(n_fn_to_viz)

for page in tqdm(range(N_FN_PAGES), desc='Generating FN pages'):
    start_idx = page * NEURONS_PER_PAGE
    end_idx = min(start_idx + NEURONS_PER_PAGE, len(df_fn_viz))

    if start_idx >= len(df_fn_viz):
        break

    neurons_page = df_fn_viz.iloc[start_idx:end_idx]
    filepath = visualize_neuron_page(neurons_page, page + 1, 'FN', output_dir)

print(f'\nFN visualizations saved to: {output_dir}/')

# Save error reports
df_fp.to_csv(output_dir / 'fp_errors_080.csv', index=False)
df_fn.to_csv(output_dir / 'fn_errors_080.csv', index=False)

print(f'\n{"="*80}')
print('SUMMARY')
print('='*80)
print(f'\nGenerated {N_FP_PAGES} FP pages ({n_fp_to_viz} neurons)')
print(f'Generated {N_FN_PAGES} FN pages ({n_fn_to_viz} neurons)')
print(f'\nTotal errors at threshold {THRESHOLD}:')
print(f'  False Positives: {n_fp:,}')
print(f'  False Negatives: {n_fn:,}')
print(f'  Total: {n_fp + n_fn:,}')
print(f'\nFiles saved to: {output_dir}/')
print(f'  - {N_FP_PAGES} FP PNG pages')
print(f'  - {N_FN_PAGES} FN PNG pages')
print(f'  - fp_errors_080.csv (all {n_fp:,} false positives)')
print(f'  - fn_errors_080.csv (all {n_fn:,} false negatives)')
print(f'\n{"="*80}')
