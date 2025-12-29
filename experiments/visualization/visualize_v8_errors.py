"""
Visualize top 100 FP and FN neurons from v8 model.
Shows traces and footprints, 5 neurons per page.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_estimates(session_name, artifacts_dir='data/capcan_validation_99_v8'):
    """Load CaImAn estimates from session artifacts."""
    artifacts_path = Path(artifacts_dir)
    session_dir = artifacts_path / f'capcan_artifacts_{session_name}'

    if not session_dir.exists():
        return None

    # Try to load pickle
    pickle_files = list(session_dir.glob("*_processed.pickle"))
    if not pickle_files:
        pickle_files = list(session_dir.glob("*.pickle"))
    if not pickle_files:
        return None

    try:
        with open(pickle_files[0], 'rb') as f:
            data = pickle.load(f)

        # Handle different pickle formats
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
        print(f'Error loading {session_name}: {e}')
        return None

def get_dims(est):
    """Get image dimensions from estimates object."""
    if hasattr(est, 'dims') and est.dims is not None:
        return est.dims
    if hasattr(est, 'imax') and est.imax is not None:
        return est.imax.shape
    if hasattr(est, 'A'):
        n_pixels = est.A.shape[0]
        for s in [512, 256, 128, 64]:
            if n_pixels == s * s:
                return (s, s)
    return None

def get_neuron_footprint(est, neuron_idx, dims):
    """Extract footprint for a neuron from sparse A matrix."""
    if not hasattr(est, 'A') or est.A is None:
        return None
    if neuron_idx >= est.A.shape[1]:
        return None

    A = est.A
    if hasattr(A, 'toarray'):
        a = A[:, neuron_idx].toarray().flatten()
    elif hasattr(A, 'todense'):
        a = np.asarray(A[:, neuron_idx].todense()).flatten()
    else:
        a = np.asarray(A[:, neuron_idx]).flatten()

    if dims is None:
        return None
    return a.reshape(dims, order='F')

def plot_neuron(ax_trace, ax_foot, est, neuron_idx, dims, title, proba, label):
    """Plot trace and footprint for a single neuron."""
    # Get trace
    if hasattr(est, 'C') and est.C is not None and neuron_idx < est.C.shape[0]:
        trace = est.C[neuron_idx]
    else:
        trace = np.zeros(100)

    # Normalize trace for display
    trace_range = trace.max() - trace.min()
    if trace_range > 1e-10:
        trace_norm = (trace - trace.min()) / trace_range
    else:
        trace_norm = np.zeros_like(trace)

    # Plot trace
    ax_trace.plot(trace_norm, 'b-', linewidth=0.8, alpha=0.8)
    ax_trace.set_xlim(0, len(trace))
    ax_trace.set_ylim(-0.1, 1.1)
    ax_trace.set_ylabel('dF/F (norm)', fontsize=9)
    ax_trace.set_title(f"{title}\nProb={proba:.3f}, GT={'KEEP' if label else 'DELETE'}",
                       fontsize=10, fontweight='bold')
    ax_trace.tick_params(labelsize=8)
    ax_trace.grid(True, alpha=0.3)

    # Get and plot footprint
    footprint = get_neuron_footprint(est, neuron_idx, dims)
    if footprint is not None:
        try:
            # Crop to non-zero region with padding
            nz = np.where(footprint > 0)
            if len(nz[0]) > 0:
                pad = 5
                y_min = max(0, nz[0].min() - pad)
                y_max = min(dims[0], nz[0].max() + pad)
                x_min = max(0, nz[1].min() - pad)
                x_max = min(dims[1], nz[1].max() + pad)
                footprint_crop = footprint[y_min:y_max, x_min:x_max]
            else:
                footprint_crop = footprint

            ax_foot.imshow(footprint_crop, cmap='hot', aspect='equal')
            ax_foot.set_title(f'Area={int(np.sum(footprint > 0))}px', fontsize=9)
            ax_foot.axis('off')
        except Exception as e:
            ax_foot.text(0.5, 0.5, f'Error: {str(e)[:20]}',
                        ha='center', va='center', transform=ax_foot.transAxes, fontsize=8)
            ax_foot.axis('off')
    else:
        ax_foot.text(0.5, 0.5, 'N/A', ha='center', va='center',
                    transform=ax_foot.transAxes, fontsize=10)
        ax_foot.axis('off')

def visualize_errors(
    fp_csv='ml/results/v8_top100_false_positives.csv',
    fn_csv='ml/results/v8_top100_false_negatives.csv',
    artifacts_dir='data/capcan_validation_99_v8',
    output_dir='ml/results/v8_error_visualizations',
    neurons_per_page=5
):
    """Visualize top FP and FN neurons."""
    print('='*80)
    print('VISUALIZING v8 MODEL ERRORS')
    print('='*80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    fp_dir = output_path / 'false_positives'
    fn_dir = output_path / 'false_negatives'
    fp_dir.mkdir(exist_ok=True)
    fn_dir.mkdir(exist_ok=True)

    # Load error reports
    df_fp = pd.read_csv(fp_csv)
    df_fn = pd.read_csv(fn_csv)

    print(f'\nLoaded:')
    print(f'  False Positives: {len(df_fp)}')
    print(f'  False Negatives: {len(df_fn)}')

    # Visualize FALSE POSITIVES
    print(f'\n{"="*80}')
    print('VISUALIZING FALSE POSITIVES (model says KEEP, should DELETE)')
    print('='*80)

    n_fp_pages = (len(df_fp) + neurons_per_page - 1) // neurons_per_page

    for page in range(n_fp_pages):
        start_idx = page * neurons_per_page
        end_idx = min(start_idx + neurons_per_page, len(df_fp))
        page_neurons = df_fp.iloc[start_idx:end_idx]

        print(f'\nPage {page+1}/{n_fp_pages}: neurons {start_idx+1}-{end_idx}')

        # Create figure (2 rows per neuron: trace + footprint)
        fig, axes = plt.subplots(
            neurons_per_page * 2, 1,
            figsize=(14, neurons_per_page * 3),
            gridspec_kw={'hspace': 0.4}
        )

        if neurons_per_page == 1:
            axes = [axes]

        fig.suptitle(
            f'FALSE POSITIVES (model says KEEP, should DELETE)\n'
            f'Page {page+1}/{n_fp_pages} - Neurons {start_idx+1}-{end_idx}',
            fontsize=14, fontweight='bold', y=0.995
        )

        for i, (idx, row) in enumerate(page_neurons.iterrows()):
            session = row['session']
            comp_idx = int(row['component_idx'])
            proba = row['y_proba']
            label = int(row['ground_truth'])

            # Load estimates
            est = load_estimates(session, artifacts_dir)

            if est is None:
                # Show error message
                axes[i*2].text(0.5, 0.5, f'{session} - No estimates found',
                              ha='center', va='center', fontsize=10)
                axes[i*2].axis('off')
                axes[i*2 + 1].axis('off')
                continue

            dims = get_dims(est)
            if dims is None:
                axes[i*2].text(0.5, 0.5, f'{session} - No dims',
                              ha='center', va='center', fontsize=10)
                axes[i*2].axis('off')
                axes[i*2 + 1].axis('off')
                continue

            title = f"#{start_idx+i+1}: {session} (component {comp_idx})"
            plot_neuron(axes[i*2], axes[i*2 + 1], est, comp_idx, dims, title, proba, label)

        # Hide unused subplots
        for i in range(len(page_neurons), neurons_per_page):
            axes[i*2].axis('off')
            axes[i*2 + 1].axis('off')

        plt.tight_layout()
        output_file = fp_dir / f'FP_page_{page+1:02d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {output_file}')

    # Visualize FALSE NEGATIVES
    print(f'\n{"="*80}')
    print('VISUALIZING FALSE NEGATIVES (model says DELETE, should KEEP)')
    print('='*80)

    n_fn_pages = (len(df_fn) + neurons_per_page - 1) // neurons_per_page

    for page in range(n_fn_pages):
        start_idx = page * neurons_per_page
        end_idx = min(start_idx + neurons_per_page, len(df_fn))
        page_neurons = df_fn.iloc[start_idx:end_idx]

        print(f'\nPage {page+1}/{n_fn_pages}: neurons {start_idx+1}-{end_idx}')

        # Create figure
        fig, axes = plt.subplots(
            neurons_per_page * 2, 1,
            figsize=(14, neurons_per_page * 3),
            gridspec_kw={'hspace': 0.4}
        )

        if neurons_per_page == 1:
            axes = [axes]

        fig.suptitle(
            f'FALSE NEGATIVES (model says DELETE, should KEEP)\n'
            f'Page {page+1}/{n_fn_pages} - Neurons {start_idx+1}-{end_idx}',
            fontsize=14, fontweight='bold', y=0.995
        )

        for i, (idx, row) in enumerate(page_neurons.iterrows()):
            session = row['session']
            comp_idx = int(row['component_idx'])
            proba = row['y_proba']
            label = int(row['ground_truth'])

            # Load estimates
            est = load_estimates(session, artifacts_dir)

            if est is None:
                axes[i*2].text(0.5, 0.5, f'{session} - No estimates found',
                              ha='center', va='center', fontsize=10)
                axes[i*2].axis('off')
                axes[i*2 + 1].axis('off')
                continue

            dims = get_dims(est)
            if dims is None:
                axes[i*2].text(0.5, 0.5, f'{session} - No dims',
                              ha='center', va='center', fontsize=10)
                axes[i*2].axis('off')
                axes[i*2 + 1].axis('off')
                continue

            title = f"#{start_idx+i+1}: {session} (component {comp_idx})"
            plot_neuron(axes[i*2], axes[i*2 + 1], est, comp_idx, dims, title, proba, label)

        # Hide unused subplots
        for i in range(len(page_neurons), neurons_per_page):
            axes[i*2].axis('off')
            axes[i*2 + 1].axis('off')

        plt.tight_layout()
        output_file = fn_dir / f'FN_page_{page+1:02d}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {output_file}')

    print(f'\n{"="*80}')
    print('SUMMARY')
    print('='*80)
    print(f'\nCreated {n_fp_pages} pages of False Positives (5 neurons each)')
    print(f'Created {n_fn_pages} pages of False Negatives (5 neurons each)')
    print(f'\nOutput directory: {output_path}')
    print(f'  - false_positives/: {n_fp_pages} PNG files')
    print(f'  - false_negatives/: {n_fn_pages} PNG files')
    print(f'\nDone!')

if __name__ == "__main__":
    visualize_errors(
        fp_csv='ml/results/v8_top100_false_positives.csv',
        fn_csv='ml/results/v8_top100_false_negatives.csv',
        artifacts_dir='data/capcan_validation_99_v8',
        output_dir='ml/results/v8_error_visualizations',
        neurons_per_page=5
    )
