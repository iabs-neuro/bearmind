"""
Universal error visualization script.

Visualizes top FP and FN errors from any iteration for manual review.
Shows traces and spatial footprints, 5 neurons per page.

Usage:
    python ml/visualize_errors.py --fp ml/ebm_v9_iter1/top100_fp.csv --fn ml/ebm_v9_iter1/top100_fn.csv --output ml/ebm_v9_iter1/visualizations
    python ml/visualize_errors.py --fp ml/ebm_v9_iter1/top100_fp.csv --output ml/ebm_v9_iter1/fp_viz --raw-dir data/raw_compressed
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def load_estimates(session_name, raw_dir='data/raw_compressed'):
    """Load CaImAn estimates from raw_compressed or LNOF directory."""
    # Try multiple search locations
    search_paths = [
        (Path(raw_dir), [
            f'{session_name}_estimates*.pickle',
            f'{session_name}_*.pickle',
            f'{session_name}.pickle'
        ])
    ]

    # If LNOF session, also check data/LNOF/inspection_artifacts_*
    if session_name.startswith('LNOF'):
        lnof_base = Path('data/LNOF')
        if lnof_base.exists():
            # Look for inspection_artifacts directories
            artifact_dirs = list(lnof_base.glob(f'inspection_artifacts_{session_name}_*'))
            for artifact_dir in artifact_dirs:
                search_paths.append((
                    artifact_dir,
                    [f'{session_name}_*_processed.pickle', f'*_processed.pickle']
                ))

    # Try each search path
    for search_dir, patterns in search_paths:
        if not search_dir.exists():
            continue

        for pattern in patterns:
            files = list(search_dir.glob(pattern))
            if files:
                try:
                    with open(files[0], 'rb') as f:
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
                    continue

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


def get_neuron_footprint(est, neuron_idx, dims, crop=True, padding=20):
    """Extract footprint for a neuron from sparse A matrix.

    Parameters
    ----------
    est : estimates object
        CaImAn estimates
    neuron_idx : int
        Neuron index
    dims : tuple
        Image dimensions
    crop : bool
        If True, crop to bounding box of non-zero pixels
    padding : int
        Pixels to add around bounding box
    """
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

    footprint = a.reshape(dims, order='F')

    if not crop:
        return footprint

    # Find bounding box of non-zero pixels
    nonzero = np.where(footprint > footprint.max() * 0.01)  # Threshold at 1% of max
    if len(nonzero[0]) == 0:
        return footprint  # Return full if no significant pixels

    y_min, y_max = nonzero[0].min(), nonzero[0].max()
    x_min, x_max = nonzero[1].min(), nonzero[1].max()

    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(dims[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(dims[1], x_max + padding)

    # Crop to bounding box
    cropped = footprint[y_min:y_max+1, x_min:x_max+1]

    return cropped


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
    ax_trace.set_title(f'{title} | p={proba:.3f} | GT={label}', fontsize=9)
    ax_trace.set_xlabel('Frame', fontsize=8)
    ax_trace.set_ylabel('Norm. Fluorescence', fontsize=8)
    ax_trace.grid(True, alpha=0.3)
    ax_trace.tick_params(labelsize=7)

    # Plot footprint
    footprint = get_neuron_footprint(est, neuron_idx, dims, crop=True, padding=20)
    if footprint is not None:
        ax_foot.imshow(footprint, cmap='hot', interpolation='nearest', aspect='equal')
        ax_foot.set_title(f'Footprint #{neuron_idx} ({footprint.shape[0]}x{footprint.shape[1]}px)', fontsize=9)
    else:
        ax_foot.text(0.5, 0.5, 'Footprint N/A', ha='center', va='center',
                    transform=ax_foot.transAxes, fontsize=10)
    ax_foot.axis('off')


def visualize_errors(
    fp_csv=None,
    fn_csv=None,
    output_dir='visualizations',
    raw_dir='data/raw_compressed',
    neurons_per_page=5,
    max_pages=None
):
    """
    Visualize error cases for manual review.

    Parameters
    ----------
    fp_csv : str, optional
        Path to false positive error CSV
    fn_csv : str, optional
        Path to false negative error CSV
    output_dir : str
        Output directory for visualization PDFs
    raw_dir : str
        Directory containing raw estimates pickle files
    neurons_per_page : int
        Number of neurons per page
    max_pages : int, optional
        Maximum number of pages to generate (for testing)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print('='*80)
    print('ERROR VISUALIZATION')
    print('='*80)
    print(f'Output: {output_dir}')
    print(f'Raw data: {raw_dir}')

    # Process FP errors
    if fp_csv:
        print(f'\n{"="*80}')
        print('VISUALIZING FALSE POSITIVES')
        print('='*80)

        df_fp = pd.read_csv(fp_csv)
        print(f'Loaded {len(df_fp)} FP errors from {fp_csv}')

        # Determine session column
        if 'session_name' in df_fp.columns:
            session_col = 'session_name'
        elif 'session' in df_fp.columns:
            session_col = 'session'
        else:
            raise ValueError('Error CSV must have "session_name" or "session" column')

        sessions = df_fp[session_col].unique()
        n_pages = (len(df_fp) + neurons_per_page - 1) // neurons_per_page
        if max_pages:
            n_pages = min(n_pages, max_pages)

        print(f'Sessions: {len(sessions)}')
        print(f'Generating {n_pages} pages ({neurons_per_page} neurons/page)...')

        neurons_plotted = 0
        for page in range(n_pages):
            start_idx = page * neurons_per_page
            end_idx = min(start_idx + neurons_per_page, len(df_fp))
            page_df = df_fp.iloc[start_idx:end_idx]

            fig, axes = plt.subplots(neurons_per_page, 2, figsize=(12, 2.5 * neurons_per_page))
            if neurons_per_page == 1:
                axes = axes.reshape(1, -1)

            for i, (idx, row) in enumerate(page_df.iterrows()):
                session = row[session_col]
                comp_idx = int(row['component_idx'])
                proba = row['y_proba']
                gt = row['ground_truth']

                # Load estimates
                est = load_estimates(session, raw_dir)
                if est is None:
                    axes[i, 0].text(0.5, 0.5, f'Session {session} not found',
                                   ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                   transform=axes[i, 1].transAxes)
                    axes[i, 0].axis('off')
                    axes[i, 1].axis('off')
                    continue

                dims = get_dims(est)
                title = f'FP #{idx-df_fp.index[0]+1} | {session}'
                plot_neuron(axes[i, 0], axes[i, 1], est, comp_idx, dims, title, proba, gt)
                neurons_plotted += 1

            # Hide unused subplots
            for i in range(len(page_df), neurons_per_page):
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')

            plt.tight_layout()
            output_file = output_path / f'fp_page_{page+1:03d}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

            if (page + 1) % 5 == 0:
                print(f'  Generated {page+1}/{n_pages} pages ({neurons_plotted} neurons)')

        print(f'FP visualizations saved: {neurons_plotted} neurons in {n_pages} pages')

    # Process FN errors
    if fn_csv:
        print(f'\n{"="*80}')
        print('VISUALIZING FALSE NEGATIVES')
        print('='*80)

        df_fn = pd.read_csv(fn_csv)
        print(f'Loaded {len(df_fn)} FN errors from {fn_csv}')

        # Determine session column
        if 'session_name' in df_fn.columns:
            session_col = 'session_name'
        elif 'session' in df_fn.columns:
            session_col = 'session'
        else:
            raise ValueError('Error CSV must have "session_name" or "session" column')

        sessions = df_fn[session_col].unique()
        n_pages = (len(df_fn) + neurons_per_page - 1) // neurons_per_page
        if max_pages:
            n_pages = min(n_pages, max_pages)

        print(f'Sessions: {len(sessions)}')
        print(f'Generating {n_pages} pages ({neurons_per_page} neurons/page)...')

        neurons_plotted = 0
        for page in range(n_pages):
            start_idx = page * neurons_per_page
            end_idx = min(start_idx + neurons_per_page, len(df_fn))
            page_df = df_fn.iloc[start_idx:end_idx]

            fig, axes = plt.subplots(neurons_per_page, 2, figsize=(12, 2.5 * neurons_per_page))
            if neurons_per_page == 1:
                axes = axes.reshape(1, -1)

            for i, (idx, row) in enumerate(page_df.iterrows()):
                session = row[session_col]
                comp_idx = int(row['component_idx'])
                proba = row['y_proba']
                gt = row['ground_truth']

                # Load estimates
                est = load_estimates(session, raw_dir)
                if est is None:
                    axes[i, 0].text(0.5, 0.5, f'Session {session} not found',
                                   ha='center', va='center', transform=axes[i, 0].transAxes)
                    axes[i, 1].text(0.5, 0.5, 'N/A', ha='center', va='center',
                                   transform=axes[i, 1].transAxes)
                    axes[i, 0].axis('off')
                    axes[i, 1].axis('off')
                    continue

                dims = get_dims(est)
                title = f'FN #{idx-df_fn.index[0]+1} | {session}'
                plot_neuron(axes[i, 0], axes[i, 1], est, comp_idx, dims, title, proba, gt)
                neurons_plotted += 1

            # Hide unused subplots
            for i in range(len(page_df), neurons_per_page):
                axes[i, 0].axis('off')
                axes[i, 1].axis('off')

            plt.tight_layout()
            output_file = output_path / f'fn_page_{page+1:03d}.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

            if (page + 1) % 5 == 0:
                print(f'  Generated {page+1}/{n_pages} pages ({neurons_plotted} neurons)')

        print(f'FN visualizations saved: {neurons_plotted} neurons in {n_pages} pages')

    print(f'\n{"="*80}')
    print('VISUALIZATION COMPLETE')
    print('='*80)
    print(f'\nReview visualizations in: {output_dir}')
    print('\nNext steps:')
    print('1. Review each error and classify as REAL or FAKE')
    print('2. For FP: Run spatial analysis to identify MERGE cases')
    print('3. Create correction list')
    print('4. Apply corrections using ml/apply_corrections.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Universal error visualization')
    parser.add_argument('--fp', default=None, help='Path to FP error CSV')
    parser.add_argument('--fn', default=None, help='Path to FN error CSV')
    parser.add_argument('--output', default='visualizations', help='Output directory')
    parser.add_argument('--raw-dir', default='data/raw_compressed', help='Raw estimates directory')
    parser.add_argument('--per-page', type=int, default=5, help='Neurons per page')
    parser.add_argument('--max-pages', type=int, default=None, help='Max pages (for testing)')

    args = parser.parse_args()

    if not args.fp and not args.fn:
        parser.error('At least one of --fp or --fn must be specified')

    visualize_errors(
        fp_csv=args.fp,
        fn_csv=args.fn,
        output_dir=args.output,
        raw_dir=args.raw_dir,
        neurons_per_page=args.per_page,
        max_pages=args.max_pages
    )
