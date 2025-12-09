"""
Compression utilities for CaImAn estimates objects.

Can be used as a module:
    from estimates_compression import compress_estimates_ultra_lightweight
    est, savings, total_saved = compress_estimates_ultra_lightweight(est)

Or as a CLI for batch processing:
    python estimates_compression.py
    python estimates_compression.py --keep-bad

Compression strategy:
1. REMOVE bad component data (optional, enabled by default)
2. Convert YrA (residuals) to float32
3. Convert all float64 arrays to float32
4. Convert S (spikes) to sparse matrix format
5. Remove intermediate computation results
"""
import pickle
import numpy as np
from pathlib import Path
from scipy import sparse
import time


def compress_estimates_ultra_lightweight(est, remove_bad_components=True):
    """
    Compress estimates object for ultra-lightweight storage.

    Args:
        est: CaImAn estimates object
        remove_bad_components: If True, removes data for idx_components_bad (default: True)

    Returns:
        (compressed_est, savings_dict, total_saved)
    """
    savings = {}
    total_saved = 0

    # 0. REMOVE BAD COMPONENT DATA (if requested)
    if remove_bad_components and hasattr(est, 'idx_components_bad') and hasattr(est, 'idx_components'):
        if len(est.idx_components_bad) > 0:
            # Get indices to keep (good components only)
            good_indices = est.idx_components
            n_bad = len(est.idx_components_bad)

            # Remove rows from C, S, YrA for bad components
            if hasattr(est, 'C') and est.C is not None:
                original_size = est.C.nbytes / (1024**2)
                est.C = est.C[good_indices, :]
                saved = original_size - est.C.nbytes / (1024**2)
                savings['C_bad_removed'] = saved
                total_saved += saved

            if hasattr(est, 'S') and est.S is not None and isinstance(est.S, np.ndarray):
                original_size = est.S.nbytes / (1024**2)
                est.S = est.S[good_indices, :]
                saved = original_size - est.S.nbytes / (1024**2)
                savings['S_bad_removed'] = saved
                total_saved += saved

            if hasattr(est, 'YrA') and est.YrA is not None:
                original_size = est.YrA.nbytes / (1024**2)
                est.YrA = est.YrA[good_indices, :]
                saved = original_size - est.YrA.nbytes / (1024**2)
                savings['YrA_bad_removed'] = saved
                total_saved += saved

            # Remove columns from A for bad components
            if hasattr(est, 'A') and est.A is not None:
                original_size = (est.A.data.nbytes + est.A.indices.nbytes +
                               est.A.indptr.nbytes) / (1024**2)
                est.A = est.A[:, good_indices]
                new_size = (est.A.data.nbytes + est.A.indices.nbytes +
                          est.A.indptr.nbytes) / (1024**2)
                saved = original_size - new_size
                savings['A_bad_removed'] = saved
                total_saved += saved

            # Remove other per-component arrays
            # CRITICAL: Use sequential indices [0,1,2,...] not component IDs
            # because we already removed bad components from C, S, YrA above
            for attr in ['bl', 'c1', 'neurons_sn', 'SNR_comp', 'r_values']:
                if hasattr(est, attr):
                    arr = getattr(est, attr)
                    if isinstance(arr, np.ndarray) and len(arr) > len(good_indices):
                        # Take first N elements where N = number of good components
                        setattr(est, attr, arr[:len(good_indices)])

            # Update g (can be list or numpy array)
            # CRITICAL: g contains AR parameters, one per neuron
            # Must use sequential indices after bad component removal
            if hasattr(est, 'g'):
                if isinstance(est.g, list):
                    # g is list: take first N elements
                    est.g = [est.g[i] for i in range(len(good_indices))]
                elif isinstance(est.g, np.ndarray):
                    # g is numpy array: slice to first N elements
                    est.g = est.g[:len(good_indices)]

            # Preserve original bad component indices for record-keeping
            # Store in metadata BEFORE updating idx_components
            est.idx_components_bad_original = est.idx_components_bad.copy()

            # Update idx_components to be 0-indexed after removal
            # After removal, all remaining components are "good"
            est.idx_components = np.arange(len(good_indices), dtype=int)
            est.idx_components_bad = np.array([], dtype=int)  # All bad data removed

            print(f'  Removed {n_bad} bad components')

    # 1. Convert YrA to float32 (KEEP - required for manual_merge!)
    if hasattr(est, 'YrA') and est.YrA is not None:
        if isinstance(est.YrA, np.ndarray) and est.YrA.dtype == np.float64:
            saved = est.YrA.nbytes / 2 / (1024**2)
            est.YrA = est.YrA.astype(np.float32)
            savings['YrA_f64->f32'] = saved
            total_saved += saved

    # 2. Convert S to sparse matrix (if dense and sparse enough)
    if hasattr(est, 'S') and isinstance(est.S, np.ndarray):
        sparsity = np.sum(est.S == 0) / est.S.size
        original_size = est.S.nbytes / (1024**2)

        if sparsity > 0.5:  # Only convert if >50% sparse
            S_sparse = sparse.csr_matrix(est.S.astype(np.float32))
            sparse_size = (S_sparse.data.nbytes + S_sparse.indices.nbytes +
                          S_sparse.indptr.nbytes) / (1024**2)
            saved = original_size - sparse_size
            est.S = S_sparse
            savings['S_to_sparse'] = saved
            total_saved += saved
        else:
            saved = est.S.nbytes / 2 / (1024**2)
            est.S = est.S.astype(np.float32)
            savings['S_f64->f32'] = saved
            total_saved += saved

    # 3. Convert other float64 arrays to float32
    float64_attrs = []
    for attr_name in dir(est):
        if attr_name.startswith('_') or attr_name in ['S', 'YrA']:
            continue
        try:
            attr = getattr(est, attr_name)
            if isinstance(attr, np.ndarray) and attr.dtype == np.float64:
                float64_attrs.append(attr_name)
        except:
            continue

    for attr_name in float64_attrs:
        arr = getattr(est, attr_name)
        saved = arr.nbytes / 2 / (1024**2)
        setattr(est, attr_name, arr.astype(np.float32))
        savings[f'{attr_name}_f64->f32'] = saved
        total_saved += saved

    # 4. Remove intermediate computation results
    removable_attrs = [
        'AtA', 'AtY_buf', 'CC', 'CY', 'R', 'Yr_buf', 'rho_buf',
        'noisyC', 'OASISinstances', 'Ab_dense', 'A_thr'
    ]

    for attr_name in removable_attrs:
        if hasattr(est, attr_name):
            attr = getattr(est, attr_name)
            if attr is not None:
                if isinstance(attr, np.ndarray):
                    saved = attr.nbytes / (1024**2)
                    savings[f'{attr_name}_removed'] = saved
                    total_saved += saved
                setattr(est, attr_name, None)

    return est, savings, total_saved


def process_file(input_path, output_path, remove_bad_components=True):
    """Process a single estimates file."""
    try:
        with open(input_path, 'rb') as f:
            est = pickle.load(f)

        original_size = input_path.stat().st_size / (1024**2)

        # Compress
        est_compressed, savings, total_saved = compress_estimates_ultra_lightweight(
            est, remove_bad_components=remove_bad_components
        )

        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(est_compressed, f, protocol=pickle.HIGHEST_PROTOCOL)

        new_size = output_path.stat().st_size / (1024**2)
        actual_saved = original_size - new_size
        compression_ratio = (1 - new_size / original_size) * 100

        return {
            'success': True,
            'original_size': original_size,
            'new_size': new_size,
            'saved': actual_saved,
            'ratio': compression_ratio,
            'savings': savings
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


def process_directory(input_dir, output_dir, pattern="*.pickle", remove_bad_components=True):
    """Process all pickle files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    files = [f for f in sorted(input_dir.glob(pattern))
             if '.compressed' not in f.name and '.lightweight' not in f.name and f.is_file()]

    if not files:
        print(f"No pickle files found in {input_dir}")
        return

    print(f"\nProcessing: {input_dir}")
    print(f"Output to: {output_dir}")
    print(f"Files to process: {len(files)}")
    print(f"Remove bad components: {remove_bad_components}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)

    total_original = 0
    total_compressed = 0
    successful = 0
    failed = 0
    start_time = time.time()

    for i, input_path in enumerate(files, 1):
        output_path = output_dir / input_path.name

        print(f"\n[{i}/{len(files)}] {input_path.name}")

        result = process_file(input_path, output_path, remove_bad_components=remove_bad_components)

        if result['success']:
            total_original += result['original_size']
            total_compressed += result['new_size']
            successful += 1

            print(f"  Original: {result['original_size']:.2f} MB")
            print(f"  Compressed: {result['new_size']:.2f} MB")
            print(f"  Saved: {result['saved']:.2f} MB ({result['ratio']:.1f}%)")

            top_savings = sorted(result['savings'].items(),
                               key=lambda x: x[1], reverse=True)[:3]
            if top_savings:
                print(f"  Top savings: {', '.join([f'{k}({v:.0f}MB)' for k,v in top_savings])}")

        else:
            failed += 1
            print(f"  FAILED: {result['error']}")
            total_original += input_path.stat().st_size / (1024**2)
            total_compressed += input_path.stat().st_size / (1024**2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"DIRECTORY COMPLETE: {input_dir.name}")
    print("-" * 80)
    print(f"  Files processed: {len(files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  Original size: {total_original:.2f} MB ({total_original/1024:.2f} GB)")
    print(f"  Compressed size: {total_compressed:.2f} MB ({total_compressed/1024:.2f} GB)")
    print(f"  Total saved: {total_original - total_compressed:.2f} MB")
    print(f"  Compression ratio: {(1 - total_compressed/total_original)*100:.1f}%")

    return {
        'files': len(files),
        'successful': successful,
        'failed': failed,
        'original_size': total_original,
        'compressed_size': total_compressed,
        'time': elapsed
    }


def main():
    """Main processing function."""
    import sys

    # Parse command line arguments
    remove_bad = '--keep-bad' not in sys.argv

    print("ULTRA-LIGHTWEIGHT ESTIMATES CREATION")
    print("=" * 80)
    print("This will process all pickle files and create ultra-lightweight versions:")
    print("  - data/raw/ -> data/raw_ultra_lightweight/")
    print("  - data/final/ -> data/final_ultra_lightweight/")
    print("\nCompression strategy:")
    print(f"  1. Remove bad component data: {remove_bad}")
    print("  2. Convert YrA to float32 (keep for manual_merge)")
    print("  3. Convert S to sparse matrix (if >50% sparse)")
    print("  4. Convert float64 -> float32")
    print("  5. Remove intermediate computation results")
    print("=" * 80)

    overall_start = time.time()

    results = {}

    # Process both directories
    results['raw'] = process_directory(
        input_dir='data/raw',
        output_dir='data/raw_ultra_lightweight',
        pattern='*.pickle',
        remove_bad_components=remove_bad
    )

    results['final'] = process_directory(
        input_dir='data/final',
        output_dir='data/final_ultra_lightweight',
        pattern='*.pickle',
        remove_bad_components=remove_bad
    )

    # Overall summary
    overall_elapsed = time.time() - overall_start
    total_files = results['raw']['files'] + results['final']['files']
    total_successful = results['raw']['successful'] + results['final']['successful']
    total_failed = results['raw']['failed'] + results['final']['failed']
    total_original = results['raw']['original_size'] + results['final']['original_size']
    total_compressed = results['raw']['compressed_size'] + results['final']['compressed_size']

    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {total_files}")
    print(f"Successful: {total_successful}")
    print(f"Failed: {total_failed}")
    print(f"Total time: {overall_elapsed/60:.1f} minutes ({overall_elapsed/3600:.2f} hours)")
    print()
    print(f"Original total size: {total_original/1024:.2f} GB")
    print(f"Compressed total size: {total_compressed/1024:.2f} GB")
    print(f"Total space saved: {(total_original - total_compressed)/1024:.2f} GB")
    print(f"Overall compression ratio: {(1 - total_compressed/total_original)*100:.1f}%")
    print()
    print("Output directories:")
    print(f"  - data/raw_ultra_lightweight/")
    print(f"  - data/final_ultra_lightweight/")
    print()
    print("Usage note:")
    print(f"  Run with --keep-bad to preserve bad component data")
    print(f"  Current mode: {'Keep bad components' if not remove_bad else 'Remove bad components (default)'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
