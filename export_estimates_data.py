"""
Export processed estimates data to NPZ and JSON files.

Usage:
    python export_estimates_data.py path/to/session_processed.pickle
    python export_estimates_data.py path/to/session_processed.pickle --incorporate-feedback
    python export_estimates_data.py path/to/session_processed.pickle -o output_dir

Output:
    {session_name}_data.npz - numpy arrays (C, asp, reconstructions, component_indices)
    {session_name}_metadata.json - all metadata as JSON
"""

import argparse
import json
import pickle
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# FPS lookup - avoid heavy import chain from ae_launch
FPS_TABLE_PATH = Path(__file__).parent / 'fps_data.csv'


def get_fps_from_table(session_name: str, default_fps: int = None) -> int | None:
    """Look up FPS for a session from fps_data.csv."""
    if not FPS_TABLE_PATH.exists():
        return default_fps

    try:
        fps_df = pd.read_csv(FPS_TABLE_PATH, sep=';')
    except Exception:
        return default_fps

    # Exact match
    if session_name in fps_df['Filename'].values:
        return round(fps_df[fps_df['Filename'] == session_name]['FPS'].values[0])

    # Extract session identifier pattern
    pattern = r'([A-Z0-9]+_[A-Z]\d+_\d[A-Z](?:_\d[A-Z])?)'
    match = re.search(pattern, session_name)
    if match:
        session_id = match.group(1)
        if session_id in fps_df['Filename'].values:
            return round(fps_df[fps_df['Filename'] == session_id]['FPS'].values[0])

        # Try without trial suffix
        base_id = '_'.join(session_id.split('_')[:3])
        if base_id in fps_df['Filename'].values:
            return round(fps_df[fps_df['Filename'] == base_id]['FPS'].values[0])

    return default_fps


def load_estimates(path: Path):
    """Load estimates from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_session_name(est_or_path) -> str:
    """Extract session identifier (e.g., LNOF_J01_1D) from estimates or path."""
    if hasattr(est_or_path, 'name'):
        name = est_or_path.name
    else:
        name = str(est_or_path)

    # Pattern: CODE_MOUSEID_DAY (e.g., LNOF_J01_1D, NOF_H03_2D)
    match = re.search(r'([A-Z0-9]+_[A-Z]\d+_\d[A-Z])', name)
    if match:
        return match.group(1)

    # Fallback to stem without common suffixes
    stem = Path(name).stem
    for suffix in ['_processed', '_estimates', '_data']:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    return stem


def get_fps(session_name: str) -> float:
    """Get FPS from fps_data.csv. Raises error if not found."""
    fps = get_fps_from_table(session_name, default_fps=None)
    if fps is None:
        raise ValueError(
            f"FPS not found for session '{session_name}' in fps_data.csv. "
            f"Please add this session to the FPS table."
        )
    return float(fps)


def find_feedback_file(estimates_path: Path, session_name: str) -> Path | None:
    """Find feedback CSV file.

    Search locations (in order):
    1. Same folder as estimates (new experiments - feedback in artifacts folder)
    2. Parent folder (legacy - feedback in output/ folder)
    """
    parent = estimates_path.parent

    # 1. Same folder as estimates (new behavior)
    direct_match = parent / f'{session_name}_feedback.csv'
    if direct_match.exists():
        return direct_match

    # Glob for variations in same folder
    for path in parent.glob(f'*{session_name}*feedback*.csv'):
        if path.exists():
            return path

    # 2. Parent folder (legacy behavior)
    grandparent = parent.parent
    legacy_match = grandparent / f'{session_name}_feedback.csv'
    if legacy_match.exists():
        return legacy_match

    for path in grandparent.glob(f'*{session_name}*feedback*.csv'):
        if path.exists():
            return path

    return None


def load_feedback(feedback_path: Path) -> pd.DataFrame:
    """Load feedback CSV file."""
    return pd.read_csv(feedback_path)


def apply_feedback(idx_components: np.ndarray, feedback_df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Apply feedback corrections to component indices.

    Args:
        idx_components: Original good component indices
        feedback_df: DataFrame with neuron_idx and feedback_type columns

    Returns:
        Corrected component indices and summary dict
    """
    idx_set = set(idx_components.tolist())

    fp_indices = feedback_df[feedback_df['feedback_type'] == 'FP']['neuron_idx'].tolist()
    fn_indices = feedback_df[feedback_df['feedback_type'] == 'FN']['neuron_idx'].tolist()

    # Apply corrections
    n_fp_removed = 0
    n_fn_added = 0

    for idx in fp_indices:
        if idx in idx_set:
            idx_set.remove(idx)
            n_fp_removed += 1

    for idx in fn_indices:
        if idx not in idx_set:
            idx_set.add(idx)
            n_fn_added += 1

    corrected = np.array(sorted(idx_set))

    summary = {
        'n_fp_removed': n_fp_removed,
        'n_fn_added': n_fn_added,
        'n_total_corrections': n_fp_removed + n_fn_added
    }

    return corrected, summary


def extract_data(est, component_indices: np.ndarray) -> dict:
    """Extract data arrays for specified components."""
    data = {}

    # Calcium traces
    data['C'] = est.C[component_indices].astype(np.float32)
    data['component_indices'] = component_indices.astype(np.int32)

    # ASP (amplitude spikes) - if cached
    if hasattr(est, 'asp_cache') and est.asp_cache:
        asp_arrays = []
        for idx in component_indices:
            if idx in est.asp_cache:
                asp_arrays.append(est.asp_cache[idx])
            else:
                # Missing ASP for this component - use zeros
                asp_arrays.append(np.zeros(est.C.shape[1], dtype=np.float32))
        if asp_arrays:
            data['asp'] = np.array(asp_arrays, dtype=np.float32)

    # Reconstructions - if available
    if hasattr(est, 'reconstructions') and est.reconstructions:
        recon_arrays = []
        for idx in component_indices:
            if idx in est.reconstructions:
                recon_arrays.append(est.reconstructions[idx])
            else:
                recon_arrays.append(np.zeros(est.C.shape[1], dtype=np.float32))
        if recon_arrays:
            data['reconstructions'] = np.array(recon_arrays, dtype=np.float32)

    return data


def build_deletion_summary(metrics_df: pd.DataFrame) -> dict:
    """Build summary of deletion reasons from failed_* columns."""
    summary = {}

    if metrics_df is None:
        return summary

    failed_cols = [c for c in metrics_df.columns if c.startswith('failed_')]
    for col in failed_cols:
        reason = col.replace('failed_', '')
        count = int(metrics_df[col].sum())
        if count > 0:
            summary[reason] = count

    return summary


def build_metadata(est, fps: float, session_name: str,
                   feedback_applied: bool = False,
                   feedback_summary: dict = None) -> dict:
    """Build metadata dictionary."""
    metadata = {
        'session_name': session_name,
        'fps': fps,
        'export_timestamp': datetime.now().isoformat(),
        'feedback_applied': feedback_applied,
    }

    # CaImAn params
    if hasattr(est, 'cnmf_dict') and est.cnmf_dict:
        # Convert numpy types to native Python types for JSON
        cnmf_params = {}
        for k, v in est.cnmf_dict.items():
            if isinstance(v, np.ndarray):
                cnmf_params[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                cnmf_params[k] = v.item()
            else:
                cnmf_params[k] = v
        metadata['cnmf_params'] = cnmf_params
    else:
        metadata['cnmf_params'] = {}

    # Autoinspection config
    if hasattr(est, 'autoinspection_config') and est.autoinspection_config:
        metadata['autoinspection_config'] = est.autoinspection_config
    else:
        metadata['autoinspection_config'] = {}

    # Autoinspection stats
    stats = {
        'n_total': int(est.C.shape[0]),
        'n_good': int(len(est.idx_components)),
        'n_bad': int(len(est.idx_components_bad)) if hasattr(est, 'idx_components_bad') else 0,
        'image_dims': list(est.imax.shape) if hasattr(est, 'imax') else None,
    }

    # Deletion summary from metrics_df
    if hasattr(est, 'metrics_df') and est.metrics_df is not None:
        stats['deletion_summary'] = build_deletion_summary(est.metrics_df)
        stats['ml_used'] = 'ml_keep_probability' in est.metrics_df.columns
    else:
        stats['deletion_summary'] = {}
        stats['ml_used'] = False

    # Feedback corrections
    if feedback_summary:
        stats['n_feedback_corrections'] = feedback_summary.get('n_total_corrections', 0)
        stats['feedback_details'] = feedback_summary

    metadata['autoinspection_stats'] = stats

    # Metrics DataFrame
    if hasattr(est, 'metrics_df') and est.metrics_df is not None:
        # Convert to dict, handling numpy types
        metrics_dict = {}
        for col in est.metrics_df.columns:
            values = est.metrics_df[col].tolist()
            # Convert numpy types
            converted = []
            for v in values:
                if isinstance(v, (np.integer, np.floating)):
                    converted.append(v.item())
                elif isinstance(v, np.ndarray):
                    converted.append(v.tolist())
                elif pd.isna(v):
                    converted.append(None)
                else:
                    converted.append(v)
            metrics_dict[col] = converted
        metadata['metrics_df'] = metrics_dict
    else:
        metadata['metrics_df'] = {}

    return metadata


def export(data: dict, metadata: dict, session_name: str, output_dir: Path):
    """Export data to NPZ and metadata to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save data as NPZ
    npz_path = output_dir / f'{session_name}_data.npz'
    np.savez_compressed(npz_path, **data)
    print(f"Saved data to: {npz_path}")

    # Save metadata as JSON
    json_path = output_dir / f'{session_name}_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {json_path}")

    return npz_path, json_path


def main():
    parser = argparse.ArgumentParser(
        description='Export processed estimates to NPZ and JSON files'
    )
    parser.add_argument('estimates_path', type=Path,
                        help='Path to processed estimates pickle file')
    parser.add_argument('-o', '--output-dir', type=Path, default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--incorporate-feedback', action='store_true',
                        help='Apply corrections from feedback CSV file')

    args = parser.parse_args()

    if not args.estimates_path.exists():
        raise FileNotFoundError(f"Estimates file not found: {args.estimates_path}")

    # Load estimates
    print(f"Loading estimates from: {args.estimates_path}")
    est = load_estimates(args.estimates_path)

    # Extract session name
    session_name = extract_session_name(est)
    print(f"Session: {session_name}")

    # Get FPS
    fps = get_fps(session_name)
    print(f"FPS: {fps}")

    # Get component indices
    component_indices = est.idx_components.copy()
    feedback_applied = False
    feedback_summary = None

    # Apply feedback if requested
    if args.incorporate_feedback:
        feedback_path = find_feedback_file(args.estimates_path, session_name)
        if feedback_path:
            print(f"Loading feedback from: {feedback_path}")
            feedback_df = load_feedback(feedback_path)
            component_indices, feedback_summary = apply_feedback(component_indices, feedback_df)
            feedback_applied = True
            print(f"Applied feedback: {feedback_summary['n_fp_removed']} FP removed, "
                  f"{feedback_summary['n_fn_added']} FN added")
        else:
            print("Warning: --incorporate-feedback specified but no feedback file found")

    print(f"Extracting data for {len(component_indices)} components...")

    # Extract data
    data = extract_data(est, component_indices)

    # Build metadata
    metadata = build_metadata(est, fps, session_name, feedback_applied, feedback_summary)

    # Export
    output_dir = args.output_dir or args.estimates_path.parent
    npz_path, json_path = export(data, metadata, session_name, output_dir)

    # Summary
    print(f"\nExport complete:")
    print(f"  Components: {len(component_indices)}")
    print(f"  C shape: {data['C'].shape}")
    if 'asp' in data:
        print(f"  ASP shape: {data['asp'].shape}")
    if 'reconstructions' in data:
        print(f"  Reconstructions shape: {data['reconstructions'].shape}")


if __name__ == '__main__':
    main()
