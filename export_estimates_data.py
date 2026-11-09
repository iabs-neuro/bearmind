"""
Export processed estimates data to NPZ, JSON and MAT files: the export library of the pipeline.

The export stage (bearmind/stages/export.py, python -m bearmind) calls it; it has no command line of its own.

Output:
    {session_name}_data.npz - numpy arrays (C, asp, reconstructions, component_indices, fps)
    {session_name}_metadata.json - all metadata as JSON
    {session_name}_filters.mat - spatial footprints as MATLAB array

Features:
    - ML threshold filtering: Removes neurons with ml_keep_probability < threshold (default 0.72);
      merged components carry no probability and are kept
    - Feedback incorporation: Applies FP/FN corrections from feedback CSV
    - Legacy file support: Works with files processed before autoinspection_config was added
"""

import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.ndimage import gaussian_filter

from naming import extract_session_id
from bearmind.config import load_defaults
from bearmind.timestamps import fps_from_table

# ML threshold for processed files without autoinspection_config: autoinspect.ml_threshold of bearmind/defaults.yaml,
# the value the export stage's hash assumes for such a file (runner.stage_params)
DEFAULT_ML_THRESHOLD = load_defaults()["autoinspect"]["ml_threshold"]

# Merge groups are sets of components that are all pairwise close (complete linkage,
# auto_inspector.assign_merge_groups); run_auto_inspection stores this value in
# autoinspection_config['merge_grouping'] and the autoinspect stage hash carries it. "fbd_pairs" = single
# linkage (chains), 28a1b4c..461a66f; earlier pickles carry no key (whole correlation clusters). The export
# guard treats both as old: their merge groups glued distant cells.
MERGE_GROUPING = "fbd_complete"

def get_fps_from_table(session_name: str, default_fps: float = None) -> float | None:
    """FPS of a session from fps_data.csv (bearmind.timestamps.fps_from_table); default_fps when it is not there.

    Returned as stored, not rounded: exported traces are put on a time axis with
    this rate, and rounding 19.76 to 20 stretches it by more than one percent.
    """
    fps = fps_from_table(session_name)
    return default_fps if fps is None else fps


def load_estimates(path: Path):
    """Load estimates from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def extract_session_name(est_or_path) -> str:
    """Extract session identifier (e.g., LNOF_J01_1D or HOS_D04_1D_2T) from estimates or path."""
    if hasattr(est_or_path, 'name'):
        name = est_or_path.name
    else:
        name = str(est_or_path)

    # Extract session ID (trial suffix kept) using flexible pattern from naming.py
    session_id = extract_session_id(name)
    if session_id:
        return session_id

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

    def same_session(path: Path) -> bool:
        # Prefix glob would match HOS_X_1D_2T for HOS_X_1D; require exact ID
        sid = extract_session_id(session_name)
        return not sid or extract_session_id(path.name) == sid

    # 1. Same folder as estimates (new behavior)
    direct_match = parent / f'{session_name}_feedback.csv'
    if direct_match.exists():
        return direct_match

    # Glob for variations in same folder
    for path in parent.glob(f'*{session_name}*feedback*.csv'):
        if path.exists() and same_session(path):
            return path

    # 2. Parent folder (legacy behavior)
    grandparent = parent.parent
    legacy_match = grandparent / f'{session_name}_feedback.csv'
    if legacy_match.exists():
        return legacy_match

    for path in grandparent.glob(f'*{session_name}*feedback*.csv'):
        if path.exists() and same_session(path):
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
    # Handle both list and numpy array
    idx_set = set(list(idx_components))

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


def ml_approved_components(metrics_df: pd.DataFrame, threshold: float) -> set:
    """
    Component indices that pass the export ML filter.

    Components produced by automatic merging are built only from neurons that
    were already kept, and the model is never applied to them, so their
    ml_keep_probability is NaN. A plain `>= threshold` comparison is False for
    NaN and would drop every merged neuron from the export.

    Neurons marked deleted stay in idx_components for the GUI, and some of them
    keep a high probability (the duplicate rule removes the lower-SNR neuron of a
    close pair regardless of it), so the delete mark is checked as well.
    """
    approved = metrics_df['ml_keep_probability'] >= threshold
    if 'decision' in metrics_df.columns:
        approved |= metrics_df['decision'] == 'from_merge'
    if 'delete' in metrics_df.columns:
        approved &= metrics_df['delete'] != 1
    return set(metrics_df.loc[approved, 'component_idx'].tolist())


def approved_components(est, threshold: float) -> tuple:
    """
    Exported component indices of a processed estimates object, and what was left out.

    Returns (indices, info). For a file whose autoinspection_config names the current merge grouping this is
    ml_approved_components. A file made before that grouping (no marker) holds merge products built from
    whole correlation clusters, which glued cells that are far apart, so its merge products are not exported:
    the rule is then `delete == 0 and not from_merge` for the scored rows. Run autoinspection again to
    export the merges. info: {"merge_grouping": value or "whole_cluster", "merge_products_excluded": n}.
    """
    df = est.metrics_df
    cfg = getattr(est, 'autoinspection_config', None) or {}
    grouping = cfg.get('merge_grouping')
    approved = ml_approved_components(df, threshold)
    if grouping == MERGE_GROUPING:
        return approved, {'merge_grouping': grouping, 'merge_products_excluded': 0}
    n_excluded = 0
    if 'decision' in df.columns and 'component_idx' in df.columns:
        merged = set(df.loc[df['decision'] == 'from_merge', 'component_idx'].tolist())
        # a merge row outside idx_components would not be exported anyway: do not report it as left out
        n_excluded = len(approved & merged & set(np.asarray(est.idx_components).tolist()))
        approved = approved - merged
    return approved, {'merge_grouping': grouping or 'whole_cluster', 'merge_products_excluded': n_excluded}


def warn_merge_products_left_out(session_name: str, info: dict) -> None:
    """Print the one-line warning when approved_components left merge products out."""
    if info.get('merge_products_excluded'):
        print(f"[WARN] {session_name}: processed before the merge-grouping fix; "
              f"{info['merge_products_excluded']} merge products not exported; "
              f"re-run autoinspection to export them")


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
                   feedback_summary: dict = None,
                   ml_filter_info: dict = None,
                   component_indices: np.ndarray = None) -> dict:
    """
    Build metadata dictionary.

    Args:
        est: CaImAn estimates object
        fps: Frames per second
        session_name: Session identifier
        feedback_applied: Whether feedback corrections were applied
        feedback_summary: Dict with feedback correction details
        ml_filter_info: Dict with ML filtering info (ml_filtered, threshold_used, n_before, n_after)
        component_indices: Array of component indices being exported (for filtering metrics_df)
    """
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

    # ML filtering info
    if ml_filter_info:
        stats['ml_filtered'] = ml_filter_info.get('ml_filtered', False)
        stats['ml_threshold_used'] = ml_filter_info.get('threshold_used')
        stats['n_before_ml_filter'] = ml_filter_info.get('n_before')
        stats['n_after_ml_filter'] = ml_filter_info.get('n_after')

    metadata['autoinspection_stats'] = stats

    # Metrics DataFrame - filter to only exported components
    if hasattr(est, 'metrics_df') and est.metrics_df is not None:
        df = est.metrics_df

        # Filter to only exported components if indices provided
        if component_indices is not None:
            component_set = set(component_indices.tolist() if hasattr(component_indices, 'tolist') else component_indices)
            df = df[df['component_idx'].isin(component_set)]

        # Validation: exported indices and metadata must match 1:1
        if component_indices is not None and 'component_idx' in df.columns:
            exported_set = set(int(i) for i in component_indices)
            metadata_set = set(int(i) for i in df['component_idx'].values)
            missing = exported_set - metadata_set
            extra = metadata_set - exported_set

            if missing:
                print(f"[WARNING] {len(missing)} exported neurons have no metadata (indices: {sorted(missing)[:5]}...)")
            if extra:
                print(f"[WARNING] {len(extra)} metadata rows don't match exported neurons (indices: {sorted(extra)[:5]}...)")

        # Convert to dict, handling numpy types
        metrics_dict = {}
        for col in df.columns:
            values = df[col].tolist()
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
    # The rate travels with the traces: the NPZ is often used without the JSON
    np.savez_compressed(npz_path, fps=np.float64(metadata['fps']), **data)
    print(f"Saved data to: {npz_path}")

    # Save metadata as JSON
    json_path = output_dir / f'{session_name}_metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {json_path}")

    return npz_path, json_path


def export_filters_mat(est, component_indices: np.ndarray, output_path: Path, sigma: int = 3):
    """
    Export spatial filters as MATLAB .mat file.

    Args:
        est: CaImAn estimates object with A matrix and imax
        component_indices: Array of component indices to export
        output_path: Path for output .mat file
        sigma: Gaussian smoothing sigma (default: 3, set to 0 for no smoothing)
    """
    if not hasattr(est, 'A') or est.A is None:
        print("Warning: Cannot export filters - missing A matrix")
        return None

    if not hasattr(est, 'imax') or est.imax is None:
        print("Warning: Cannot export filters - missing imax")
        return None

    ims = []
    for idx in component_indices:
        sp = est.A.T[idx]
        im = np.asarray(sp.reshape(est.imax.shape[::-1]).todense())
        if sigma:
            im = gaussian_filter(im, sigma=sigma)
        # Normalize to 0-255 uint8
        max_val = np.max(im)
        if max_val > 0:
            ims.append((im * 255 / max_val).astype(np.uint8))
        else:
            ims.append(np.zeros(est.imax.shape[::-1], dtype=np.uint8))

    savemat(output_path, {"A": np.array(ims)})
    print(f"Saved filters to: {output_path}")
    return output_path
