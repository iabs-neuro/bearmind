"""
Centralized data loading utilities for ML training and evaluation.

CRITICAL: Corner artifacts are ALWAYS excluded from GT matching.
This is the single source of truth for data loading - all ML scripts should use these functions.
"""
import numpy as np
import pandas as pd
from pathlib import Path


# Columns that are NOT features (IDs, labels, spatial info, group assignments)
# Everything else in the metrics DataFrame is automatically a feature
NON_FEATURE_COLS = {
    'component_idx',      # Neuron ID
    'center',             # Spatial position (not numeric)
    'is_corner_artifact', # Label (artifact flag)
    'corr_groups',        # Merge group ID (not quality metric)
    'delete',             # Decision label
    'merge',              # Decision label
}


def get_feature_cols(df):
    """
    Dynamically extract feature columns from a metrics DataFrame.

    Returns all numeric columns except those in NON_FEATURE_COLS.
    This allows new features to be automatically included without
    updating a hardcoded list.

    Parameters
    ----------
    df : pd.DataFrame
        Metrics DataFrame from estimates_to_metrics()

    Returns
    -------
    list of str
        Feature column names in consistent order
    """
    feature_cols = [
        col for col in df.columns
        if col not in NON_FEATURE_COLS
        and df[col].dtype in ['float64', 'float32', 'int64', 'int32', 'float', 'int']
    ]
    return sorted(feature_cols)  # Sorted for consistency


# Static list for backward compatibility and when DataFrame not available
# This should match what get_feature_cols() returns for a full metrics DataFrame
FEATURE_COLS = [
    'area', 'aspect_ratio', 'baseline', 'caiman_r_score', 'caiman_snr',
    'circularity', 'convexity', 'eccentricity', 'edge_distance', 'ellipse_r',
    'event_r2_score', 'events_fraction', 'events_per_min', 'footprint_compactness',
    'local_density', 'max_edge', 'nmae', 'nn_distance_center', 'noise_level',
    'nrmse', 'peak_amplitude_cv', 'r2_score', 'snr_recon', 't_off', 't_rise',
    'tau_decay', 'trace_kurtosis', 'trace_skewness', 'wavelet_snr'
]


def parse_center(x):
    """Parse center from string format '[y x]' to numpy array."""
    if isinstance(x, str):
        return np.fromstring(x.strip('[]'), sep=' ')
    return x


def load_session_data(session_dir, exclude_corner_artifacts=True):
    """
    Load metrics for a session from capcan_artifacts directory.

    Parameters
    ----------
    session_dir : str or Path
        Path to session's capcan_artifacts directory
    exclude_corner_artifacts : bool, default True
        If True, corner artifacts are excluded from the returned data.
        This should ALWAYS be True for GT matching and model training/evaluation.

    Returns
    -------
    df_raw : pd.DataFrame or None
        Raw neuron metrics (corner artifacts excluded if exclude_corner_artifacts=True)
    df_gt : pd.DataFrame or None
        Ground truth neuron metrics
    """
    session_dir = Path(session_dir)

    try:
        raw_metrics = session_dir / "metrics_init.csv"
        gt_metrics = session_dir / "metrics_gt.csv"

        if not raw_metrics.exists() or not gt_metrics.exists():
            return None, None

        df_raw = pd.read_csv(raw_metrics)
        df_gt = pd.read_csv(gt_metrics)

        # Parse centers
        if df_raw['center'].dtype == 'object':
            df_raw['center'] = df_raw['center'].apply(parse_center)
        if df_gt['center'].dtype == 'object':
            df_gt['center'] = df_gt['center'].apply(parse_center)

        # CRITICAL: Exclude corner artifacts from raw data
        # Corner artifacts should NEVER be considered for GT matching
        if exclude_corner_artifacts and 'is_corner_artifact' in df_raw.columns:
            df_raw = df_raw[df_raw['is_corner_artifact'] == 0].copy()

        return df_raw, df_gt
    except Exception as e:
        return None, None


def construct_gt_labels(df_raw, df_gt, max_distance=3):
    """
    Construct ground truth labels by matching raw neurons to GT neurons.

    A raw neuron is labeled as KEEP (1) if its center is within max_distance
    pixels of any GT neuron center. Otherwise it's labeled as DELETE (0).

    IMPORTANT: This function assumes corner artifacts have already been filtered
    from df_raw. Use load_session_data() with exclude_corner_artifacts=True.

    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw neuron metrics with parsed 'center' column
    df_gt : pd.DataFrame
        Ground truth neuron metrics with parsed 'center' column
    max_distance : float, default 3
        Maximum distance (pixels) for matching

    Returns
    -------
    labels : np.ndarray
        Binary labels (1=KEEP, 0=DELETE)
    """
    raw_centers = np.array(df_raw['center'].tolist())
    gt_centers = np.array(df_gt['center'].tolist())

    labels = np.zeros(len(df_raw), dtype=int)
    for i, raw_center in enumerate(raw_centers):
        distances = np.linalg.norm(gt_centers - raw_center, axis=1)
        if distances.min() <= max_distance:
            labels[i] = 1

    return labels


def create_dataset(session_dirs, feature_cols=None, max_distance=3):
    """
    Create dataset from capcan_artifacts directories.

    Corner artifacts are automatically excluded.

    Parameters
    ----------
    session_dirs : list of str or Path
        List of session directories
    feature_cols : list of str, optional
        Feature columns to include. Defaults to FEATURE_COLS.
    max_distance : float, default 3
        Maximum distance for GT matching

    Returns
    -------
    features_df : pd.DataFrame
        Feature matrix
    labels : np.ndarray
        Binary labels
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    all_features = []
    all_labels = []

    for session_dir in session_dirs:
        df_raw, df_gt = load_session_data(session_dir, exclude_corner_artifacts=True)
        if df_raw is None or len(df_raw) == 0:
            continue

        labels = construct_gt_labels(df_raw, df_gt, max_distance)

        # Extract features (only columns that exist)
        available_cols = [c for c in feature_cols if c in df_raw.columns]
        features = df_raw[available_cols].copy()
        features = features.replace([np.inf, -np.inf], np.nan)

        # Add missing columns as NaN
        for col in feature_cols:
            if col not in features.columns:
                features[col] = np.nan
        features = features[feature_cols]

        all_features.append(features)
        all_labels.append(labels)

    if not all_features:
        return pd.DataFrame(columns=feature_cols), np.array([])

    features_df = pd.concat(all_features, ignore_index=True)
    labels = np.concatenate(all_labels)

    return features_df, labels


def load_all_sessions(artifacts_dir, experiments=None):
    """
    Get list of session directories from artifacts directory.

    Parameters
    ----------
    artifacts_dir : str or Path
        Base directory containing capcan_artifacts_* folders
    experiments : list of str, optional
        Filter to specific experiment prefixes (e.g., ['NOF', 'RFC'])

    Returns
    -------
    session_dirs : list of Path
        List of session directories
    """
    artifacts_dir = Path(artifacts_dir)
    session_dirs = sorted(artifacts_dir.glob('capcan_artifacts_*'))

    if experiments:
        filtered = []
        for d in session_dirs:
            session_name = d.name.replace('capcan_artifacts_', '')
            if any(session_name.startswith(exp) for exp in experiments):
                filtered.append(d)
        session_dirs = filtered

    return session_dirs
