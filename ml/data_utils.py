"""
Centralized data loading utilities for ML training and evaluation.

CRITICAL: Corner artifacts are ALWAYS excluded from GT matching.
This is the single source of truth for data loading - all ML scripts should use these functions.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


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


# =============================================================================
# SCORING CONFIGURATION
# =============================================================================
# Precision/recall importance ratio for model selection
# Higher values prioritize precision (fewer false positives)
# Default 3.0 means precision is 3x more important than recall
PRECISION_RECALL_RATIO = 3.0

# Derived beta for F-beta score: beta = sqrt(1/ratio)
# beta < 1 weights precision more, beta > 1 weights recall more
FBETA_BETA = np.sqrt(1 / PRECISION_RECALL_RATIO)


def compute_fbeta(precision, recall, beta=None):
    """
    Compute F-beta score from precision and recall.

    Parameters
    ----------
    precision : float
    recall : float
    beta : float, optional
        If None, uses global FBETA_BETA

    Returns
    -------
    float : F-beta score
    """
    if beta is None:
        beta = FBETA_BETA
    if (precision + recall) == 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


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


def get_session_experiment(session_dir):
    """
    Extract experiment ID from session directory name.

    Parameters
    ----------
    session_dir : Path or str
        Session directory path (e.g., 'capcan_artifacts_NOF_M12_1D')

    Returns
    -------
    str
        Experiment ID (e.g., 'NOF')
    """
    session_dir = Path(session_dir)
    session_name = session_dir.name.replace('capcan_artifacts_', '')
    return session_name.split('_')[0]


def stratified_session_split(session_dirs, test_fraction=0.25, random_state=42):
    """
    Split sessions into train/test with stratification by experiment.

    This ensures each experiment (NOF, RFC, FOF, 3DM, etc.) is proportionally
    represented in both train and test sets, avoiding biased evaluation.

    IMPORTANT: Always use this function instead of random shuffling for
    train/test splits to ensure proper experiment balance.

    Parameters
    ----------
    session_dirs : list of Path
        List of session directories
    test_fraction : float, default 0.25
        Fraction of sessions for test set
    random_state : int, default 42
        Random seed for reproducibility

    Returns
    -------
    train_sessions : list of Path
        Training session directories
    test_sessions : list of Path
        Test session directories
    split_info : dict
        Information about the split (experiment counts, etc.)
    """
    session_dirs = list(session_dirs)  # Ensure it's a list

    # Get experiment labels for stratification
    experiments = [get_session_experiment(d) for d in session_dirs]

    # Check if stratification is possible (need at least 2 samples per class)
    from collections import Counter
    exp_counts = Counter(experiments)
    min_count = min(exp_counts.values())

    if min_count < 2:
        # Fall back to random split if stratification not possible
        import random
        random.seed(random_state)
        shuffled = session_dirs.copy()
        random.shuffle(shuffled)
        n_train = int(len(shuffled) * (1 - test_fraction))
        train_sessions = shuffled[:n_train]
        test_sessions = shuffled[n_train:]
        split_info = {
            'stratified': False,
            'reason': f'Min experiment count ({min_count}) < 2',
            'train_experiments': Counter([get_session_experiment(d) for d in train_sessions]),
            'test_experiments': Counter([get_session_experiment(d) for d in test_sessions])
        }
        return train_sessions, test_sessions, split_info

    # Stratified split
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_fraction,
        random_state=random_state
    )

    train_idx, test_idx = next(splitter.split(session_dirs, experiments))

    train_sessions = [session_dirs[i] for i in train_idx]
    test_sessions = [session_dirs[i] for i in test_idx]

    # Collect split info
    train_exp_counts = Counter([get_session_experiment(d) for d in train_sessions])
    test_exp_counts = Counter([get_session_experiment(d) for d in test_sessions])

    split_info = {
        'stratified': True,
        'total_sessions': len(session_dirs),
        'train_sessions': len(train_sessions),
        'test_sessions': len(test_sessions),
        'train_experiments': dict(train_exp_counts),
        'test_experiments': dict(test_exp_counts),
        'random_state': random_state
    }

    return train_sessions, test_sessions, split_info


def print_split_info(split_info):
    """Print formatted information about train/test split."""
    print(f"Stratified split: {split_info['stratified']}")
    if not split_info['stratified']:
        print(f"  Reason: {split_info.get('reason', 'unknown')}")

    print(f"Train sessions: {split_info.get('train_sessions', len(split_info['train_experiments']))}")
    print(f"Test sessions: {split_info.get('test_sessions', len(split_info['test_experiments']))}")

    print("Train experiments:", dict(split_info['train_experiments']))
    print("Test experiments:", dict(split_info['test_experiments']))
