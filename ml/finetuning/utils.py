"""
Shared utilities for finetuning system.

Provides common functions for:
- Session name extraction
- Feature column management
- Timestamp parsing
- File path handling
"""
import re
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# Session name extraction pattern
SESSION_PATTERN = r'[A-Z]{3,4}_[A-Z]\d{2}_(\dD|\dT)_(_?\dT_)?'


def extract_session_name(text: str) -> str:
    """
    Extract session name from file path or text using standard pattern.

    Pattern: [A-Z]{3,4}_[A-Z]\d{2}_(\dD|\dT)_
    Examples: NOF_H03_2D, LNOF_J12_1D, 3DM_F48_1D

    Args:
        text: File path or text containing session name

    Returns:
        Extracted session name, or empty string if not found
    """
    match = re.search(SESSION_PATTERN, text)

    if match:
        end_pos = match.end()
        return text[:end_pos]
    else:
        return ""


def parse_feedback_timestamp(filename: str) -> Optional[datetime]:
    """
    Parse timestamp from feedback CSV filename.

    Expected format: *_feedback_{DD-MM-YYYY HH-MM-SS}.csv

    Args:
        filename: Feedback CSV filename

    Returns:
        datetime object, or None if parsing fails
    """
    # Extract timestamp pattern
    timestamp_pattern = r'(\d{2}-\d{2}-\d{4} \d{2}-\d{2}-\d{2})'
    match = re.search(timestamp_pattern, filename)

    if not match:
        return None

    timestamp_str = match.group(1)

    try:
        # Parse: DD-MM-YYYY HH-MM-SS
        return datetime.strptime(timestamp_str, '%d-%m-%Y %H-%M-%S')
    except ValueError:
        return None


def get_experiment_from_session(session_name: str) -> str:
    """
    Extract experiment ID from session name.

    Args:
        session_name: e.g., "NOF_H03_2D", "LNOF_J12_1D"

    Returns:
        Experiment ID: "NOF", "RFC", "FOF", "3DM", "LNOF", etc.
    """
    if not session_name:
        return ""

    # Extract up to first underscore
    parts = session_name.split('_')
    if len(parts) > 0:
        return parts[0]

    return ""


# Feature columns (from ml/data_utils.py)
# This should match the 35 features used in v8_iter5 model
FEATURE_COLS = [
    'area', 'aspect_ratio', 'baseline', 'baseline_drift', 'bimodality',
    'caiman_r_score', 'caiman_snr', 'circularity', 'convexity',
    'eccentricity', 'edge_distance', 'ellipse_r', 'event_r2_score',
    'event_snr', 'events_fraction', 'events_per_min', 'footprint_compactness',
    'half_crossing_rate', 'hurst_exponent', 'kinetics_opt', 'local_density',
    'max_edge', 'mean_time_at_peak', 'nmae', 'nn_distance_center',
    'noise_level', 'nrmse', 'peak_amplitude_cv', 'r2_score', 'snr_recon',
    't_off', 't_rise', 'tau_decay', 'trace_kurtosis', 'trace_skewness'
]


# F-beta configuration (from ml/data_utils.py)
PRECISION_RECALL_RATIO = 3.0
FBETA_BETA = np.sqrt(1 / PRECISION_RECALL_RATIO)  # ~0.577


def compute_fbeta(precision: float, recall: float, beta: Optional[float] = None) -> float:
    """
    Compute F-beta score from precision and recall.

    Args:
        precision: Precision value [0, 1]
        recall: Recall value [0, 1]
        beta: Beta value (default: FBETA_BETA = 0.577 for 3:1 precision:recall ratio)

    Returns:
        F-beta score
    """
    if beta is None:
        beta = FBETA_BETA

    if precision + recall == 0:
        return 0.0

    beta_squared = beta ** 2
    return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)


def validate_feedback_columns(df, required_cols=None):
    """
    Validate that feedback DataFrame has required columns.

    Args:
        df: Feedback DataFrame
        required_cols: List of required column names (default: essential columns)

    Returns:
        tuple: (is_valid, missing_columns)
    """
    if required_cols is None:
        required_cols = ['neuron_idx', 'feedback_type', 'ml_keep_probability']

    missing = [col for col in required_cols if col not in df.columns]

    return len(missing) == 0, missing


def standardize_feedback_columns(df):
    """
    Standardize feedback DataFrame column names (handle legacy formats).

    Handles:
    - 'idx' -> 'neuron_idx' (if neuron_idx missing)
    - 'delete_status' -> 'delete' (if delete missing)

    Args:
        df: Feedback DataFrame

    Returns:
        DataFrame with standardized column names
    """
    df = df.copy()

    # Rename idx to neuron_idx if needed
    if 'neuron_idx' not in df.columns and 'idx' in df.columns:
        df['neuron_idx'] = df['idx']

    # Rename delete_status to delete if needed
    if 'delete' not in df.columns and 'delete_status' in df.columns:
        df['delete'] = df['delete_status']

    return df


def create_y_true_labels(feedback_df):
    """
    Create binary labels from feedback_type.

    FP feedback: model said KEEP (P>threshold), user says DELETE -> y=0
    FN feedback: model said DELETE (P<threshold), user says KEEP -> y=1

    Args:
        feedback_df: DataFrame with 'feedback_type' column

    Returns:
        DataFrame with added 'y_true' column
    """
    feedback_df = feedback_df.copy()

    feedback_df['y_true'] = (feedback_df['feedback_type'] == 'FN').astype(int)

    return feedback_df
