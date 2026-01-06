"""
Session naming conventions - Single source of truth.

Provides patterns and functions for parsing session identifiers:
- Session IDs (e.g., NOF_H01_1D, LNOF_J53_3D_1T)
- Experiment codes (e.g., NOF, LNOF, 3DM)
- Timestamps (e.g., 22-12-2025 14-20-16)
"""
import re

# =============================================================================
# SESSION NAMING PATTERNS - Single source of truth
# =============================================================================
# Session name structure: {EXP}_{MOUSE}_{DAY}[_{TRIAL}]
#   EXP:   Experiment code - alphanumeric, any length (NOF, LNOF, 3DM, FOF, RFC, etc.)
#   MOUSE: Mouse ID - letter + digits (H01, J53, F05, D17)
#   DAY:   Day identifier - digit + letter (1D, 2D, 3D, 4D)
#   TRIAL: Optional trial - digit + letter (1T, 2T)
#
# Examples: NOF_H01_1D, LNOF_J53_3D, 3DM_D17_1D_1T, FOF_F05_2D_2T

# Full session with optional trial suffix
SESSION_PATTERN = r'([A-Z0-9]+_[A-Z]\d+_\d[A-Z](?:_\d[A-Z])?)'

# Base session without trial (EXP_MOUSE_DAY only)
SESSION_PATTERN_BASE = r'([A-Z0-9]+_[A-Z]\d+_\d[A-Z])'

# Timestamp format used in filenames
TIMESTAMP_PATTERN = r'(\d{2}-\d{2}-\d{4} \d{2}-\d{2}-\d{2})'


def extract_session_id(text: str) -> str:
    """
    Extract full session identifier from text/path.

    Args:
        text: File path or string containing session name

    Returns:
        Session ID (e.g., 'LNOF_J01_1D' or 'LNOF_J01_1D_1T'), empty string if not found
    """
    if not text:
        return ""
    match = re.search(SESSION_PATTERN, str(text))
    return match.group(1) if match else ""


def extract_base_session(text: str) -> str:
    """
    Extract base session (without trial suffix) from text/path.

    Args:
        text: File path or string containing session name

    Returns:
        Base session ID (e.g., 'LNOF_J01_1D'), empty string if not found
    """
    if not text:
        return ""
    match = re.search(SESSION_PATTERN_BASE, str(text))
    return match.group(1) if match else ""


def extract_experiment_id(session_name: str) -> str:
    """
    Extract experiment code from session name.

    Args:
        session_name: e.g., 'NOF_H03_2D', 'LNOF_J12_1D', '3DM_D17_1D'

    Returns:
        Experiment code: 'NOF', 'LNOF', '3DM', etc.
    """
    if not session_name:
        return ""
    parts = session_name.split('_')
    return parts[0] if parts else ""


def extract_timestamp(text: str) -> str:
    """
    Extract timestamp from filename/path.

    Args:
        text: String potentially containing timestamp

    Returns:
        Timestamp string (e.g., '22-12-2025 14-20-16'), empty string if not found
    """
    if not text:
        return ""
    match = re.search(TIMESTAMP_PATTERN, str(text))
    return match.group(1) if match else ""
