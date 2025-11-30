"""
Edge Artifact Detection Module

Detects edge artifacts in calcium imaging data using two complementary methods:
1. Corner-based: Valley detection on sum of border distances (detects corner clusters)
2. Ellipse-based: Normalized radial distance > 0.9 (detects outer-edge neurons)

Combined detection uses union of both methods for robust artifact identification.
"""

import numpy as np
import pandas as pd


def find_gap_threshold(
    distances,
    bins=50,
    fov_width=None,
    fov_height=None,
    min_gap_width=3,
    truly_empty_threshold=2,
    max_empty_pct=0.05,
    search_region_pct=0.25,
    min_cluster_neurons=2,
    small_large_threshold=50,
    small_concentration_pct=0.70,
    small_concentration_bins=3,
    small_fov_factor=0.20,
    small_fallback_px=200,
    large_concentration_pct=0.50,
    large_concentration_bins=5,
    large_fov_factor=0.35,
    large_fallback_px=350
):
    """
    Detect gap by finding empty or near-empty histogram bins.

    A true bimodal distribution will have consecutive bins with very few neurons
    separating the edge cluster from the main cluster.

    Parameters:
        distances: array of distance sums to corner borders
        bins: number of histogram bins (default: 50)
        fov_width: width of field of view (for relative threshold)
        fov_height: height of field of view (for relative threshold)

        Valley detection:
        min_gap_width: minimum consecutive empty bins for valid gap (default: 3)
        truly_empty_threshold: max neurons per bin to be "truly empty" (default: 2)
        max_empty_pct: max percent of peak to consider "empty" (default: 0.05 = 5%)
        search_region_pct: only search left X% of histogram (default: 0.25 = 25%)
        min_cluster_neurons: minimum neurons on each side of gap (default: 2)

        Cluster size threshold:
        small_large_threshold: neurons separating small/large clusters (default: 50)

        Small cluster criteria (strict, for true corner artifacts):
        small_concentration_pct: min % in first N bins (default: 0.70 = 70%)
        small_concentration_bins: number of first bins to check (default: 3)
        small_fov_factor: max threshold as fraction of FOV perimeter (default: 0.20 = 20%)
        small_fallback_px: max threshold in pixels if no FOV (default: 200)

        Large cluster criteria (relaxed, for edge artifacts):
        large_concentration_pct: min % in first N bins (default: 0.50 = 50%)
        large_concentration_bins: number of first bins to check (default: 5)
        large_fov_factor: max threshold as fraction of FOV perimeter (default: 0.35 = 35%)
        large_fallback_px: max threshold in pixels if no FOV (default: 350)

    Returns:
        threshold distance, or None if no clear gap exists.
    """
    # Create histogram
    counts, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Define empty as bins with very few neurons
    max_count = counts.max()
    empty_threshold = max(truly_empty_threshold, max_count * max_empty_pct)

    # Find consecutive empty bins
    is_empty = counts <= empty_threshold

    # Look for gaps (consecutive empty bins)
    gap_starts = []
    gap_ends = []
    in_gap = False

    for i in range(len(is_empty)):
        if is_empty[i] and not in_gap:
            gap_starts.append(i)
            in_gap = True
        elif not is_empty[i] and in_gap:
            gap_ends.append(i - 1)
            in_gap = False

    if in_gap:
        gap_ends.append(len(is_empty) - 1)

    if len(gap_starts) == 0:
        return None

    # Find the first significant gap in the LEFT portion
    search_region_idx = int(len(counts) * search_region_pct)

    for gap_start, gap_end in zip(gap_starts, gap_ends):
        if gap_start >= search_region_idx:
            break

        gap_width = gap_end - gap_start + 1

        if gap_width < min_gap_width:
            continue

        # Require at least one truly empty bin in the valley
        has_truly_empty = (counts[gap_start:gap_end+1] <= truly_empty_threshold).any()
        if not has_truly_empty:
            continue

        # Check if there are neurons on BOTH sides of the gap
        left_total = counts[:gap_start].sum()
        has_left = left_total >= min_cluster_neurons
        has_right = counts[gap_end+1:].sum() >= min_cluster_neurons

        if not (has_left and has_right):
            continue

        # Use first empty bin as threshold
        threshold_idx = gap_start
        threshold = bin_centers[threshold_idx]

        # Two-tier approach based on cluster size
        if left_total < small_large_threshold:
            # Small cluster: strict concentration requirement
            first_n_bins = counts[:min(small_concentration_bins, len(counts))].sum()
            if first_n_bins / left_total < small_concentration_pct:
                continue

            # Small cluster: strict threshold limit
            if fov_width is not None and fov_height is not None:
                max_threshold = small_fov_factor * (fov_width + fov_height)
            else:
                max_threshold = small_fallback_px

            if threshold > max_threshold:
                continue
        else:
            # Large cluster: relaxed rules
            first_n_bins = counts[:min(large_concentration_bins, len(counts))].sum()
            if first_n_bins / left_total < large_concentration_pct:
                continue

            # Large cluster: allow higher thresholds
            if fov_width is not None and fov_height is not None:
                max_threshold = large_fov_factor * (fov_width + fov_height)
            else:
                max_threshold = large_fallback_px

            if threshold > max_threshold:
                continue

        return threshold

    return None


def detect_corner_artifacts_from_positions(positions, fov_width=None, fov_height=None, **gap_params):
    """
    Detect corner/edge artifacts using valley detection on sum of border distances.

    Imported from spatial_clustering.py detect_corner_clusters_v5 algorithm.

    Parameters:
        positions: Nx2 array of neuron positions (x, y coordinates)
        fov_width: Width of field of view (optional, calculated from positions if not provided)
        fov_height: Height of field of view (optional, calculated from positions if not provided)
        **gap_params: Optional parameters for gap detection (bins, min_gap_width, etc.)

    Returns:
        labels: 1 = corner/edge artifact, 0 = main cluster
        corner_info: dict with info about each corner
    """
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    if fov_width is None:
        fov_width = x_max - x_min
    if fov_height is None:
        fov_height = y_max - y_min

    # Compute distance to each border for all neurons
    dist_left = positions[:, 0] - x_min
    dist_right = x_max - positions[:, 0]
    dist_bottom = positions[:, 1] - y_min
    dist_top = y_max - positions[:, 1]

    corner_borders = {
        'bottom_left': (dist_left, dist_bottom),
        'bottom_right': (dist_right, dist_bottom),
        'top_left': (dist_left, dist_top),
        'top_right': (dist_right, dist_top)
    }

    labels = np.zeros(len(positions), dtype=int)
    corner_info = {}

    for corner_name, (border_dist1, border_dist2) in corner_borders.items():
        # Sum of distances to the 2 borders forming this corner
        border_sum = border_dist1 + border_dist2

        # Try to find valley/gap in distribution
        threshold = find_gap_threshold(border_sum, fov_width=fov_width, fov_height=fov_height, **gap_params)

        if threshold is None:
            corner_info[corner_name] = {'cluster_found': False, 'reason': 'no_valley'}
            continue

        # Mark neurons below threshold as edge artifacts
        corner_mask = border_sum < threshold
        n_corner = corner_mask.sum()

        labels[corner_mask] = 1

        corner_info[corner_name] = {
            'cluster_found': True,
            'n_neurons': n_corner,
            'threshold': threshold,
            'max_distance': border_sum[corner_mask].max() if n_corner > 0 else 0
        }

    return labels, corner_info


def detect_corner_artifacts(metrics_df, **gap_params):
    """
    Detect corner artifacts from metrics DataFrame.

    Parameters:
        metrics_df: DataFrame with 'center' column containing neuron positions
        **gap_params: Optional parameters for gap detection

    Returns:
        metrics_df: DataFrame with 'is_corner_artifact' column added
        corner_info: Dict with information about detected corners
        labels: Array of 0/1 labels (1 = artifact)
    """
    # Parse center column (stored as array or list)
    if 'center' not in metrics_df.columns:
        raise ValueError("metrics_df must have 'center' column")

    # Extract positions
    positions = np.array([np.array(c) if not isinstance(c, np.ndarray) else c
                         for c in metrics_df['center']])

    # Detect corner artifacts
    labels, corner_info = detect_corner_artifacts_from_positions(positions, **gap_params)

    # Add column to dataframe
    metrics_df = metrics_df.copy()
    metrics_df['is_corner_artifact'] = labels

    return metrics_df, corner_info, labels


def detect_ellipse_artifacts_from_positions(positions, fov_width=None, fov_height=None, threshold=0.9):
    """
    Detect edge artifacts using normalized radial distance from center of mass.

    Neurons beyond the threshold ellipse (r > threshold) are marked as artifacts.
    r = sqrt((dx/(fov_width/2))^2 + (dy/(fov_height/2))^2)
    where r=1.0 is the ellipse touching the bounding box edges.

    Parameters:
        positions: Nx2 array of neuron positions (x, y coordinates)
        fov_width: Width of field of view (optional, calculated from positions if not provided)
        fov_height: Height of field of view (optional, calculated from positions if not provided)
        threshold: Normalized radial distance threshold (default: 0.9)

    Returns:
        labels: 1 = edge artifact, 0 = main cluster
        ellipse_info: dict with detection info
    """
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()

    if fov_width is None:
        fov_width = x_max - x_min
    if fov_height is None:
        fov_height = y_max - y_min

    # Compute center of mass
    center_x, center_y = np.mean(positions[:, 0]), np.mean(positions[:, 1])

    # Compute normalized radial distance
    dx = positions[:, 0] - center_x
    dy = positions[:, 1] - center_y
    dx_norm = dx / (fov_width / 2) if fov_width > 0 else dx
    dy_norm = dy / (fov_height / 2) if fov_height > 0 else dy
    radial_dist = np.sqrt(dx_norm**2 + dy_norm**2)

    # Mark neurons beyond threshold as artifacts
    labels = (radial_dist > threshold).astype(int)
    n_artifacts = labels.sum()

    ellipse_info = {
        'n_artifacts': n_artifacts,
        'threshold': threshold,
        'center': (center_x, center_y),
        'fov_width': fov_width,
        'fov_height': fov_height,
        'max_radial': radial_dist.max()
    }

    return labels, ellipse_info


def detect_edge_artifacts_from_positions(positions, fov_width=None, fov_height=None,
                                         ellipse_threshold=0.9, **gap_params):
    """
    Detect edge artifacts using BOTH corner-based and ellipse-based methods.

    Returns the union of both detection methods for robust artifact identification.

    Parameters:
        positions: Nx2 array of neuron positions (x, y coordinates)
        fov_width: Width of field of view (optional)
        fov_height: Height of field of view (optional)
        ellipse_threshold: Threshold for ellipse detection (default: 0.9)
        **gap_params: Parameters for corner-based gap detection

    Returns:
        labels: 1 = edge artifact, 0 = main cluster (union of both methods)
        info: dict with detection info including breakdown by method
    """
    # Run corner-based detection
    corner_labels, corner_info = detect_corner_artifacts_from_positions(
        positions, fov_width=fov_width, fov_height=fov_height, **gap_params
    )

    # Run ellipse-based detection
    ellipse_labels, ellipse_info = detect_ellipse_artifacts_from_positions(
        positions, fov_width=fov_width, fov_height=fov_height, threshold=ellipse_threshold
    )

    # Union of both methods
    combined_labels = np.maximum(corner_labels, ellipse_labels)

    # Count categories
    corner_only = ((corner_labels == 1) & (ellipse_labels == 0)).sum()
    ellipse_only = ((corner_labels == 0) & (ellipse_labels == 1)).sum()
    both = ((corner_labels == 1) & (ellipse_labels == 1)).sum()

    info = {
        'n_artifacts': combined_labels.sum(),
        'n_corner_only': int(corner_only),
        'n_ellipse_only': int(ellipse_only),
        'n_both': int(both),
        'corner_info': corner_info,
        'ellipse_info': ellipse_info
    }

    return combined_labels, info


def detect_edge_artifacts(metrics_df, ellipse_threshold=0.9, **gap_params):
    """
    Detect edge artifacts from metrics DataFrame using combined corner+ellipse detection.

    Parameters:
        metrics_df: DataFrame with 'center' column containing neuron positions
        ellipse_threshold: Threshold for ellipse detection (default: 0.9)
        **gap_params: Parameters for corner-based gap detection

    Returns:
        metrics_df: DataFrame with 'is_corner_artifact' column added
        info: Dict with information about detected artifacts
        labels: Array of 0/1 labels (1 = artifact)
    """
    if 'center' not in metrics_df.columns:
        raise ValueError("metrics_df must have 'center' column")

    # Extract positions
    positions = np.array([np.array(c) if not isinstance(c, np.ndarray) else c
                         for c in metrics_df['center']])

    # Detect edge artifacts using combined method
    labels, info = detect_edge_artifacts_from_positions(
        positions, ellipse_threshold=ellipse_threshold, **gap_params
    )

    # Add column to dataframe (keep column name for backward compatibility)
    metrics_df = metrics_df.copy()
    metrics_df['is_corner_artifact'] = labels

    return metrics_df, info, labels
