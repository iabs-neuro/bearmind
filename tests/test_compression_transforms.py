"""
Comprehensive tests for component index transformations during compression.

Tests verify that all component-indexed data structures (matrices, dicts, DataFrames)
are properly transformed when bad components are removed during compression.
"""
import pytest
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from estimates_compression import compress_estimates_ultra_lightweight


class MockEstimates:
    """Minimal estimates object for testing."""
    pass


def create_test_estimates():
    """Create a minimal estimates object for testing.

    Creates 10 components total, marks indices 0, 3, 7 as bad.
    Good components at old indices: [1, 2, 4, 5, 6, 8, 9] (7 good)
    After compression, these should be remapped to: [0, 1, 2, 3, 4, 5, 6]
    """
    est = MockEstimates()

    # 10 components total: 0-9
    # Bad components: 0, 3, 7
    # Good components: 1, 2, 4, 5, 6, 8, 9
    est.idx_components = np.array([1, 2, 4, 5, 6, 8, 9])  # 7 good (OLD indices)
    est.idx_components_bad = np.array([0, 3, 7])  # 3 bad

    # Create data matrices with 10 rows (one per component)
    est.C = np.random.rand(10, 1000).astype(np.float64)
    est.S = np.random.rand(10, 1000).astype(np.float64)
    est.YrA = np.random.rand(10, 1000).astype(np.float64)

    # Create metrics_df with OLD indices [0, 1, 2, ..., 9]
    est.metrics_df = pd.DataFrame({
        'component_idx': list(range(10)),
        'snr': np.random.rand(10),
        'r_value': np.random.rand(10),
        'ml_keep_probability': np.random.rand(10)
    })

    # Create reconstructions dict with OLD indices (only for good components)
    est.reconstructions = {i: np.random.rand(1000) for i in est.idx_components}

    # Create asp_cache dict with OLD indices (only for good components)
    est.asp_cache = {i: np.random.rand(1000) for i in est.idx_components}

    return est


def test_metrics_df_transformation():
    """Test that metrics_df component_idx is transformed correctly."""
    est = create_test_estimates()

    # Before compression
    assert len(est.metrics_df) == 10, "Should start with 10 rows"
    assert set(est.metrics_df['component_idx']) == set(range(10))

    # Compress
    est_compressed, savings, total_saved = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    # After compression, should have 7 components
    assert len(est_compressed.idx_components) == 7, "Should have 7 good components"

    # metrics_df should also have 7 rows
    assert len(est_compressed.metrics_df) == 7, "metrics_df should have 7 rows"

    # component_idx should be [0, 1, 2, 3, 4, 5, 6]
    expected_indices = set(range(7))
    actual_indices = set(est_compressed.metrics_df['component_idx'].values)
    assert actual_indices == expected_indices, f"Expected {expected_indices}, got {actual_indices}"

    # component_idx should match idx_components
    assert set(est_compressed.idx_components) == actual_indices


def test_all_structures_consistent():
    """Test that all component-indexed structures are consistent."""
    est = create_test_estimates()

    est_compressed, _, _ = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    n_good = len(est_compressed.idx_components)

    # Check idx_components is contiguous [0, 1, 2, ..., n_good-1]
    assert list(est_compressed.idx_components) == list(range(n_good))

    # Check matrix dimensions
    assert est_compressed.C.shape[0] == n_good
    assert est_compressed.S.shape[0] == n_good
    assert est_compressed.YrA.shape[0] == n_good

    # Check dict keys
    assert set(est_compressed.reconstructions.keys()) == set(range(n_good))
    assert set(est_compressed.asp_cache.keys()) == set(range(n_good))

    # Check metrics_df
    assert len(est_compressed.metrics_df) == n_good
    assert set(est_compressed.metrics_df['component_idx']) == set(range(n_good))


def test_compression_validation():
    """Test that validation catches inconsistencies."""
    est = create_test_estimates()

    # Manually break metrics_df to trigger validation error
    # Make it have fewer rows than expected
    est.metrics_df = est.metrics_df.iloc[:5].copy()

    with pytest.raises(ValueError, match="COMPRESSION VALIDATION FAILED"):
        compress_estimates_ultra_lightweight(est, remove_bad_components=True)


def test_no_bad_components():
    """Test compression when no bad components are removed."""
    est = MockEstimates()

    # All components are good
    est.idx_components = np.array([0, 1, 2, 3, 4])
    est.idx_components_bad = np.array([])

    est.C = np.random.rand(5, 1000).astype(np.float64)
    est.S = np.random.rand(5, 1000).astype(np.float64)
    est.YrA = np.random.rand(5, 1000).astype(np.float64)

    est.metrics_df = pd.DataFrame({
        'component_idx': list(range(5)),
        'snr': np.random.rand(5)
    })

    est.reconstructions = {i: np.random.rand(1000) for i in range(5)}
    est.asp_cache = {i: np.random.rand(1000) for i in range(5)}

    # Compress
    est_compressed, _, _ = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    # Everything should remain the same
    assert len(est_compressed.idx_components) == 5
    assert len(est_compressed.metrics_df) == 5
    assert set(est_compressed.metrics_df['component_idx']) == set(range(5))


def test_scattered_bad_components():
    """Test with bad components scattered throughout the range."""
    est = MockEstimates()

    # 20 components, every 4th is bad: 0, 4, 8, 12, 16
    all_indices = set(range(20))
    bad_indices = {0, 4, 8, 12, 16}
    good_indices = sorted(all_indices - bad_indices)

    est.idx_components = np.array(good_indices)
    est.idx_components_bad = np.array(sorted(bad_indices))

    est.C = np.random.rand(20, 1000).astype(np.float64)
    est.S = np.random.rand(20, 1000).astype(np.float64)
    est.YrA = np.random.rand(20, 1000).astype(np.float64)

    est.metrics_df = pd.DataFrame({
        'component_idx': list(range(20)),
        'snr': np.random.rand(20)
    })

    est.reconstructions = {i: np.random.rand(1000) for i in good_indices}
    est.asp_cache = {i: np.random.rand(1000) for i in good_indices}

    # Compress
    est_compressed, _, _ = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    # Should have 15 good components
    n_good = len(good_indices)
    assert len(est_compressed.idx_components) == n_good
    assert len(est_compressed.metrics_df) == n_good

    # Indices should be remapped to [0, 1, 2, ..., 14]
    assert set(est_compressed.idx_components) == set(range(n_good))
    assert set(est_compressed.metrics_df['component_idx']) == set(range(n_good))


def test_consecutive_bad_at_start():
    """Test with consecutive bad components at the start."""
    est = MockEstimates()

    # First 5 components are bad
    est.idx_components = np.array([5, 6, 7, 8, 9])  # Good components
    est.idx_components_bad = np.array([0, 1, 2, 3, 4])  # Bad components

    est.C = np.random.rand(10, 1000).astype(np.float64)
    est.S = np.random.rand(10, 1000).astype(np.float64)
    est.YrA = np.random.rand(10, 1000).astype(np.float64)

    est.metrics_df = pd.DataFrame({
        'component_idx': list(range(10)),
        'snr': np.random.rand(10)
    })

    est.reconstructions = {i: np.random.rand(1000) for i in range(5, 10)}
    est.asp_cache = {i: np.random.rand(1000) for i in range(5, 10)}

    # Compress
    est_compressed, _, _ = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    # Should have 5 good components remapped to [0, 1, 2, 3, 4]
    assert len(est_compressed.idx_components) == 5
    assert set(est_compressed.idx_components) == set(range(5))
    assert set(est_compressed.metrics_df['component_idx']) == set(range(5))


def test_consecutive_bad_at_end():
    """Test with consecutive bad components at the end."""
    est = MockEstimates()

    # Last 5 components are bad
    est.idx_components = np.array([0, 1, 2, 3, 4])  # Good components
    est.idx_components_bad = np.array([5, 6, 7, 8, 9])  # Bad components

    est.C = np.random.rand(10, 1000).astype(np.float64)
    est.S = np.random.rand(10, 1000).astype(np.float64)
    est.YrA = np.random.rand(10, 1000).astype(np.float64)

    est.metrics_df = pd.DataFrame({
        'component_idx': list(range(10)),
        'snr': np.random.rand(10)
    })

    est.reconstructions = {i: np.random.rand(1000) for i in range(5)}
    est.asp_cache = {i: np.random.rand(1000) for i in range(5)}

    # Compress
    est_compressed, _, _ = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    # Should have 5 good components (already [0, 1, 2, 3, 4])
    assert len(est_compressed.idx_components) == 5
    assert set(est_compressed.idx_components) == set(range(5))
    assert set(est_compressed.metrics_df['component_idx']) == set(range(5))


def test_empty_metrics_df():
    """Test with no metrics_df attached."""
    est = MockEstimates()

    est.idx_components = np.array([1, 2, 4, 5])
    est.idx_components_bad = np.array([0, 3])

    est.C = np.random.rand(6, 1000).astype(np.float64)
    est.S = np.random.rand(6, 1000).astype(np.float64)
    est.YrA = np.random.rand(6, 1000).astype(np.float64)

    # No metrics_df
    est.metrics_df = None

    est.reconstructions = {i: np.random.rand(1000) for i in [1, 2, 4, 5]}
    est.asp_cache = {i: np.random.rand(1000) for i in [1, 2, 4, 5]}

    # Should compress without error
    est_compressed, _, _ = compress_estimates_ultra_lightweight(
        est, remove_bad_components=True
    )

    assert len(est_compressed.idx_components) == 4
    assert est_compressed.metrics_df is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
