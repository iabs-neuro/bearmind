"""polygon.py shape features used by autoinspection."""
import numpy as np

import polygon


def _contour(points):
    return {"coordinates": np.array(points, dtype=float)}


def test_convexities_with_2d_contours():
    """2-D cross product (z component) of neighbouring edges; NumPy 2 no longer accepts 2-vectors in np.cross.

    angle = 1 where the z component is <= 0. Dart (0,0) (2,1) (0,2) (1,1): z = -1, -4, -1, +2 -> 3 of 4.
    """
    square_cw = _contour([[0, 0], [0, 1], [1, 1], [1, 0]])         # z = +1 at every corner
    square_ccw = _contour([[0, 0], [1, 0], [1, 1], [0, 1]])        # z = -1 at every corner
    dart = _contour([[0, 0], [2, 1], [np.nan, np.nan], [0, 2], [1, 1]])  # NaN row dropped
    out = polygon.get_convexities([square_cw, square_ccw, dart])
    assert np.allclose(out, [0.0, 1.0, 0.75])
