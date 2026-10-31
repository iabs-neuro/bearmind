"""auto_inspector footprint features under NumPy 2."""
import numpy as np
import scipy.sparse

import auto_inspector


def test_eccentricity_is_real():
    """NumPy 2 np.linalg.eigvals returns complex128 even for real eigenvalues; the complex column crashed the EBM
    (Unsupported pandas dtype: complex128) on the first real sessions. Values must stay what NumPy 1 gave."""
    dims = (40, 50)
    yy, xx = np.mgrid[:dims[0], :dims[1]]
    blob = np.exp(-((yy - 20) ** 2 / (2 * 1.0 ** 2) + (xx - 25) ** 2 / (2 * 2.0 ** 2)))  # sigma_x 2, sigma_y 1
    round_ = np.exp(-((yy - 20) ** 2 + (xx - 25) ** 2) / (2 * 1.5 ** 2))
    A = scipy.sparse.csc_matrix(np.stack([blob.ravel(order="F"), round_.ravel(order="F")], axis=1))
    ecc = auto_inspector.compute_eccentricity(A, dims)
    assert ecc.dtype == np.float64
    assert np.allclose(ecc, [2.0, 1.0], atol=1e-3)
