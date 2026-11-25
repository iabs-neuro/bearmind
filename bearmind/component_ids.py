"""Match unmodified component footprints across estimates files."""
import hashlib
from collections import Counter

import numpy as np
from scipy import sparse


def selected(est):
    ids = getattr(est, "idx_components", None)
    return set(range(est.A.shape[1])) if ids is None else {int(i) for i in ids}


def signatures(est):
    matrix = sparse.csc_matrix(est.A, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    result = []
    for start, end in zip(matrix.indptr, matrix.indptr[1:]):
        digest = hashlib.sha256(matrix.indices[start:end].astype("<i8").tobytes())
        digest.update(matrix.data[start:end].astype("<f8").tobytes())
        result.append(digest.digest())
    return result


def match(before, after):
    """Return after-index to before-index for exact, unique footprint matches."""
    if before.A.shape[0] != after.A.shape[0]:
        return {}
    if before.A.shape == after.A.shape and before.C.shape == after.C.shape:
        if sparse.issparse(before.A) and sparse.issparse(after.A):
            same_footprints = (before.A != after.A).nnz == 0
        else:
            same_footprints = np.array_equal(before.A, after.A)
        if same_footprints and np.array_equal(before.C, after.C):
            return {i: i for i in range(before.A.shape[1])}
    original, current = signatures(before), signatures(after)
    counts = Counter(original)
    unique = {key: i for i, key in enumerate(original) if counts[key] == 1}
    counts = Counter(current)
    return {i: unique[key] for i, key in enumerate(current) if key in unique and counts[key] == 1}
