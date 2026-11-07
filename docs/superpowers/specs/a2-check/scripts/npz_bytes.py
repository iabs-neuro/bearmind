"""Byte equality (dtype, shape, raw bytes) of every array in two npz files. usage: npz_bytes.py A.npz B.npz"""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
bad = [k for k in a.files if k not in b.files or a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
       or a[k].tobytes() != b[k].tobytes()]
print(f"{len(a.files)} arrays, keys equal {sorted(a.files) == sorted(b.files)}, byte-different: {bad}")
print("[OK] byte-identical" if not bad and sorted(a.files) == sorted(b.files) else "[FAILED]")
