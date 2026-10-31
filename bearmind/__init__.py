"""bearmind: series-level calcium-imaging pipeline."""
import os

# OpenBLAS 0.3.34 does not recognise Zen 4 ("Core not found") and its AVX2 dgemm
# kernels segfault on this machine. Must be set before numpy loads; child
# processes inherit it. A value the user already set is kept. No effect if
# numpy was imported before bearmind in the same process.
os.environ.setdefault("OPENBLAS_CORETYPE", "Sandybridge")
