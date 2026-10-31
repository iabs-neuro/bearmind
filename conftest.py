import os

# Same workaround as bearmind/__init__.py, at the repo root so that the legacy
# root-level tests (which import numpy before bearmind) are covered too.
# Must run before anything imports numpy.
os.environ.setdefault("OPENBLAS_CORETYPE", "Sandybridge")
