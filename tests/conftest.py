import os

# Same workaround as bearmind/__init__.py; set before anything imports numpy.
os.environ.setdefault("OPENBLAS_CORETYPE", "Sandybridge")
