"""Install check of the bearmind environment: every package the program needs imports.

install2_pip.bat runs it after `pip install -r requirements-pip.txt`. Exit code 0: all packages import; 1: some
do not. tests/test_environment.py checks that this list covers every package the program imports.
"""
import os
import sys

# Before NumPy loads, as bearmind/__init__.py does: install2_pip.bat runs the env's python without activation, so
# the conda env variable (the Zen 4 workaround) is not set here.
os.environ.setdefault("OPENBLAS_CORETYPE", "Sandybridge")

# (import name, what it is for); the packages of environment.yml and requirements-pip.txt
REQUIRED_PACKAGES = [
    # CNMF-E, motion correction and the autoinspection
    ("caiman", "CaImAn: motion correction, CNMF-E"),
    ("driada", "driada: autoinspection features"),
    ("ssqueezepy", "ssqueezepy: wavelet transform of the autoinspection"),
    ("interpret", "InterpretML: production EBM model"),
    ("sortedcontainers", "sortedcontainers"),
    ("joblib", "joblib: loads the EBM model"),
    ("sklearn", "scikit-learn"),
    ("torch", "PyTorch (OpenBLAS build)"),
    # Numbers and tables
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("pandas", "pandas"),
    ("threadpoolctl", "threadpoolctl: BLAS thread limits"),
    ("psutil", "psutil: memory and processes"),
    ("h5py", "h5py: HDF5 files"),
    ("pywt", "PyWavelets"),
    # Video and images
    ("av", "PyAV: FFV1 and mp4 video"),
    ("cv2", "OpenCV"),
    ("tifffile", "tifffile: TIFF files"),
    ("imageio_ffmpeg", "imageio-ffmpeg: ffmpeg binary of the QC videos"),
    ("moviepy", "moviepy: old pipeline (equivalence test)"),
    # Window and manual inspection
    ("PySide6", "PySide6: BEARMiND window"),
    ("bokeh", "bokeh: manual-inspection page"),
    ("tornado", "tornado: manual-inspection server"),
    ("IPython", "IPython"),
    ("ipywidgets", "ipywidgets"),
    ("matplotlib", "matplotlib: plots"),
    # Utilities
    ("yaml", "PyYAML: configs"),
    ("natsort", "natsort: natural sorting"),
    ("tqdm", "tqdm: progress bars"),
    ("pytz", "pytz: time zones"),
    ("pytest", "pytest: tests"),
]


def check_imports():
    """(lines to print, import names that failed)."""
    lines, failed = [], []
    for name, what in REQUIRED_PACKAGES:
        try:
            module = __import__(name)
        except Exception as e:  # a broken DLL raises OSError or ImportError; any failure means "not usable"
            lines.append(f"[FAILED] {name}: {what}: {type(e).__name__}: {str(e)[:80]}")
            failed.append(name)
        else:
            lines.append(f"[OK] {name} {getattr(module, '__version__', '')}: {what}")
    return lines, failed


def main() -> int:
    print("BEARMiND environment check")
    print("=" * 50)
    lines, failed = check_imports()
    for line in lines:
        print("  " + line.encode("ascii", "replace").decode())
    print("=" * 50)
    print(f"Results: {len(lines) - len(failed)} OK, {len(failed)} FAILED")
    if failed:
        print("[FAILED] Some packages do not import. Recreate the environment (docs/students/README.md, section 2).")
        return 1
    print("[OK] Environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
