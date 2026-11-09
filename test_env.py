"""Install check of the bearmind environment: every package the program needs imports, and python and CaImAn are
the versions environment.yml pins.

install2_pip.bat runs it after `pip install -r requirements-pip.txt`; the student guide runs it on an env named
bearmind that is already there. Exit code 0: all packages import and the versions match ("0 FAILED"); 1: not.
tests/test_environment.py checks that this list covers every package the program imports and that the versions
agree with environment.yml.
"""
import os
import platform
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

# main's older environment.yml makes an env with the same name (python 3.10, caiman 1.11.3, no PySide6): the import
# check alone can pass there, and the batch would then run with another CaImAn. A version matches when it is this
# one or a release of it ("3.12" takes 3.12.16).
REQUIRED_VERSIONS = {"python": "3.12", "caiman": "1.13.2"}


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


def installed_versions():
    """name -> version of this python and of the caiman check_imports loaded (None: caiman did not import, which
    check_imports reports)."""
    caiman = sys.modules.get("caiman")
    return {"python": platform.python_version(), "caiman": getattr(caiman, "__version__", None)}


def _matches(have, want):
    return have == want or have.startswith((want + ".", want + "+"))


def check_versions(installed=None):
    """(lines to print, names whose version is not REQUIRED_VERSIONS); installed: name -> version, by default
    installed_versions(). A version that cannot be read is left out: its import already failed. No annotations
    that need a new python: the check must run, and say so, in an env of another python."""
    installed = installed_versions() if installed is None else installed
    lines, failed = [], []
    for name, want in REQUIRED_VERSIONS.items():
        have = installed.get(name)
        if have is None:
            continue
        if _matches(str(have), want):
            lines.append(f"[OK] {name} {have}: version {want} required")
        else:
            lines.append(f"[FAILED] {name} {have}: version {want} required; this is another environment")
            failed.append(name)
    return lines, failed


def main() -> int:
    print("BEARMiND environment check")
    print("=" * 50)
    lines, failed = check_imports()
    version_lines, version_failed = check_versions()
    lines, failed = lines + version_lines, failed + version_failed
    for line in lines:
        print("  " + line.encode("ascii", "replace").decode())
    print("=" * 50)
    print(f"Results: {len(lines) - len(failed)} OK, {len(failed)} FAILED")
    if failed:
        print("[FAILED] The environment does not match: see the lines above. Recreate it (docs/students/README.md, "
              "section 2).")
        return 1
    print("[OK] Environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
