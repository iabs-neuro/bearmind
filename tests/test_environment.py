import ast
import fnmatch
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
LOCK = REPO / "env" / "bearmind-lock.txt"
ENV_YML = REPO / "environment.yml"
PIP_TXT = REPO / "requirements-pip.txt"

# The program's own imports (root modules, bearmind/, tests/) by the package that provides them in the env files.
# A package that comes only as another package's dependency can change version in a fresh solve; the program's own
# imports must not (cleanup audit 2026-11-09, section 7). A new third-party import fails
# test_program_imports_are_pinned until it is pinned and named here.
IMPORT_PACKAGE = {
    "av": "av", "bokeh": "bokeh", "caiman": "caiman", "cv2": "opencv", "driada": "driada",
    "imageio_ffmpeg": "imageio-ffmpeg", "IPython": "ipython", "ipywidgets": "ipywidgets", "joblib": "joblib",
    "matplotlib": "matplotlib", "pylab": "matplotlib", "numpy": "numpy", "pandas": "pandas", "psutil": "psutil",
    "PySide6": "pyside6", "shiboken6": "pyside6", "pytest": "pytest", "pytz": "pytz", "scipy": "scipy",
    "ssqueezepy": "ssqueezepy", "threadpoolctl": "threadpoolctl", "tifffile": "tifffile", "torch": "pytorch",
    "tornado": "tornado", "tqdm": "tqdm", "yaml": "pyyaml",
}
# Imported only when present: the GPU wavelet backend (wavelet_backend.py) falls back to NumPy without it.
OPTIONAL_IMPORTS = {"cupy"}
# Unpinned on purpose: pip is the installer, not a package the program imports.
UNPINNED_OK = {"pip"}


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def _lock() -> dict:
    """name -> (version, build) from `conda list --export`; pip-installed lines have the build pypi_0."""
    pkgs = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version, build = line.strip().split("=")
            pkgs[_norm(name)] = (version, build)
    return pkgs


def _conda_specs() -> list:
    """(name, version or None, build or None) of environment.yml; only the forms the file uses are accepted."""
    deps = yaml.safe_load(ENV_YML.read_text(encoding="utf-8"))["dependencies"]
    out = []
    for d in deps:
        assert isinstance(d, str), f"environment.yml: pip packages belong in requirements-pip.txt, not {d!r}"
        m = re.fullmatch(r"([A-Za-z0-9][\w.-]*)(?:=([^=\s]+))?(?:=([^=\s]+))?", d)
        assert m, f"environment.yml: unsupported spec {d!r} (use name=version or name=version=build)"
        out.append((_norm(m[1]), m[2], m[3]))
    return out


def _pip_specs() -> list:
    out = []
    for line in PIP_TXT.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.fullmatch(r"([A-Za-z0-9][\w.-]*)==([^\s;]+)", line)
        assert m, f"requirements-pip.txt: unsupported line {line!r} (use name==version)"
        out.append((_norm(m[1]), m[2], None))
    return out


def _pinned() -> dict:
    """name -> version of every package the two env files pin to a version."""
    return {n: v for n, v, _ in _conda_specs() + _pip_specs() if v and v != "*"}


def test_versions():
    import caiman, numpy
    assert caiman.__version__.startswith("1.13.2")
    assert int(numpy.__version__.split(".")[0]) >= 2


def test_linalg_does_not_crash():
    import numpy as np, scipy.linalg as sl
    a = np.random.default_rng(0).random((80, 80))
    assert sl.inv(a).shape == (80, 80)
    assert np.linalg.lstsq(a, a[:, 0], rcond=None)[0].shape == (80,)


def test_openblas_coretype_applied():
    """The root conftest sets OPENBLAS_CORETYPE before numpy loads (the Zen 4 workaround, and the kernels the
    golden record was made with). A pytest plugin that imports numpy before any conftest (zarr's) undoes it
    silently outside an activated env; pytest.ini blocks that plugin."""
    import os
    import pytest
    from threadpoolctl import threadpool_info
    blas = [d for d in threadpool_info() if d.get("internal_api") == "openblas"]
    if not blas:
        pytest.skip("numpy does not use OpenBLAS here")
    want = os.environ["OPENBLAS_CORETYPE"].lower()
    assert [d["architecture"].lower() for d in blas] == [want] * len(blas)


def test_ffv1_codec_available():
    import av
    assert "ffv1" in av.codecs_available and "libx264" in av.codecs_available


def test_production_model_loads():
    import joblib, numpy as np
    m = joblib.load(REPO / "production_models" / "ebm_v9_iter8.pkl")
    assert m.predict_proba(np.zeros((2, len(m.feature_names_in_)))).shape == (2, 2)


def test_env_files_match_lock():
    """Every package environment.yml and requirements-pip.txt name is in the lock with the pinned version (and build
    pattern); an unpinned one is allowed only as a build selector (libblas=*=*openblas) or in UNPINNED_OK."""
    lock = _lock()
    wrong = []
    for name, version, build in _conda_specs() + _pip_specs():
        if name not in lock:
            wrong.append(f"{name}: not in {LOCK.name}")
            continue
        have_version, have_build = lock[name]
        if version is None or (version == "*" and build is None):
            if name not in UNPINNED_OK:
                wrong.append(f"{name}: no version (lock has {have_version})")
            continue
        if version != "*" and version != have_version:
            wrong.append(f"{name}: pinned {version}, lock has {have_version}")
        if build is not None and not fnmatch.fnmatchcase(have_build, build):
            wrong.append(f"{name}: build {build}, lock has {have_build}")
    assert not wrong, "env files and lock disagree:\n" + "\n".join(wrong)


def _program_imports() -> dict:
    """Third-party top-level module -> files that import it, over the root modules, bearmind/ and tests/."""
    files = sorted(REPO.glob("*.py")) + sorted((REPO / "bearmind").rglob("*.py")) + sorted((REPO / "tests").rglob("*.py"))
    local = ({p.stem for p in REPO.glob("*.py")} | {d.name for d in REPO.iterdir() if d.is_dir()}
             | {p.stem for p in (REPO / "tests").rglob("*.py")})
    found = {}
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"), str(f))):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for top in {n.split(".")[0] for n in names}:
                if top not in sys.stdlib_module_names and top not in local:
                    found.setdefault(top, set()).add(f.relative_to(REPO).as_posix())
    return found


def test_program_imports_are_pinned():
    imports = _program_imports()
    assert {"numpy", "caiman", "PySide6"} <= set(imports), f"import scan found too little: {sorted(imports)}"
    unknown = {m: sorted(f)[:3] for m, f in imports.items() if m not in IMPORT_PACKAGE and m not in OPTIONAL_IMPORTS}
    assert not unknown, f"third-party imports not in IMPORT_PACKAGE (pin them and add them there): {unknown}"
    pinned = _pinned()
    loose = sorted({p for m, p in IMPORT_PACKAGE.items() if m in imports and _norm(p) not in pinned})
    assert not loose, f"imported by the program but not pinned in environment.yml or requirements-pip.txt: {loose}"


def test_env_file_sets_openblas_coretype():
    """A fresh env gets the Zen 4 workaround as a conda env variable (set on `conda activate` and `conda run`), with
    the value bearmind/__init__.py and the root conftest set for a bare python."""
    want = re.search(r'setdefault\("OPENBLAS_CORETYPE", "(\w+)"\)', (REPO / "bearmind" / "__init__.py").read_text(
        encoding="utf-8"))[1]
    assert f'setdefault("OPENBLAS_CORETYPE", "{want}")' in (REPO / "conftest.py").read_text(encoding="utf-8")
    env = yaml.safe_load(ENV_YML.read_text(encoding="utf-8"))
    assert (env.get("variables") or {}).get("OPENBLAS_CORETYPE") == want


def test_install_check_script_checks_the_bearmind_env():
    """install2_pip.bat ends with test_env.py: it checks this env's packages (not the old caiman env's) and every
    package the program imports imports here."""
    sys.path.insert(0, str(REPO))
    try:
        import test_env
    finally:
        sys.path.remove(str(REPO))
    names = {m for m, _ in test_env.REQUIRED_PACKAGES}
    assert not names & {"tensorflow", "jupyter_bokeh", "peakutils"}
    assert set(IMPORT_PACKAGE) - {"pylab", "shiboken6"} <= names  # pylab, shiboken6: parts of matplotlib, pyside6
    assert test_env.check_imports()[1] == []
    assert '"%PYTHON%" -u test_env.py' in (REPO / "install2_pip.bat").read_text(encoding="ascii")


def test_installers_and_guide_use_the_env_files():
    """The environment.yml header, the two installers and the student guide install the same way: the conda env
    from environment.yml, then requirements-pip.txt with pip, into the env environment.yml names."""
    env_text = ENV_YML.read_text(encoding="utf-8")
    name = yaml.safe_load(env_text)["name"]
    guide = (REPO / "docs" / "students" / "README.md").read_text(encoding="utf-8")
    install1 = (REPO / "install1_conda.bat").read_text(encoding="ascii")
    install2 = (REPO / "install2_pip.bat").read_text(encoding="ascii")
    for text in (env_text, guide):
        assert "env create -f environment.yml" in text and f"conda activate {name}" in text
        assert "pip install -r requirements-pip.txt" in text
    assert "env create -f environment.yml" in install1 and f"set ENV_NAME={name}" in install1
    assert "pip install -r requirements-pip.txt" in install2 and f"set ENV_NAME={name}" in install2
    assert not (REPO / "bearmind2025.yml").exists()  # the old caiman env export: nothing installs from it


def test_gitignore_hides_a_root_venv_not_the_lock():
    """env/ at the root holds the lock file; a `python -m venv env` there (env/Scripts, env/Lib, env/pyvenv.cfg)
    must stay ignored, a new file next to the lock must not."""
    import shutil
    import subprocess
    if shutil.which("git") is None or not (REPO / ".git").exists():
        pytest.skip("not a git checkout")

    def ignored(path):
        return subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "--no-index", path]).returncode == 0

    venv = ["env/Scripts/python.exe", "env/Lib/site-packages/x.py", "env/Include/x.h", "env/pyvenv.cfg"]
    assert [p for p in venv if not ignored(p)] == []
    assert not ignored("env/bearmind-lock.txt") and not ignored("env/other-lock.txt")
