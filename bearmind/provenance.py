"""Provenance block written into the export json: code version, packages, model, inputs, decisions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = ("caiman", "numpy", "driada", "interpret")


def _git(*args) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                           timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout if r.returncode == 0 else None


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def model_file(cfg) -> Path:
    """The ML model file; a relative model_path is relative to the repo root (production_models/)."""
    f = Path(cfg["autoinspect"]["model_path"])
    return f if f.is_absolute() else REPO_ROOT / f


def collect(paths, rec, cfg) -> dict:
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    model = cfg["autoinspect"]["model_path"]
    info = rec.fps_info or {}
    return {
        "code": {"git_commit": commit.strip() if commit else None,
                 "git_dirty": bool(status.strip()) if status is not None else None},
        "packages": {**{p: _version(p) for p in PACKAGES}, "python": platform.python_version()},
        "model": {"path": model, "sha256": _sha256(model_file(cfg))},
        "inputs": [p.folder for p in rec.parts],
        "mode": rec.mode,
        "fps": {"value": rec.fps, "source": info.get("source"),
                "n_outliers": info.get("n_outliers"), "max_gap_ms": info.get("max_gap_ms")},
        # resolved parameters each stage ran with (runner.stage_params), not their hashes
        "stage_params": {n: s.params for n, s in rec.stages.items() if s.params},
        "decisions": [asdict(d) for d in rec.decisions],
    }
