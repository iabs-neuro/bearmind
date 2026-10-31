"""Layered configs: defaults < series < mouse < session, plus machine settings."""
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import psutil
import yaml

from . import naming

_DEFAULTS = Path(__file__).with_name("defaults.yaml")


def _read_yaml(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, over: dict) -> dict:
    """Nested dicts merge recursively; lists and scalars from `over` replace."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_defaults() -> dict:
    return _read_yaml(_DEFAULTS)


def _home() -> Path:
    # BEARMIND_HOME lets tests keep machine.yaml out of the real home directory.
    return Path(os.environ.get("BEARMIND_HOME") or Path.home() / ".bearmind")


def machine_config() -> dict:
    f = _home() / "machine.yaml"
    if not f.exists():
        ram_gb = round(psutil.virtual_memory().total / 1024 ** 3, 1)
        update_yaml(f, {
            "temp_dir": str(Path(tempfile.gettempdir()) / "bearmind"),
            "n_cpus": psutil.cpu_count(logical=True) or 1,
            "ram_gb": ram_gb,
            "cnmf_ram_budget_gb": 0.8 * ram_gb,
        })
    return _read_yaml(f)


def resolve(paths, sid=None, mouse=None) -> dict:
    if sid is not None and mouse is None:
        mouse = naming.mouse_id(sid)
    cfg = load_defaults()
    layers = [paths.series_config]
    if mouse:
        layers.append(paths.mouse_config(mouse))
    if sid:
        layers.append(paths.session_config(sid))
    for f in layers:
        cfg = deep_merge(cfg, _read_yaml(f))
    cfg["machine"] = machine_config()
    return cfg


def update_yaml(path, patch: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = deep_merge(_read_yaml(path), patch)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
