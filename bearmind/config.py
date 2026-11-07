"""Layered configs: defaults < series < mouse < session, plus machine settings."""
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

import psutil
import yaml

from . import fileio, naming

_DEFAULTS = Path(__file__).with_name("defaults.yaml")


class ConfigError(ValueError):
    """A config file that cannot be parsed; the message starts with the file."""


_PARSED: dict = {}  # path -> (text, parsed): the runner resolves every session's config each loop


def _read_yaml(path) -> dict:
    """Parsed YAML ({} when absent); a private copy, reparsed only when the file content changed."""
    path = Path(path)
    try:
        text = fileio.read_text(path)
    except FileNotFoundError:
        return {}
    key = str(path)
    hit = _PARSED.get(key)
    if hit is None or hit[0] != text:  # content, not mtime: an edit may keep size and mtime
        try:
            parsed = yaml.safe_load(text) or {}
        except yaml.YAMLError as e:  # the parser names only "<unicode string>": the operator needs the file
            raise ConfigError(f"{path}: {' '.join(str(e).split())}") from e
        hit = (text, parsed)
        _PARSED[key] = hit
    return copy.deepcopy(hit[1])


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


def machine_file() -> Path:
    """machine.yaml: temp folder, cores, memory of this computer."""
    return _home() / "machine.yaml"


def machine_config() -> dict:
    f = machine_file()
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
    """Merge `patch` into the YAML file: reread under the file lock, write atomically (fileio)."""
    path = Path(path)
    with fileio.file_lock(path):
        data = deep_merge(_read_yaml(path), patch)
        fileio.write_atomic(path, lambda f: yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True))


def params_hash(params: dict) -> str:
    return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()


def fallback_gsig(gsigs) -> int:
    """gSig when nothing suggests one: 4, else the middle of the range. One rule for the thresholds stage
    (its suggestion maps) and the Thresholds window (start values)."""
    gsigs = list(gsigs)
    return 4 if 4 in gsigs else gsigs[len(gsigs) // 2]
