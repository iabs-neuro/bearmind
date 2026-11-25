"""Prepare a smaller CNMF-E view while keeping the original estimates authoritative."""
import copy
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.sparse import issparse

from bearmind import fileio


def light_path(full: Path) -> Path:
    return full.with_stem(full.stem + "_light")


def _signature(path):
    st = path.stat()
    return dict(name=path.name, size=st.st_size, mtime_ns=st.st_mtime_ns)


def _manifest(light):
    return light.with_suffix(".json")


def current(full: Path) -> Path | None:
    light = light_path(full)
    try:
        data = json.loads(fileio.read_text(_manifest(light)))
        if data == {"full": _signature(full), "light": _signature(light)}:
            return light
    except (OSError, ValueError):
        pass
    return None


def _prune(est, signature):
    light = copy.copy(est)
    accepted = getattr(light, "idx_components", None)
    accepted = np.arange(est.A.shape[1]) if accepted is None else np.asarray(accepted, dtype=int)
    if np.any(accepted < 0) or np.any(accepted >= est.A.shape[1]) or len(np.unique(accepted)) != len(accepted):
        raise ValueError("Invalid accepted component indices.")
    light.coordinates = None  # cached contours refer to the full-file numbering
    spikes = getattr(light, "S", None)
    sparse_spikes = spikes if issparse(spikes) else None
    if getattr(light, "cnn_preds", None) is not None and not np.size(light.cnn_preds):
        light.cnn_preds = None
    if sparse_spikes is not None:
        light.S = None
    light.select_components(idx_components=accepted, save_discarded_components=False)
    if sparse_spikes is not None:
        light.S = sparse_spikes[accepted]
    light.idx_components = np.arange(len(accepted))
    light.idx_components_bad = np.empty(0, dtype=int)
    light.discarded_components = None
    light.light_of = signature
    light.original_component_indices = accepted.copy()
    for name in ("C", "S", "YrA", "SNR_comp", "r_values"):
        value = getattr(light, name, None)
        if value is not None and value.shape[0] != len(accepted):
            raise ValueError(f"{name} is not aligned with the accepted components.")
    return light


def make(full: Path, estimates=None) -> tuple[Path, bool]:
    """Return the light file and whether it was created. Repeated calls reuse a current file."""
    light = light_path(full)
    with fileio.file_lock(light):
        if current(full) is not None:
            return light, False
        signature = _signature(full)
        if estimates is None:
            with full.open("rb") as stream:
                estimates = pickle.load(stream)
        pruned = _prune(estimates, signature)
        if _signature(full) != signature:
            raise ValueError("The full estimates changed while they were being read.")
        fileio.write_atomic(light, lambda fh: pickle.dump(pruned, fh, protocol=pickle.HIGHEST_PROTOCOL), binary=True)
        if _signature(full) != signature:
            raise ValueError("The full estimates changed while the light file was being written.")
        manifest = {"full": signature, "light": _signature(light)}
        fileio.write_atomic(_manifest(light), lambda fh: json.dump(manifest, fh))
    return light, True


def _source_pin(paths, sid):
    return paths.inspection_dir(sid) / "inspection_source.json"


def remember_input(paths, sid, estimates):
    signature = getattr(estimates, "light_of", None)
    pin = _source_pin(paths, sid)
    if signature is not None:
        fileio.write_atomic(pin, lambda fh: json.dump(signature, fh))
    elif pin.exists():
        pin.unlink()


def inspection_input(paths, sid, full):
    pinned = False
    has_saved = paths.manual_result(sid).is_file() or any(paths.inspection_progress_dir(sid).glob("*.pickle"))
    if has_saved:
        try:
            pinned = json.loads(fileio.read_text(_source_pin(paths, sid))) == _signature(full)
        except (OSError, ValueError):
            pass
        if not pinned:
            return full
    light = current(full)
    if pinned and light is None:
        raise ValueError("This inspection was started on the light estimates. Rebuild that file before continuing.")
    return light or full
