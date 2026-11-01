"""Session registry: one JSON file per session (stage states, anomalies, decisions).

Several processes write the same file (the batch runner, the CLI, later the GUI). A writer that changes a
record while another may also change it uses update_record: under a short file lock it rereads the file,
applies its change to the fresh copy and writes it back, so the other writer's fields survive.
"""
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path

from bearmind import fileio

log = logging.getLogger(__name__)

STAGES = ("crop", "mc", "thresholds", "cnmf", "gate", "autoinspect", "export")


class State(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    WAITING = "waiting"


@dataclass
class Check:
    stage: str
    name: str
    level: str  # "ok" | "warning" | "error"
    value: float | None
    message: str


@dataclass
class Anomaly:
    code: str
    level: str
    message: str
    auto_fixed: bool = False
    data: dict = field(default_factory=dict)


@dataclass
class Decision:
    what: str
    value: str
    who: str
    when: str  # ISO string supplied by the caller


@dataclass
class RawPart:
    folder: str
    chunks: list[str]
    n_frames: int
    shape: tuple[int, int]
    timestamps_file: str | None
    ts_format: str | None
    start_ms: float | None
    metadata_file: str | None
    drop_chunk_tail: bool = False
    drop_last_frame: bool = False
    chunk_frames: list[int] = field(default_factory=list)  # frames per chunk as the container reports


@dataclass
class StageRecord:
    state: State = State.PENDING
    params_hash: str | None = None
    started: str | None = None
    finished: str | None = None
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    error: str | None = None
    params: dict = field(default_factory=dict)  # resolved parameters the stage ran with (provenance)


@dataclass
class SessionRecord:
    sid: str
    mouse: str
    mode: str
    parts: list[RawPart]
    n_frames: int
    fps: float | None
    fps_info: dict
    anomalies: list[Anomaly]
    decisions: list[Decision]
    excluded: bool
    stages: dict[str, StageRecord]


def _build(cls, d: dict):
    """Construct a dataclass from a dict, ignoring keys from newer versions."""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def _to_json(rec: SessionRecord) -> dict:
    d = asdict(rec)  # tuples stay tuples, json writes them as lists
    for st in d["stages"].values():
        st["state"] = State(st["state"]).value
    return d


def _from_json(d: dict) -> SessionRecord:
    parts = []
    for p in d.get("parts", []):
        p = dict(p)
        p["shape"] = tuple(p["shape"])
        parts.append(_build(RawPart, p))
    stages = {}
    for name, s in d.get("stages", {}).items():
        s = dict(s)
        s["state"] = State(s.get("state", State.PENDING.value))
        s["checks"] = [_build(Check, c) for c in s.get("checks", [])]
        stages[name] = _build(StageRecord, s)
    d = dict(d)
    d["parts"] = parts
    d["stages"] = stages
    d["anomalies"] = [_build(Anomaly, a) for a in d.get("anomalies", [])]
    d["decisions"] = [_build(Decision, x) for x in d.get("decisions", [])]
    return _build(SessionRecord, d)


def _read(f) -> SessionRecord:
    return _from_json(json.loads(fileio.read_text(f)))


def _plain(x):
    """json default: numpy scalars / arrays in stage metrics become Python numbers / lists."""
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(f"{type(x).__name__} is not JSON serializable")


def _write(f, rec: SessionRecord) -> None:
    fileio.write_atomic(f, lambda fh: json.dump(_to_json(rec), fh, indent=2, ensure_ascii=False,
                                                default=_plain))


def load_record(paths, sid: str) -> SessionRecord:
    return _read(paths.registry_file(sid))


def save_record(paths, rec: SessionRecord) -> None:
    """Write the whole record atomically (unique temp file, fsync, replace) under the file lock."""
    f = paths.registry_file(rec.sid)
    with fileio.file_lock(f):
        _write(f, rec)


def update_record(paths, sid: str, change) -> SessionRecord:
    """Reread the record under the lock, call change(record) on it, write it back; returns the new record."""
    f = paths.registry_file(sid)
    with fileio.file_lock(f):
        rec = _read(f)
        change(rec)
        _write(f, rec)
    return rec


def replace_broken(paths, rec: SessionRecord, kept) -> bool:
    """Save `rec` in place of its registry file that cannot be read, which is renamed to `kept` (a name that
    is not globbed as a record). False, and nothing changed, when the file reads fine by now."""
    f = paths.registry_file(rec.sid)
    with fileio.file_lock(f):
        try:
            _read(f)
            return False
        except (ValueError, KeyError, TypeError, AttributeError):
            pass
        fileio.replace(f, kept)
        _write(f, rec)
    return True


_reported: set = set()  # broken files already logged: the runner calls all_records every loop


def read_all(paths) -> tuple[list[SessionRecord], list[tuple[Path, str]]]:
    """All readable records sorted by sid, and every registry file that cannot be read with the reason."""
    folder = paths.registry_file("x").parent
    if not folder.is_dir():
        return [], []
    out, broken = [], []
    for p in sorted(folder.glob("*.json")):
        try:
            out.append(_read(p))
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
            broken.append((p, " ".join(f"{type(e).__name__}: {e}".split())))
    return sorted(out, key=lambda r: r.sid), broken


def all_records(paths) -> list[SessionRecord]:
    """All readable records, sorted by sid; a broken file is logged (once per file version) and skipped."""
    out, broken = read_all(paths)
    for p, why in broken:
        try:
            st = p.stat()
            key = (str(p), st.st_size, st.st_mtime_ns)
        except OSError:
            key = (str(p), None, None)
        if key not in _reported:
            _reported.add(key)
            log.error("registry file skipped, cannot read %s: %s", p.name, why)
    return out
