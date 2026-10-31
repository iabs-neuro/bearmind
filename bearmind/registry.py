"""Session registry: one JSON file per session (stage states, anomalies, decisions)."""
import json
import os
from dataclasses import asdict, dataclass, field, fields
from enum import Enum

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


def load_record(paths, sid: str) -> SessionRecord:
    f = paths.registry_file(sid)
    with open(f, encoding="utf-8") as fh:
        return _from_json(json.load(fh))


def save_record(paths, rec: SessionRecord) -> None:
    """Write atomically: temp file in the same folder, fsync, os.replace."""
    f = paths.registry_file(rec.sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_to_json(rec), fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, f)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def all_records(paths) -> list[SessionRecord]:
    folder = paths.registry_file("x").parent
    if not folder.is_dir():
        return []
    files = [p for p in folder.glob("*.json") if not p.name.endswith(".tmp")]
    return sorted((_from_json(json.loads(p.read_text(encoding="utf-8")))
                   for p in files), key=lambda r: r.sid)
