"""Pipeline stages; each returns a StageResult."""
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from bearmind import config
from bearmind.registry import Anomaly


@dataclass
class StageResult:
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    anomalies: list[Anomaly] = field(default_factory=list)


class WaitingForInput(Exception):
    """A stage needs a user decision (e.g. thresholds); the scheduler marks it WAITING instead of FAILED."""

    def __init__(self, what: str, detail: str | None = None):
        super().__init__(f"waiting for {what}" + (f": {detail}" if detail else ""))
        self.what, self.detail = what, detail

    def __reduce__(self):
        # pickled from an isolated stage process: the default would call WaitingForInput(<message>)
        return type(self), (self.what, self.detail)


class NotEnoughTempSpace(RuntimeError):
    """The temp folder's disk has less free space than the stage's temp files need."""


def require_temp_space(folder, need_bytes: float, stage: str) -> None:
    """Fail the stage at once, before any heavy work, when the disk of the temp folder has less free space
    than `need_bytes`: a disk that fills up in the middle of an hour-long CNMF-E costs the hour."""
    folder = Path(folder)
    probe = folder
    while not probe.exists() and probe.parent != probe:  # the session's folder is made only later
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < need_bytes:
        raise NotEnoughTempSpace(
            f"not enough free space for the {stage} temp files in {folder}: about {need_bytes / 1e9:.1f} GB "
            f"needed, {free / 1e9:.1f} GB free; set temp_dir in {config.machine_file()} to a folder on a "
            f"disk with more free space")
