"""Pipeline stages; each returns a StageResult."""
from dataclasses import dataclass, field

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
        self.what = what
