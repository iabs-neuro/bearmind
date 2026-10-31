"""Pipeline stages; each returns a StageResult."""
from dataclasses import dataclass, field

from bearmind.registry import Anomaly


@dataclass
class StageResult:
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    anomalies: list[Anomaly] = field(default_factory=list)
