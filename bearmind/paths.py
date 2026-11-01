"""Folder layout of one series (single source of truth for all paths)."""
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeriesPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root))

    # --- input / intermediate stages ---
    @property
    def raw_root(self) -> Path:
        return self.root / "1_Raw"

    @property
    def timestamps_root(self) -> Path:
        return self.root / "0_TimeStamps"

    def mc_video(self, sid: str) -> Path:
        return self.root / "3_MC" / f"{sid}_MC.avi"

    def mc_shifts(self, sid: str) -> Path:
        return self.root / "3_MC" / f"{sid}_MC_shifts.npy"

    def estimates(self, sid: str) -> Path:
        return self.root / "4_Estimates" / f"{sid}_estimates.pickle"

    def inspection_dir(self, sid: str) -> Path:
        return self.root / "5_Inspection" / sid

    def manual_result(self, sid: str) -> Path:
        return self.inspection_dir(sid) / f"{sid}_manual_estimates.pickle"

    # --- export ---
    def export_npz(self, sid: str) -> Path:
        return self.root / "6_Export" / "npz" / f"{sid}_data.npz"

    def export_json(self, sid: str) -> Path:
        return self.root / "6_Export" / "npz" / f"{sid}_metadata.json"

    def export_timestamps(self, sid: str) -> Path:
        return self.root / "6_Export" / "npz" / f"{sid}_timestamps.csv"

    def export_filters(self, sid: str) -> Path:
        return self.root / "6_Export" / "filters" / f"{sid}_filters.mat"

    # --- previews ---
    def preview(self, sid: str, kind: str) -> Path:
        return self.root / "Preview" / f"{sid}_{kind}.mp4"

    def residual_png(self, sid: str) -> Path:
        return self.root / "Preview" / f"{sid}_cnmf_residual.png"

    # --- service folder ---
    @property
    def _service(self) -> Path:
        return self.root / "_bearmind"

    def registry_file(self, sid: str) -> Path:
        return self._service / "registry" / f"{sid}.json"

    @property
    def series_config(self) -> Path:
        return self._service / "config" / "series.yaml"

    def mouse_config(self, mouse: str) -> Path:
        return self._service / "config" / "mice" / f"{mouse}.yaml"

    def session_config(self, sid: str) -> Path:
        return self._service / "config" / "sessions" / f"{sid}.yaml"

    def threshold_maps(self, sid: str) -> Path:
        return self._service / "thresholds" / f"{sid}_maps.npz"

    @property
    def logs_dir(self) -> Path:
        return self._service / "logs"

    @property
    def series_log(self) -> Path:
        return self.logs_dir / "series.log"

    def session_log(self, sid: str) -> Path:
        return self.logs_dir / f"{sid}.log"

    @property
    def heartbeat(self) -> Path:
        return self._service / "heartbeat.json"

    @property
    def lock(self) -> Path:
        return self._service / "run.lock"

    @property
    def summary_csv(self) -> Path:
        return self._service / "summary.csv"

    @property
    def scan_unrecognized_csv(self) -> Path:
        return self._service / "scan_unrecognized.csv"

    # --- machine temp folder ---
    def temp_dir(self, temp_root, sid: str) -> Path:
        """The session's folder under the machine temp folder, keyed by the series too: two series with the
        same session id on one machine never share it, or remove each other's."""
        key = hashlib.sha1(os.path.normcase(os.path.abspath(self.root)).encode("utf-8")).hexdigest()[:8]
        return Path(temp_root) / f"{key}_{sid}"

    def rel(self, p) -> str:
        """Path relative to the series root, with '/' separators."""
        return Path(p).relative_to(self.root).as_posix()
