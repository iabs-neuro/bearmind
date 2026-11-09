"""Export of one processed pickle the way the export stage does it, without a series.

The export stage (bearmind.stages.export.run) needs a series registry and timestamp files. The tests that check a
real autoinspected session from output/ have only the processed pickle, so they use the stage's own pieces: its
component rule (_ml_filter), its metadata input and filters writer, and the export library calls it makes. The frame
rate comes from fps_data.csv (export_estimates_data.get_fps), where the stage takes the timestamps' rate.
"""
from pathlib import Path

import export_estimates_data as old
from bearmind.stages import export as stage


def export_processed(processed, out) -> str:
    """Write <name>_data.npz, <name>_metadata.json and <name>_filters.mat of `processed` into `out`; returns
    the session name the files carry (export_estimates_data.extract_session_name)."""
    out = Path(out)
    est = old.load_estimates(Path(processed))
    name = old.extract_session_name(est)
    idx, ml_info = stage._ml_filter(est)
    data = old.extract_data(est, idx)
    if not len(idx):
        data = stage._empty_data(est, data)
    meta = old.build_metadata(stage._metadata_input(est), old.get_fps(name), name, False, None, ml_info, idx)
    old.export(data, meta, name, out)
    stage._filters(est, idx, out / f"{name}_filters.mat")
    return name
