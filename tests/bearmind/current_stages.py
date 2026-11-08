"""Test helper: stage records that the runner counts as up to date, by its own rule (runner.StageView.current):
DONE, the hash of the current configs, and the required output files present.

Tests that build records by hand (DONE without a hash) would otherwise show every such stage as out of date in
the window and in status, as the batch would redo it.
"""
import copy

from bearmind import registry, runner
from bearmind.registry import STAGES, State

_FILES = {
    "mc_video": lambda p, sid: p.mc_video(sid),
    "mc_shifts": lambda p, sid: p.mc_shifts(sid),
    "threshold_maps": lambda p, sid: p.threshold_maps(sid),
    "estimates": lambda p, sid: p.estimates(sid),
    "processed": lambda p, sid: p.inspection_dir(sid) / f"{sid}_processed.pickle",
    "npz": lambda p, sid: p.export_npz(sid),
    "json": lambda p, sid: p.export_json(sid),
    "timestamps": lambda p, sid: p.export_timestamps(sid),
}


def mark_current(paths, rec, stages=None):
    """The record's DONE main stages (or `stages`, which are set DONE) get the hash of the current configs and
    their required outputs; an output file that does not exist is made empty. Changes `rec` in place and
    returns it; it is not saved."""
    if stages is None:
        names = [s for s in STAGES if rec.stages.get(s) is not None and rec.stages[s].state is State.DONE]
    else:
        names = [s for s in STAGES if s in stages]
    view = runner.StageView(paths)
    for stage in names:
        # a copy: tests share one StageRecord between several records
        st = rec.stages[stage] = copy.deepcopy(rec.stages[stage])
        st.state, st.error = State.DONE, None
        for key in runner.REQUIRED[stage]:
            f = _FILES[key](paths, rec.sid)
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_bytes(b"x")
            st.outputs[key] = paths.rel(f)
        h = view.hash(rec, stage)
        assert h is not None, f"{rec.sid}: the stage hashes cannot be computed: {view.errors.get(rec.sid)}"
        st.params_hash = h
    return rec


def make_current(paths, sid, *stages):
    """mark_current on the saved record of `sid`, for the named main stages (in stage order)."""
    rec = mark_current(paths, registry.load_record(paths, sid), stages)
    registry.update_record(paths, sid, lambda r: r.stages.update({s: rec.stages[s] for s in stages}))
