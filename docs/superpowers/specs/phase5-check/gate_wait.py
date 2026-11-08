r"""Check-root setup (phase5_short only): both sessions wait at the gate as right after a new CNMF-E.

rerun --from gate (gate, autoinspect, export -> PENDING; a rerun_from decision), then the old
manual_inspection decisions ("not_needed" of 2026-11-01 and 2026-11-07) are removed from the copy's registry:
otherwise the newest one (newer than cnmf.finished) still passes the gate (R28)."""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind import actions, registry  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402

ROOT = Path(r"H:\Projects\_bearmind_test\phase5_short")
assert ROOT.name == "phase5_short"
paths = SeriesPaths(ROOT)
sids = ["NOF_H01_1D", "NOF_H03_3D"]
print("waiting before:", actions.waiting_gate_sids(paths))
res = actions.rerun_from(paths, sids, "gate", "t7-check")
print("rerun ok:", res.ok, "failed:", res.failed)
for sid in sids:
    removed = []

    def drop(rec):
        keep = []
        for d in rec.decisions:
            (removed if d.what == "manual_inspection" else keep).append(d)
        rec.decisions = keep
    registry.update_record(paths, sid, drop)
    print(sid, "removed decisions:", [(d.value, d.who, d.when) for d in removed])
    rec = registry.load_record(paths, sid)
    print("  stages:", {s: st.state.value for s, st in rec.stages.items()})
print("waiting after:", actions.waiting_gate_sids(paths))
