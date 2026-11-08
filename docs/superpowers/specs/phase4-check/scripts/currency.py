"""Read-only: per session and stage, the registry state and whether the runner counts the stage as up to date
(runner._View.current: DONE, params_hash equal to the hash for the current configs, required outputs present),
plus the first stage the runner would start (runner.next_ready) for all sessions and for each one alone.

usage: python currency.py SERIES
"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind import registry, runner  # noqa: E402
from bearmind.paths import SeriesPaths  # noqa: E402
from bearmind.registry import ALL_STAGES  # noqa: E402

paths = SeriesPaths(Path(sys.argv[1]).resolve())
records, broken = registry.read_all(paths)
view = runner._View(paths)
print(f"series {paths.root}; broken records: {len(broken)}")
for r in sorted(records, key=lambda r: r.sid):
    cells = []
    for s in ALL_STAGES:
        st = r.stages.get(s)
        state = st.state.value if st else "-"
        if s in runner.SIDE_STAGES and not view.side_on(r, s):
            cur = "off"
        else:
            cur = "current" if view.current(r, s) else "NOT current"
            if st is not None and st.state.value == "done" and not view.current(r, s):
                h = view.hash(r, s)
                why = []
                if h != st.params_hash:
                    why.append("hash differs")
                missing = [k for k in runner.REQUIRED[s] if not runner._output_ok(paths, st.outputs, k)]
                if missing:
                    why.append("missing " + ",".join(missing))
                cur += " (" + "; ".join(why) + ")"
        cells.append(f"{s}={state}/{cur}")
    print(r.sid)
    for c in cells:
        print("   ", c)
    print("    errors:", view.errors.get(r.sid))
print("next_ready all:", runner.next_ready(records, paths))
for r in sorted(records, key=lambda r: r.sid):
    print(f"next_ready --sessions {r.sid}:", runner.next_ready(records, paths, sids=[r.sid]))
