"""Box before / after per session: crop records of a check root against the old record (series1, series_short) or,
where no old record exists (LT1), against the brightness rule's box from the same detection (max fov_lit of the
mouse + 2 * margin). Also the crop stage time of each session (finished - started). Read-only.

usage: crop_table.py CHECK_ROOT [OLD_ROOT | -] [OLD_REGISTRY_DIR]
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind.crop import mouse_box_size
from bearmind.naming import mouse_id


def load(reg):
    return {f.stem: json.loads(f.read_text()) for f in sorted(Path(reg).glob("*.json"))}


def secs(st):
    f = "%Y-%m-%dT%H:%M:%S.%f"
    return (datetime.strptime(st["finished"], f) - datetime.strptime(st["started"], f)).total_seconds()


def main():
    new = load(Path(sys.argv[1]) / "_bearmind" / "registry")
    old = None
    if len(sys.argv) > 2 and sys.argv[2] != "-":
        old = load(Path(sys.argv[2]) / "_bearmind" / "registry")
    if len(sys.argv) > 3:
        old = load(sys.argv[3])
    by_mouse = {}
    for sid, r in new.items():
        by_mouse.setdefault(mouse_id(sid), []).append(r["stages"]["crop"]["metrics"])
    bright = {}
    for mouse, ms in by_mouse.items():
        fr = (min(m["frame_shape"][0] for m in ms), min(m["frame_shape"][1] for m in ms))
        bright[mouse] = mouse_box_size([m["fov_lit"] for m in ms if m.get("fov_lit")], fr, 10)
    rows = []
    for sid, r in new.items():
        st = r["stages"]["crop"]
        m = st["metrics"]
        b = m["box"]
        if old is not None and sid in old:
            ob = old[sid]["stages"].get("crop", {}).get("metrics", {}).get("box")
            before = f"{ob[2]}x{ob[3]}@{ob[0]},{ob[1]}" if ob else "-"
        else:
            bh, bw = bright[mouse_id(sid)]
            before = f"{bh}x{bw} (brightness)"
        lit, fov = m.get("fov_lit"), m.get("fov")
        rows.append((sid, r["n_frames"], "x".join(map(str, m["frame_shape"])), before,
                     f"{b[2]}x{b[3]}@{b[0]},{b[1]}" if b else "none",
                     f"{lit[2]}x{lit[3]}" if lit else "-", f"{fov[2]}x{fov[3]}" if fov else "-",
                     m["fov_method"], m.get("activity_pos_neg"), round(secs(st)), len([1])))
        print("| " + " | ".join(str(x) for x in rows[-1][:-1]) + " |")


if __name__ == "__main__":
    main()
