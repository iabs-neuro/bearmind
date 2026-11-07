"""Markdown crop table (Russian headers) from the crop_table.py outputs."""
import sys
from pathlib import Path

S = Path(__file__).resolve().parent
rows = ["| Серия | Сессия | T | было | стало | освещённая область | поле | fov_method | pos/neg | кроп, с |",
        "|---|---|---|---|---|---|---|---|---|---|"]


def add(series, fname, fix=None):
    for line in (S / fname).read_text(encoding="utf-8").splitlines():
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        sid, T, frame, before, after, lit, fov, meth, pn, secs = c[:10]
        if fix and sid in fix:
            before = fix[sid]
        rows.append(f"| {series} | {sid} | {T} | {before.replace('x', '×')} | {after.replace('x', '×')} | "
                    f"{lit.replace('x', '×')} | {fov.replace('x', '×')} | {meth} | {pn} | {secs} |")


add("series_lnof", "table_lnof.txt", {"LNOF_J01_1D": "567x520@0,0"})
add("crop_check_series1", "table_series1.txt")
if (S / "table_short.txt").is_file():
    add("series_short", "table_short.txt")
add("crop_check_lt1", "table_lt1.txt")
(S / "crop_table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
print(len(rows) - 2, "rows")
