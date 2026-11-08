"""S1: the check root phase4_short = copies of series_short\\_bearmind and \\3_MC (mtime kept) + 1_Raw junction.

Reads series_short only. Refuses to run when the root already holds any of the three entries.
"""
import shutil
import subprocess
from pathlib import Path

SRC = Path(r"H:\Projects\_bearmind_test\series_short")
DST = Path(r"H:\Projects\_bearmind_test\phase4_short")
DST.mkdir(exist_ok=True)
for name in ("_bearmind", "3_MC", "1_Raw"):
    if (DST / name).exists():
        raise SystemExit(f"[FAILED] {DST / name} exists already")
for name in ("_bearmind", "3_MC"):
    shutil.copytree(SRC / name, DST / name, copy_function=shutil.copy2)
    print(f"[OK] copied {name}")
r = subprocess.run(["cmd", "/c", "mklink", "/J", str(DST / "1_Raw"), str(SRC / "1_Raw")], capture_output=True,
                   text=True)
print(r.stdout.strip().encode("ascii", "backslashreplace").decode(), r.returncode)
