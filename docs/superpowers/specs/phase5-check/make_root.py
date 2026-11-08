r"""Step 1 / S1: the check root phase5_short = copies of series_short\_bearmind, \3_MC and \4_Estimates (mtime
kept, shutil.copy2) + a 1_Raw junction to series_short\1_Raw. Reads series_short only. Refuses to run when the
root already holds any of the four entries."""
import shutil
import subprocess
from pathlib import Path

SRC = Path(r"H:\Projects\_bearmind_test\series_short")
DST = Path(r"H:\Projects\_bearmind_test\phase5_short")
DST.mkdir(exist_ok=True)
for name in ("_bearmind", "3_MC", "4_Estimates", "1_Raw"):
    if (DST / name).exists():
        raise SystemExit(f"[FAILED] {DST / name} exists already")
for name in ("_bearmind", "3_MC", "4_Estimates"):
    shutil.copytree(SRC / name, DST / name, copy_function=shutil.copy2)
    print(f"[OK] copied {name}")
r = subprocess.run(["cmd", "/c", "mklink", "/J", str(DST / "1_Raw"), str(SRC / "1_Raw")], capture_output=True,
                   text=True)
print(r.stdout.strip().encode("ascii", "backslashreplace").decode(), r.returncode)
