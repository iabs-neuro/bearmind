"""Read-only snapshot of folders: relative path, size, mtime_ns and sha256 of every file.

usage: python snapshot.py OUT.json ROOT [ROOT ...]   (stat:ROOT = size and mtime only, no sha256)
Only reads. Folders are listed too (path, mtime_ns), so a new or removed file shows.
"""
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def snap(root: Path, hashed: bool = True) -> dict:
    files, dirs = {}, {}
    for d, subdirs, names in os.walk(root):
        subdirs.sort()
        dp = Path(d)
        dirs[str(dp.relative_to(root))] = os.stat(dp).st_mtime_ns
        for n in sorted(names):
            p = dp / n
            st = os.stat(p)
            files[str(p.relative_to(root))] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256(p) if hashed else None}
    return {"files": files, "dirs": dirs}


def main():
    out = Path(sys.argv[1])
    res = {}
    for r in sys.argv[2:]:
        hashed = not r.startswith("stat:")  # stat:ROOT lists size and mtime only (big videos)
        root = Path(r[5:] if not hashed else r)
        res[str(root)] = snap(root, hashed)
        print(f"[OK] {root}: {len(res[str(root)]['files'])} files, {len(res[str(root)]['dirs'])} folders")
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    blob = json.dumps(res, sort_keys=True).encode()
    print(f"[OK] written {out}; digest {hashlib.sha256(blob).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
