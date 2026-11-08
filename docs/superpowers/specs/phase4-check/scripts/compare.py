"""Compare two snapshot.py outputs: count and identity of files (path, size, mtime_ns, sha256) and folders.

usage: python compare.py BEFORE.json AFTER.json
"""
import json
import sys


def main():
    a = json.load(open(sys.argv[1], encoding="utf-8"))
    b = json.load(open(sys.argv[2], encoding="utf-8"))
    ok = True
    for root in sorted(set(a) | set(b)):
        if root not in a or root not in b:
            print(f"[FAILED] {root}: only in one snapshot")
            ok = False
            continue
        fa, fb = a[root]["files"], b[root]["files"]
        da, db = a[root]["dirs"], b[root]["dirs"]
        added = sorted(set(fb) - set(fa))
        removed = sorted(set(fa) - set(fb))
        changed = sorted(k for k in set(fa) & set(fb) if fa[k] != fb[k])
        dchanged = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
        same = not (added or removed or changed or dchanged)
        ok &= same
        print(f"[{'OK' if same else 'FAILED'}] {root}: files {len(fa)} -> {len(fb)}, folders {len(da)} -> {len(db)}, "
              f"added {len(added)}, removed {len(removed)}, changed {len(changed)}, folder mtime changed {len(dchanged)}")
        for k in added[:20]:
            print(f"    added {k}")
        for k in removed[:20]:
            print(f"    removed {k}")
        for k in changed[:20]:
            print(f"    changed {k}: {fa[k]} -> {fb[k]}")
        for k in dchanged[:20]:
            print(f"    folder {k}: {da.get(k)} -> {db.get(k)}")
    print("[OK] identical" if ok else "[FAILED] snapshots differ")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
