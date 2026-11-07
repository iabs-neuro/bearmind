"""Snapshot a folder tree (relative path, size, mtime_ns of every file and folder) or compare two snapshots.

usage: snapshot.py take ROOT OUT.tsv | snapshot.py diff A.tsv B.tsv
"""
import os
import sys


def take(root, out):
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for d, dirs, files in os.walk(root):
            dirs.sort()
            st = os.stat(d)
            f.write(f"D\t{os.path.relpath(d, root)}\t0\t{st.st_mtime_ns}\n")
            for name in sorted(files):
                p = os.path.join(d, name)
                st = os.stat(p)
                f.write(f"F\t{os.path.relpath(p, root)}\t{st.st_size}\t{st.st_mtime_ns}\n")
                n += 1
    print(f"[OK] {n} files written to {out}")


def diff(a, b):
    la = open(a, encoding="utf-8").read().splitlines()
    lb = open(b, encoding="utf-8").read().splitlines()
    sa, sb = set(la), set(lb)
    print(f"lines: {len(la)} vs {len(lb)}; only in first: {len(sa - sb)}; only in second: {len(sb - sa)}")
    for x in sorted(sa - sb)[:20]:
        print("  - " + x)
    for x in sorted(sb - sa)[:20]:
        print("  + " + x)
    print("[OK] identical" if la == lb else "[FAILED] different")


if __name__ == "__main__":
    {"take": take, "diff": diff}[sys.argv[1]](sys.argv[2], sys.argv[3])
