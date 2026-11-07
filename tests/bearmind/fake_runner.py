"""Stand-in for a live batch: takes the real run.lock and writes a real heartbeat, then waits for stdin to close.

Usage: python fake_runner.py SERIES_ROOT SID STAGE STATUS   (SID "-": no current stage)
Prints "ready" once the lock is held and the heartbeat is written. When stdin closes it ends like the real
runner does: an idle heartbeat with the exit reason, then the lock is removed. Killed, it leaves both.
"""
import sys

from bearmind import runner
from bearmind.paths import SeriesPaths

paths = SeriesPaths(sys.argv[1])
sid, stage, status = sys.argv[2:5]
lock = runner._acquire_lock(paths)
hb = runner._Heartbeat(paths)
hb.update(status=status, current=None if sid == "-" else {"sid": sid, "stage": stage})
print("ready", flush=True)
sys.stdin.read()
hb.update(status="idle", current=None, exit="finished")
runner._release_lock(paths, lock)
