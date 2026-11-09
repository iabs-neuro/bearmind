"""Wait until batch_autoinspect_all completes (by tailing its log), then run
batch_export_all and write a final summary report.

Designed to be started as a separate background task while the autoinspect
is already running. Polls the autoinspect log file every POLL_SEC seconds
for the "ALL DONE" marker. Once detected, runs batch_export_all.py and
captures its log. Writes a final morning report consolidating both.

Usage:
    python chain_run_export_after_autoinspect.py
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
AUTOINSPECT_LOG = ROOT / 'batch_autoinspect_all.log'
EXPORT_LOG = ROOT / 'batch_export_all.log'
MORNING_REPORT = ROOT / 'overnight_report.txt'

POLL_SEC = 60                # how often to poll autoinspect log
MAX_WAIT_HOURS = 16          # safety cap (script exits after this)
DONE_MARKER_RE = re.compile(r'^ALL DONE in [\d.]+ min', re.MULTILINE)
DATASET_DONE_RE = re.compile(r'^\[(CC|RFC|FOF)\] DONE: ok=(\d+), skip=(\d+), fail=(\d+)',
                              re.MULTILINE)


def ts() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def wait_for_autoinspect() -> bool:
    deadline = time.time() + MAX_WAIT_HOURS * 3600
    last_size = -1
    while time.time() < deadline:
        if not AUTOINSPECT_LOG.exists():
            print(f'[{ts()}] waiting for {AUTOINSPECT_LOG} to appear...', flush=True)
            time.sleep(POLL_SEC)
            continue
        size = AUTOINSPECT_LOG.stat().st_size
        text = AUTOINSPECT_LOG.read_text(encoding='utf-8', errors='replace')
        if DONE_MARKER_RE.search(text):
            print(f'[{ts()}] autoinspect ALL DONE detected ({size} bytes)', flush=True)
            return True
        if size != last_size:
            last_size = size
            ok_count = len(re.findall(r'^\[[^\]]+\]\[\d+/\d+\] OK ', text, flags=re.MULTILINE))
            fail_count = len(re.findall(r'^\[[^\]]+\]\[\d+/\d+\] FAIL ', text, flags=re.MULTILINE))
            print(f'[{ts()}] still running... so far: ok={ok_count}, fail={fail_count}',
                  flush=True)
        time.sleep(POLL_SEC)
    print(f'[{ts()}] timed out after {MAX_WAIT_HOURS}h waiting for autoinspect', flush=True)
    return False


def run_export() -> int:
    """Run batch_export_all.py via the same conda env we are in.
    Returns the subprocess exit code."""
    cmd = [sys.executable, str(ROOT / 'batch_export_all.py')]
    print(f'[{ts()}] starting export: {" ".join(cmd)}', flush=True)
    with open(EXPORT_LOG, 'w', encoding='utf-8') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
        rc = proc.wait()
    print(f'[{ts()}] export finished with exit code {rc}', flush=True)
    return rc


def write_report():
    lines = []
    lines.append(f'OVERNIGHT REPORT  ({ts()})')
    lines.append('=' * 60)

    # Autoinspect summary
    if AUTOINSPECT_LOG.exists():
        text = AUTOINSPECT_LOG.read_text(encoding='utf-8', errors='replace')
        lines.append('\n[1] AUTOINSPECT')
        for m in DATASET_DONE_RE.finditer(text):
            lines.append(f'   {m.group(0)}')
        # failures detail
        fails = re.findall(r'^\[[^\]]+\]\[\d+/\d+\] FAIL [^\n]+', text, flags=re.MULTILINE)
        if fails:
            lines.append('   Autoinspect failures:')
            for f in fails:
                lines.append(f'     {f}')
        else:
            lines.append('   No autoinspect failures.')
    else:
        lines.append('\n[1] AUTOINSPECT — log not found')

    # Export summary
    if EXPORT_LOG.exists():
        text = EXPORT_LOG.read_text(encoding='utf-8', errors='replace')
        lines.append('\n[2] EXPORT')
        for m in re.finditer(r'^\[(CC|RFC|FOF)\] DONE: ok=\d+, fail=\d+, neurons=\d+, size=[\d.]+ MB, time=[\d.]+s',
                             text, flags=re.MULTILINE):
            lines.append(f'   {m.group(0)}')
        # Final summary block
        m = re.search(r'={5,} SUMMARY ={5,}\n(.*?)\nALL DONE in [\d.]+ min',
                      text, flags=re.DOTALL)
        if m:
            lines.append('   final summary:')
            for ln in m.group(1).strip().split('\n'):
                lines.append(f'     {ln}')
        fails = re.findall(r'^\s*\[ERROR\] [^\n]+', text, flags=re.MULTILINE)
        if fails:
            lines.append('   Export ERROR lines:')
            for f in fails[:50]:
                lines.append(f'     {f.strip()}')
        else:
            lines.append('   No export errors.')
    else:
        lines.append('\n[2] EXPORT — log not found')

    MORNING_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[{ts()}] wrote {MORNING_REPORT}', flush=True)


def main():
    print(f'[{ts()}] chain runner starting; will poll {AUTOINSPECT_LOG} every {POLL_SEC}s',
          flush=True)
    if not wait_for_autoinspect():
        write_report()
        return 1
    rc = run_export()
    write_report()
    return rc


if __name__ == '__main__':
    sys.exit(main())
