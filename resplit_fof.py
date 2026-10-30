"""Re-split FOF stitched npz (cutoff=15 re-export) into per-day 1D/2D/3D.

Mirrors iteration-1 logic (see memory: FOF pickle = 3 stitched sessions in
order 1D,2D,3D; split by per-mouse Mini_TS row counts; sum must equal ca_T;
F36 had +1 mismatch -> trim last segment; F48 3D truncated -> Mini_TS_3D
already short, sum still == ca_T).

Reads stitched npz from output/FOF/inspection_artifacts_*/, splits using
per-mouse Mini_TS lengths in W:\\DATASET\\FOF\\Synchro\\act_ts, writes per-day
npz to traces/ and per-day metadata json (copied from the mouse's 1D json with
session_name patched). Backs up the stitched source to _originals_pre_split.
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BM_OUT = Path(r'C:\Users\1\pycharmprojects\bearmind\output\FOF')
SYN = Path(r'W:\DATASET\FOF\Synchro')
TRACES = SYN / 'traces'
META = SYN / 'metadata'
ACT_TS = SYN / 'act_ts'
BACKUP = Path(r'W:\DATASET\FOF\_originals_pre_split')

DAYS = ['1D', '2D', '3D']


def main():
    BACKUP.mkdir(parents=True, exist_ok=True)
    # Clear stale per-day traces/metadata; keep features/act_ts/feat_ts
    for d in (TRACES, META):
        for f in d.glob('*'):
            f.unlink()
    syn_done = SYN / 'synchronized'
    if syn_done.exists():
        shutil.rmtree(syn_done)

    stitched = sorted(BM_OUT.glob('inspection_artifacts_*/FOF_*_1D_data.npz'))
    print(f'stitched npz: {len(stitched)}')

    stats = {'ok': 0, 'trim': 0, 'fail': 0}
    for npz_path in stitched:
        mouse = npz_path.stem.replace('_1D_data', '')  # FOF_F01

        # per-day Mini_TS lengths
        seg_lens = {}
        miss = []
        for day in DAYS:
            ts = ACT_TS / f'{mouse}_{day}_Mini_TS.csv'
            if not ts.exists():
                miss.append(day)
            else:
                seg_lens[day] = pd.read_csv(ts, header=None).shape[0]
        if miss:
            print(f'[FAIL] {mouse}: missing Mini_TS for {miss}')
            stats['fail'] += 1
            continue

        # backup stitched source (overwrite — reflects current cutoff=15 export)
        bkp = BACKUP / f'{mouse}_1D_data.npz.orig'
        shutil.copy2(npz_path, bkp)

        with np.load(npz_path, allow_pickle=True) as d:
            arrays = {k: d[k] for k in d.keys()}
        ca_T = arrays['C'].shape[1]
        total = sum(seg_lens.values())
        note = ''
        if total != ca_T:
            # iteration-1 rule: keep 1D & 2D, clip 3D to remainder (handles F36 +1)
            seg_lens[DAYS[-1]] = ca_T - seg_lens[DAYS[0]] - seg_lens[DAYS[1]]
            note = f' (TRIM 3D: sum {total} != ca_T {ca_T})'
            stats['trim'] += 1
            if seg_lens[DAYS[-1]] < 0:
                print(f'[FAIL] {mouse}: 1D+2D > ca_T, cannot split')
                stats['fail'] += 1
                continue

        # source metadata for the mouse (named _1D by export)
        src_meta = None
        for art in BM_OUT.glob(f'inspection_artifacts_{mouse}_1D/{mouse}_1D_metadata.json'):
            src_meta = art
            break

        cursor = 0
        for day in DAYS:
            n = seg_lens[day]
            s, e = cursor, cursor + n
            sliced = {}
            for k, v in arrays.items():
                if v.ndim == 2 and v.shape[1] == ca_T:
                    sliced[k] = v[:, s:e]
                elif v.ndim == 1 and v.shape[0] == ca_T:
                    sliced[k] = v[s:e]
                else:
                    sliced[k] = v
            np.savez_compressed(TRACES / f'{mouse}_{day}_data.npz', **sliced)

            if src_meta and src_meta.exists():
                with open(src_meta) as f:
                    md = json.load(f)
                md['session_name'] = f'{mouse}_{day}'
                with open(META / f'{mouse}_{day}_metadata.json', 'w') as f:
                    json.dump(md, f, indent=2)
            cursor = e

        print(f'[OK] {mouse}: 1D={seg_lens["1D"]} 2D={seg_lens["2D"]} '
              f'3D={seg_lens["3D"]} sum={cursor} ca_T={ca_T}{note}')
        stats['ok'] += 1

    print(f'\nsummary: {stats}')
    print(f'traces now: {len(list(TRACES.glob("*_data.npz")))}')
    print(f'metadata now: {len(list(META.glob("*_metadata.json")))}')


if __name__ == '__main__':
    main()
