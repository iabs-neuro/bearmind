"""Why the LNOF_J01_1D field (436x415) differs from the plan's 452x416: the committed rule against variants of
the details the plan text leaves open (rim at the frame edge, 99th-percentile clip, tsub of the seed maps).
Reads 3000 frames of series_lnof (read-only); writes only into the scratchpad.
"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np
from scipy import ndimage

import bm_qc
from bearmind import crop
from bearmind.paths import SeriesPaths
from bearmind.registry import load_record
from bearmind.video import SessionReader

P = SeriesPaths(Path(r"H:\Projects\_bearmind_test\series_lnof"))
SID = "LNOF_J01_1D"


def field(e, lit_box, protect=None):
    return crop.activity_fov(e, lit_box, 0.005, 20, protect=protect)


def emap(maps, inner, clip=True):
    excess = (maps["cn"] * maps["pnr"] - maps["cn_neg"] * maps["pnr_neg"]).astype(np.float32)
    e = ndimage.gaussian_filter(excess, crop.ACTIVITY_SIGMA_PX)
    if inner is not None:
        e[~inner] = 0
    pos = e[e > 0]
    if clip and pos.size:
        np.minimum(e, np.percentile(pos, 99), out=e)
    return e


def main():
    rec = load_record(P, SID)
    frames = SessionReader(P.root, rec, crop=None).windows(3, 1000)
    mean = frames.mean(axis=0, dtype=np.float64).astype(np.float32)
    lit_box, mask = crop.lit_fov(mean, 0.05, 0.2)
    inner_frame = crop.lit_inner(mask)  # committed: the frame edge counts as an edge
    inner_lens = ndimage.distance_transform_edt(mask) > crop.RIM_PX  # frame edge not an edge
    inner_sq = ndimage.binary_erosion(mask, iterations=crop.RIM_PX, border_value=1)
    print("lit box", lit_box)
    out = np.load if False else None
    for tsub in (1, 2):
        maps = bm_qc.compute_init_maps(frames, gSig=4, tsub=tsub, mirror=True, max_mem_gb=16, verbose=False)
        seeds = crop.strong_seeds(maps, inner_frame, 4)
        for name, inner, clip in (("committed rim (frame edge counts), clip", inner_frame, True),
                                  ("rim at the lens edge only, clip", inner_lens, True),
                                  ("square erosion 10 it, frame edge not an edge, clip", inner_sq, True),
                                  ("no rim, clip", None, True),
                                  ("committed rim, no clip", inner_frame, False),
                                  ("rim at the lens edge only, no clip", inner_lens, False)):
            e = emap(maps, inner, clip)
            f0 = field(e, lit_box)
            f1 = field(e, lit_box, protect=seeds)
            print(f"tsub {tsub} | {name}: field {f0} -> with strong seeds {f1}; pos/neg {crop.pos_neg_ratio(e):.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
