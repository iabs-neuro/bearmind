"""ssub_B decision and map workers for the hour session LNOF_J01_1D with the box from its crop record.

usage: lnof_decisions.py   (reads series_lnof's registry and configs; writes nothing)
"""
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import load_record
from bearmind.stages.cnmf import background_downsample, cnmf_need_gib, mem_budget_gb
from bearmind.stages.thresholds import map_peak_gb, map_workers

P = SeriesPaths(Path(r"H:\Projects\_bearmind_test\series_lnof"))
SID = "LNOF_J01_1D"


def main():
    rec = load_record(P, SID)
    cfg = config.resolve(P, SID)
    m = rec.stages["crop"].metrics
    _, _, h, w = m["box"]
    T = 107107
    print(f"n_frames in record {rec.n_frames}; box {h}x{w}; fov {m['fov']}; method {m['fov_method']}")
    for name, (hh, ww) in (("new box", (h, w)), ("plan new 472x436", (472, 436)), ("old auto 567x520", (567, 520))):
        d = background_downsample(T, hh, ww, cfg)
        needs = {s: round(cnmf_need_gib(T, hh, ww, s, cfg), 1) for s in (2, 3, 4)}
        print(f"{name}: ssub_B {d.ssub_B} (configured {d.configured}, fits {d.fits}); need GiB by ssub_B {needs}; "
              f"budget {d.budget_gib}")
        gs = [int(g) for g in cfg["thresholds"]["gsig_range"]]
        tsub = cfg["cnmf"]["tsub"]
        peak = max(map_peak_gb(T, hh, ww, g, tsub, mem_budget_gb(cfg)) for g in gs)
        print(f"   map_workers {map_workers(T, hh, ww, gs, tsub, cfg)} (peak of the largest gSig {peak:.1f} GB, "
              f"gsig_range {gs}, tsub {tsub}, n_cpus {cfg['machine']['n_cpus']})")


if __name__ == "__main__":
    main()
