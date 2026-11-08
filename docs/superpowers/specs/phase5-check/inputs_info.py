"""Read-only look at the series_short files the page measured (component counts, shapes, types)."""
import sys, pickle, os
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import torch  # noqa: F401  (torch first: shm.dll)
import numpy as np
P = r"H:\Projects\_bearmind_test\series_short\5_Inspection\NOF_H01_1D\NOF_H01_1D_processed.pickle"
P4 = r"H:\Projects\_bearmind_test\series_short\4_Estimates\NOF_H01_1D_estimates.pickle"
for p in (P, P4):
    st = os.stat(p)
    with open(p, "rb") as f:
        e = pickle.load(f)
    print(p, "bytes", st.st_size)
    print("  A", e.A.shape, type(e.A).__name__, "C", e.C.shape, e.C.dtype, "YrA", None if e.YrA is None else e.YrA.shape,
          "S", type(e.S).__name__, getattr(e.S, "shape", None), "dims", e.dims)
    print("  idx_components", len(e.idx_components), "bad", 0 if e.idx_components_bad is None else len(e.idx_components_bad))
    print("  metrics_df", getattr(e, "metrics_df", None) is not None, "neurons_sn", None if getattr(e, "neurons_sn", None) is None else np.shape(e.neurons_sn))
    print("  SNR_comp", np.shape(getattr(e, "SNR_comp", None)), "r_values", np.shape(getattr(e, "r_values", None)), "imax", np.shape(getattr(e, "imax", None)))
    print("  has cnmf_dict", hasattr(e, "cnmf_dict"), "time", hasattr(e, "time"))
    if getattr(e, "metrics_df", None) is not None:
        df = e.metrics_df
        print("  metrics_df rows", len(df), "delete==1", int((df.get("delete") == 1).sum()) if "delete" in df else None)
    for k in ("autoinspection_config",):
        print("  ", k, getattr(e, k, None) if not isinstance(getattr(e, k, None), dict) else {kk: str(v)[:60] for kk, v in getattr(e, k).items()})
