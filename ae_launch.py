from bm_examinator import LoadEstimates
from auto_inspector import *

fname = "D://Projects//estim_data//NOF_H01_3D_CR_MC_4_1_0.85_estimates.pickle"
fps = 20

est = LoadEstimates(fname, default_fps=fps)
print('loaded estimates')
df = estimates_to_metrics(est, fps)
print(df)
