import scipy.stats

from bm_examinator import LoadEstimates
from auto_inspector import *

fname = "D://Projects//estim_data//NOF_H01_3D_CR_MC_4_1_0.85_estimates.pickle"
fps = 20

est = LoadEstimates(fname, default_fps=fps)
print('loaded estimates')

df, corr, dist_c, dist_b = estimates_to_metrics(est, fps)
print(df)
print(corr.shape)
print(dist_c.shape)
print(dist_b.shape)

d1 = dist_b[np.nonzero(dist_b)]
d2 = dist_c[np.nonzero(dist_b)]

print(d1[:10])
print(d2[:10])
print(scipy.stats.pearsonr(d1[~np.isnan(d1)], d2[~np.isnan(d1)]))

#ddf = metrics_to_decision(df, circ_thr=4, maxedge_thr=100)
#print(ddf)

ddf2 = metrics_to_dummy_decision(df)
print(ddf2)
