import scipy.stats

from bm_examinator import LoadEstimates
from auto_inspector import *

pd.options.display.max_rows = None
pd.options.display.max_columns = None


fname = "D://Projects//estim_data//NOF_H01_3D_CR_MC_4_1_0.85_estimates.pickle"
fps = 20

est = LoadEstimates(fname, default_fps=fps)
print('loaded estimates')

df, corr, dist_c, dist_b = estimates_to_metrics(est, fps, match_threshold=1,
                                                include_heavy=False)
# print(df.sort_values('corr_groups'))
# print(corr)
# print(dist_c)
# print(dist_b)
#
# d1 = dist_b[np.nonzero(dist_b)]
# d2 = dist_c[np.nonzero(dist_b)]
#
# print(d1[:10])
# print(d2[:10])
# print(scipy.stats.pearsonr(d1[~np.isnan(d1)], d2[~np.isnan(d1)]))
df.sort_values('corr_groups', ascending=False, inplace=True)
df = metrics_to_decision(df[:100], corr, dist_c, dist_b)
print(df)

# ddf2 = metrics_to_dummy_decision(df)
# print(ddf2)

# m2d = metrics_to_decision(df)
# print(m2d)