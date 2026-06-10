"""Baseline comparison (LR / RF / EBM / CaImAn rule) on the v9_iter8 test split."""
import sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, 'ml')
from data_utils import get_feature_cols, FBETA_BETA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, fbeta_score

beta = FBETA_BETA
df = pd.read_csv('ml/results/training_dataset_v9_corrected_iter7.csv')
fc = get_feature_cols(df)
sessions = df['session_name'].unique()
s2e = df.groupby('session_name')['experiment'].first().to_dict()
exps = [s2e[s] for s in sessions]
tr_idx, te_idx = next(StratifiedShuffleSplit(1, test_size=0.25, random_state=46).split(sessions, exps))
trS, teS = set(sessions[tr_idx]), set(sessions[te_idx])
trm, tem = df['session_name'].isin(trS), df['session_name'].isin(teS)

Xtr = df.loc[trm, fc].replace([np.inf, -np.inf], np.nan)
Xte = df.loc[tem, fc].replace([np.inf, -np.inf], np.nan)
ytr = df.loc[trm, 'ground_truth'].values
yte = df.loc[tem, 'ground_truth'].values
print(f'n_features={len(fc)}  train={len(ytr)}  test={len(yte)}  test KEEP%={yte.mean()*100:.1f}')


def best_thr(y, p):
    ts = np.linspace(0.05, 0.95, 181)
    return max(ts, key=lambda t: fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0))


def row(name, ytrue, p_tr, p_te):
    t = best_thr(ytr, p_tr)            # threshold picked on TRAIN (no test leakage)
    yp = (p_te >= t).astype(int)
    pr, rc, _, _ = precision_recall_fscore_support(ytrue, yp, average='binary', zero_division=0)
    fb = fbeta_score(ytrue, yp, beta=beta, zero_division=0)
    auc = roc_auc_score(ytrue, p_te)
    print(f'{name:<20} thr={t:.2f}  P={pr:.3f}  R={rc:.3f}  Fb={fb:.3f}  AUC={auc:.3f}')


imp = SimpleImputer(strategy='median').fit(Xtr)
Xtr_i, Xte_i = imp.transform(Xtr), imp.transform(Xte)
sc = StandardScaler().fit(Xtr_i)
Xtr_s, Xte_s = sc.transform(Xtr_i), sc.transform(Xte_i)

lr = LogisticRegression(max_iter=2000).fit(Xtr_s, ytr)
rf = RandomForestClassifier(n_estimators=300, random_state=46, n_jobs=-1).fit(Xtr_i, ytr)
ebm = pickle.load(open('ml/ebm_v9_iter8/model.pkl', 'rb'))
fn = list(ebm.feature_names_in_) if hasattr(ebm, 'feature_names_in_') else fc
print('EBM n features:', len(fn), '| match get_feature_cols:', set(fn) == set(fc))

print('--- TEST (threshold tuned on train) ---')
row('LogReg', yte, lr.predict_proba(Xtr_s)[:, 1], lr.predict_proba(Xte_s)[:, 1])
row('RandomForest', yte, rf.predict_proba(Xtr_i)[:, 1], rf.predict_proba(Xte_i)[:, 1])
row('EBM (ours)', yte, ebm.predict_proba(Xtr[fn])[:, 1], ebm.predict_proba(Xte[fn])[:, 1])

# --- CaImAn-metric baselines ---
g_all = df['ground_truth'].values
print('single-metric AUC: caiman_r_score=%.3f  caiman_snr=%.3f'
      % (roc_auc_score(g_all, df['caiman_r_score'].fillna(df['caiman_r_score'].median())),
         roc_auc_score(g_all, df['caiman_snr'].fillna(df['caiman_snr'].median()))))

rtr, str_ = df.loc[trm, 'caiman_r_score'].values, df.loc[trm, 'caiman_snr'].values
rte, ste = df.loc[tem, 'caiman_r_score'].values, df.loc[tem, 'caiman_snr'].values
best = (0.0, (-1, 0))
for rt in np.linspace(-0.2, 0.95, 40):
    for st in np.linspace(0, 8, 40):
        fb = fbeta_score(ytr, ((rtr >= rt) & (str_ >= st)).astype(int), beta=beta, zero_division=0)
        if fb > best[0]:
            best = (fb, (rt, st))
rt, st = best[1]
yp = ((rte >= rt) & (ste >= st)).astype(int)
pr, rc, _, _ = precision_recall_fscore_support(yte, yp, average='binary', zero_division=0)
fb = fbeta_score(yte, yp, beta=beta, zero_division=0)
print(f'{"CaImAn best rule":<20} r>={rt:.2f},snr>={st:.2f}  P={pr:.3f}  R={rc:.3f}  Fb={fb:.3f}  AUC=  n/a')

# trivial keep-all reference
yp = np.ones_like(yte)
pr, rc, _, _ = precision_recall_fscore_support(yte, yp, average='binary', zero_division=0)
print(f'{"keep-all":<20}            P={pr:.3f}  R={rc:.3f}  Fb={fbeta_score(yte, yp, beta=beta, zero_division=0):.3f}  AUC=  n/a')
