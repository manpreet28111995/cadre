import os, numpy as np, pandas as pd, lightgbm as lgb
from common import load_base
from config import RES, CACHE

B = load_base(); T = B['t']; Y = B['y']
caf_cols = list(B['caf_cols']); s_idx = [i for i, c in enumerate(caf_cols) if c.startswith('S_')]
Rz = np.load(os.path.join(CACHE, 'R_delay.npz'), allow_pickle=True); rcols = [str(c) for c in Rz['R_cols']]
names = [str(c) for c in B['cols']] + [caf_cols[i] for i in s_idx] + rcols
group = ['T'] * len(B['cols']) + ['S'] * len(s_idx) + ['R'] * len(rcols)
X = np.hstack([B['X'], B['caf'][:, s_idx], Rz['R_d1']])
P = dict(objective='binary', learning_rate=0.1, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, min_child_samples=20, verbose=-1,
         num_threads=2, seed=0, num_leaves=31, scale_pos_weight=3, metric='average_precision')

def fit(r):
    hi = r - 5
    tr = np.where((Y >= 0) & (T <= hi))[0]; va = np.where((Y >= 0) & (T > hi) & (T <= r))[0]
    d1 = lgb.Dataset(X[tr], Y[tr]); d2 = lgb.Dataset(X[va], Y[va], reference=d1)
    return lgb.train(P, d1, 600, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False)])

def contrib(m, rows):
    c = m.predict(X[rows], num_iteration=m.best_iteration, pred_contrib=True)
    return np.abs(c[:, :-1]).mean(0), c[:, :-1]

out = pd.DataFrame(dict(feature=names, group=group))
m0 = fit(34)
for nm, lo, hi in (('static_early', 35, 42), ('static_late', 43, 49), ('static_all', 35, 49)):
    rows = np.where((Y >= 0) & (T >= lo) & (T <= hi))[0]
    out[nm], _ = contrib(m0, rows)
m1 = fit(48)
rows = np.where((Y >= 0) & (T >= 45) & (T <= 49))[0]
out['retrained_late'], _ = contrib(m1, rows)
out['static_late_ill'] = 0.0
rows_i = np.where((Y == 1) & (T >= 43) & (T <= 49))[0]
out['static_late_ill'], _ = contrib(m0, rows_i)
rows_e = np.where((Y == 1) & (T >= 35) & (T <= 42))[0]
out['static_early_ill'], _ = contrib(m0, rows_e)
out.to_csv(os.path.join(RES, 'shap_importance.csv'), index=False)
g = out.groupby('group')[['static_early', 'static_late', 'static_all', 'retrained_late']].sum()
g.to_csv(os.path.join(RES, 'shap_groups.csv'))
print(g.round(3)); print(out.sort_values('static_all', ascending=False).head(12)[['feature', 'group', 'static_all', 'static_late', 'retrained_late']].round(3))
