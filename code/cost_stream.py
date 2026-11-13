import os, time, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from common import load_base
from config import RES, CACHE

B = load_base(); T = B['t']; Y = B['y']
caf_cols = list(B['caf_cols']); s_idx = [i for i, c in enumerate(caf_cols) if c.startswith('S_')]
Rz = np.load(os.path.join(CACHE, 'R_delay.npz'))
FS = {'TX': B['X'], 'TXSR': np.hstack([B['X'], B['caf'][:, s_idx], Rz['R_d1']])}

def fit(kind, Xtr, ytr, Xva, yva, seed):
    if kind == 'LGBM':
        P = dict(objective='binary', learning_rate=0.1, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, min_child_samples=20, verbose=-1,
                 num_threads=2, seed=seed, num_leaves=31, scale_pos_weight=3, metric='average_precision')
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(P, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        return lambda Z: m.predict(Z, num_iteration=m.best_iteration)
    if kind == 'XGB':
        m = xgb.XGBClassifier(tree_method='hist', n_estimators=600, learning_rate=0.1, subsample=0.8, colsample_bytree=0.7, n_jobs=2, random_state=seed,
                              eval_metric='aucpr', early_stopping_rounds=50, max_depth=8, scale_pos_weight=3)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return lambda Z: m.predict_proba(Z)[:, 1]
    m = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=seed, max_depth=20, min_samples_leaf=1, class_weight='balanced_subsample')
    m.fit(np.nan_to_num(Xtr, nan=0.0), ytr)
    return lambda Z: m.predict_proba(np.nan_to_num(Z, nan=0.0))[:, 1]

rows = []
test_rows = np.where(T >= 35)[0]
for kind, fsn in (('LGBM', 'TX'), ('LGBM', 'TXSR'), ('XGB', 'TXSR'), ('RF', 'TXSR')):
    X = FS[fsn]
    for r, win in ((34, None), (42, None), (48, None), (48, 10)):
        hi = r - 5; lo = 1 if win is None else max(1, hi - win + 1)
        tr = np.where((Y >= 0) & (T >= lo) & (T <= hi))[0]; va = np.where((Y >= 0) & (T > hi) & (T <= r))[0]
        tt, ti = [], []
        for rep in range(3):
            t0 = time.time(); pred = fit(kind, X[tr], Y[tr], X[va], Y[va], rep); tt.append(time.time() - t0)
            t0 = time.time(); pred(X[test_rows]); ti.append(time.time() - t0)
        rows.append(dict(model=kind, fs=fsn, horizon=r, window='all' if win is None else f'last{win}', n_train=len(tr), train_s_mean=np.mean(tt), train_s_std=np.std(tt, ddof=1),
                         infer_ms_per_1k=1000 * np.mean(ti) / len(test_rows) * 1000))
        print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, 'stream_cost.csv'), index=False)
