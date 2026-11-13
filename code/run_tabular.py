import sys, time, json, itertools, numpy as np
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import *
from config import SEEDS

model, fsname = sys.argv[1], sys.argv[2]
B = load_base(); FS = feature_sets(B); X = FS[fsname]; y = B['y']
tr, va, te = splits(B)
Xtr, ytr, Xva, yva, Xte = X[tr], y[tr], X[va], y[va], X[te]

def lgb_fit(params, seed):
    p = dict(objective='binary', learning_rate=0.05, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, min_child_samples=20,
             verbose=-1, num_threads=2, seed=seed, **params)
    dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
    m = lgb.train({**p, 'metric': 'average_precision'}, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
    return m, (lambda A: m.predict(A, num_iteration=m.best_iteration)), m.best_iteration

def xgb_fit(params, seed):
    m = xgb.XGBClassifier(tree_method='hist', n_estimators=1500, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, n_jobs=2,
                          random_state=seed, eval_metric='aucpr', early_stopping_rounds=60, **params)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, (lambda A: m.predict_proba(A)[:, 1]), m.best_iteration

def rf_fit(params, seed):
    m = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=seed, **params).fit(Xtr, ytr)
    return m, (lambda A: m.predict_proba(A)[:, 1]), 300

sc = StandardScaler().fit(Xtr)
def prep(A):
    Z = np.clip(sc.transform(A), -10, 10)
    miss = np.isnan(Z).any(1, keepdims=True).astype(np.float32)
    return np.hstack([np.nan_to_num(Z, nan=0.0), miss])

def lr_fit(params, seed):
    m = LogisticRegression(max_iter=3000, **params).fit(prep(Xtr), ytr)
    return m, (lambda A: m.predict_proba(prep(A))[:, 1]), 0

grids = {
    'LGBM': (lgb_fit, [dict(num_leaves=a, scale_pos_weight=b) for a, b in itertools.product([15, 31, 63], [1, 3])]),
    'XGB': (xgb_fit, [dict(max_depth=a, scale_pos_weight=b) for a, b in itertools.product([4, 6, 8], [1, 3])]),
    'RF': (rf_fit, [dict(max_depth=a, min_samples_leaf=b, class_weight=c) for a, b, c in itertools.product([None, 20], [1, 3], [None, 'balanced_subsample'])]),
    'LR': (lr_fit, [dict(C=c) for c in (0.01, 0.1, 1.0)]),
}
fit, grid = grids[model]
best = None
for g in grid:
    m, pred, it = fit(g, 0)
    s = ap(yva, pred(Xva))
    print(model, fsname, g, 'val AP %.4f' % s, flush=True)
    if best is None or s > best[0]: best = (s, g)
hp = best[1]
pv, pt, ttr, tin, its, nparams = [], [], [], [], [], []
for s in SEEDS:
    t0 = time.time(); m, pred, it = fit(hp, s); ttr.append(time.time() - t0)
    t0 = time.time(); a = pred(Xva); b = pred(Xte); tin.append(time.time() - t0)
    pv.append(a); pt.append(b); its.append(it)
    print(model, fsname, 'seed', s, 'val AP %.4f' % ap(yva, a), 'test AP %.4f' % ap(y[te], b), flush=True)
save_preds(f'{model}__{fsname}', pv, pt, ttr, tin, dict(model=model, features=fsname, n_features=int(X.shape[1]), hyperparams=hp, val_ap_seed0=best[0], iterations=its))
