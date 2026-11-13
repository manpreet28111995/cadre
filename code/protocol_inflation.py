import os, sys, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from common import load_base, feature_sets, best_f1_threshold
from config import RES

B = load_base(); FS = feature_sets(B); y = B['y']; t = B['t']
lab = np.where(y >= 0)[0]
P = dict(objective='binary', learning_rate=0.1, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, min_child_samples=20, verbose=-1,
         num_threads=int(os.environ.get('NT', '2')), num_leaves=31, scale_pos_weight=3, metric='average_precision')

def f1_at(yy, pp, thr):
    pr = (pp >= thr).astype(int); tp = ((pr == 1) & (yy == 1)).sum(); fp = ((pr == 1) & (yy == 0)).sum(); fn = ((pr == 0) & (yy == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

def run(X, tr, va, te, seed):
    P2 = dict(P, seed=seed)
    d1 = lgb.Dataset(X[tr], y[tr]); d2 = lgb.Dataset(X[va], y[va], reference=d1)
    m = lgb.train(P2, d1, 600, valid_sets=[d2], callbacks=[lgb.early_stopping(50, verbose=False)])
    pv = m.predict(X[va], num_iteration=m.best_iteration); pt = m.predict(X[te], num_iteration=m.best_iteration)
    thr, _ = best_f1_threshold(y[va], pv)
    return dict(f1=f1_at(y[te], pt, thr), pr_auc=average_precision_score(y[te], pt), roc_auc=roc_auc_score(y[te], pt))

rows = []
for fsname in ('TX', 'TX+W'):
    X = FS[fsname]
    for seed in range(10):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(lab)
        n = len(perm); a, b = int(0.7 * n), int(0.85 * n)
        r = run(X, perm[:a], perm[a:b], perm[b:], seed); rows.append(dict(protocol='random', features=fsname, seed=seed, **r))
        tr = lab[t[lab] <= 29]; va = lab[(t[lab] > 29) & (t[lab] <= 34)]; te = lab[t[lab] > 34]
        r = run(X, tr, va, te, seed); rows.append(dict(protocol='chronological', features=fsname, seed=seed, **r))
        print(fsname, seed, flush=True)
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'protocol_inflation.csv'), index=False)
print(D.groupby(['protocol', 'features'])[['f1', 'pr_auc', 'roc_auc']].agg(['mean', 'std']).round(3))
