import numpy as np, pandas as pd, os
from sklearn.metrics import roc_auc_score, average_precision_score
from common import load_base, splits
from config import RES
B = load_base(); y, t = B['y'], B['t']; tr, va, te = splits(B)
yt, tt = y[te], t[te]
rows = []
for cfg in ('LGBM__TX', 'LGBM__TX+S+R', 'RF__TX', 'XGB__TX', 'SAGE__TX', 'GCN__TX', 'MLP__TX', 'GAT__TX'):
    f = os.path.join(RES, 'preds', cfg + '.npz')
    if not os.path.exists(f): continue
    PT = np.load(f)['p_test']
    for nm, mk in (('early', tt <= 42), ('late', tt >= 43)):
        auc = [roc_auc_score(yt[mk], p[mk]) for p in PT]; ap = [average_precision_score(yt[mk], p[mk]) for p in PT]
        pos = [np.median(p[mk & (yt == 1)]) for p in PT]; neg = [np.median(p[mk & (yt == 0)]) for p in PT]
        rows.append(dict(config=cfg, period=nm, roc_auc=np.mean(auc), pr_auc=np.mean(ap), median_score_illicit=np.mean(pos), median_score_licit=np.mean(neg), base_rate=float(yt[mk].mean())))
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'late_period_ranking.csv'), index=False); print(D.round(3).to_string())
