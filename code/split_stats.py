import numpy as np, pandas as pd, os
from common import load_base, splits
from config import RES
B = load_base(); y, t = B['y'], B['t']; tr, va, te = splits(B)
rows = []
for nm, idx, rng in (('train', tr, '1-29'), ('validation', va, '30-34'), ('test', te, '35-49')):
    rows.append(dict(split=nm, steps=rng, labelled=len(idx), illicit=int((y[idx] == 1).sum()), prevalence=float((y[idx] == 1).mean())))
for nm, mk, rng in (('test early', t[te] <= 42, '35-42'), ('test late', t[te] >= 43, '43-49')):
    idx = te[mk]; rows.append(dict(split=nm, steps=rng, labelled=len(idx), illicit=int((y[idx] == 1).sum()), prevalence=float((y[idx] == 1).mean())))
D = pd.DataFrame(rows); D.to_csv(os.path.join(RES, 'split_stats.csv'), index=False); print(D.to_string())
