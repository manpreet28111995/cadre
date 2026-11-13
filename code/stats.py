import numpy as np, pandas as pd, os
from scipy.stats import wilcoxon
from config import RES
rng = np.random.default_rng(0)
P = pd.read_csv(os.path.join(RES, 'per_step_per_seed.csv'))
steps = sorted(P.step.unique())
mean_f1 = P.groupby(['config', 'policy', 'step'])['f1'].mean().unstack('step')
cnt = P.groupby(['config', 'policy', 'step'])[['tp', 'fp', 'fn']].mean()

def pooled(config, policy, idx):
    c = cnt.loc[(config, policy)].loc[steps]
    tp, fp, fn = c.tp.values[idx].sum(), c.fp.values[idx].sum(), c.fn.values[idx].sum()
    return 2 * tp / (2 * tp + fp + fn)

B = 5000
boot_idx = rng.integers(0, len(steps), size=(B, len(steps)))
def compare(a, b, label):
    (ca, pa), (cb, pb) = a, b
    d = mean_f1.loc[(ca, pa)].values - mean_f1.loc[(cb, pb)].values
    try: w = wilcoxon(d, zero_method='wilcox', alternative='two-sided'); p = w.pvalue
    except ValueError: p = 1.0
    diffs = np.array([pooled(ca, pa, ix) - pooled(cb, pb, ix) for ix in boot_idx])
    full = np.arange(len(steps))
    return dict(comparison=label, a=f'{ca}|{pa}', b=f'{cb}|{pb}', pooled_f1_a=pooled(ca, pa, full), pooled_f1_b=pooled(cb, pb, full),
                diff=pooled(ca, pa, full) - pooled(cb, pb, full), ci_low=np.percentile(diffs, 2.5), ci_high=np.percentile(diffs, 97.5),
                steps_a_better=int((d > 0).sum()), steps_b_better=int((d < 0).sum()), wilcoxon_p=p)
ref = ('LGBM__TX+S+R', 'psa')
rows = []
configs = sorted(set(P.config))
for c in configs:
    for pol in ('psa', 'fix'):
        if (c, pol) != ref: rows.append(compare(ref, (c, pol), 'proposed_vs_' + pol))
rows.append(compare(('LGBM__TX+S+R', 'fix'), ('LGBM__TX', 'fix'), 'features_R_fix'))
rows.append(compare(('LGBM__TX+S+R', 'psa'), ('LGBM__TX', 'psa'), 'features_R_psa'))
rows.append(compare(('LGBM__TX+S', 'fix'), ('LGBM__TX', 'fix'), 'features_S_fix'))
for c in configs: rows.append(compare((c, 'psa'), (c, 'fix'), 'policy_psa_vs_fix'))
D = pd.DataFrame(rows)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(run, 1)
    return adj
D['holm_p'] = np.nan
fam = D.comparison.map(lambda c: 'features' if c.startswith('features') else ('policy' if c.startswith('policy') else c))
for lab, g in D.groupby(fam):
    D.loc[g.index, 'holm_p'] = holm(g.wilcoxon_p.values)
D.to_csv(os.path.join(RES, 'significance.csv'), index=False)
print(D[['comparison', 'a', 'b', 'diff', 'ci_low', 'ci_high', 'steps_a_better', 'steps_b_better', 'wilcoxon_p', 'holm_p']].round(4).to_string())
