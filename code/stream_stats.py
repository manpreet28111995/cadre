import numpy as np, pandas as pd, os
from scipy.stats import wilcoxon
from config import RES

rng = np.random.default_rng(0)
P = pd.read_csv(os.path.join(RES, 'stream_per_step_per_seed.csv'))
P = P[P.tag.fillna('') == '']
STEPS = np.arange(35, 50)
LATE = STEPS >= 43
B = 5000
key = ['model', 'fs', 'policy', 'd', 'cut', 'dec']
mean_cnt = P.groupby(key + ['step'])[['tp', 'fp', 'fn']].mean()
mean_f1 = P.groupby(key + ['step'])['f1'].mean().unstack('step')
BI_ALL = rng.integers(0, len(STEPS), size=(B, len(STEPS)))
BI_LATE = rng.integers(0, int(LATE.sum()), size=(B, int(LATE.sum())))

def counts(k):
    c = mean_cnt.loc[k].loc[STEPS]
    return c.tp.values, c.fp.values, c.fn.values

def pooled(c, idx):
    tp, fp, fn = c[0][idx].sum(), c[1][idx].sum(), c[2][idx].sum()
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0

def compare(fam, label, ka, kb, d):
    ca, cb = counts(ka), counts(kb)
    diff_steps = mean_f1.loc[ka].loc[STEPS].values - mean_f1.loc[kb].loc[STEPS].values
    try: p = wilcoxon(diff_steps, zero_method='wilcox', alternative='two-sided').pvalue
    except ValueError: p = 1.0
    full = np.arange(len(STEPS)); late = np.where(LATE)[0]
    da = np.array([pooled(ca, ix) - pooled(cb, ix) for ix in BI_ALL])
    cl = tuple(x[late] for x in ca); cm = tuple(x[late] for x in cb)
    dl = np.array([pooled(cl, ix) - pooled(cm, ix) for ix in BI_LATE])
    return dict(family=fam, comparison=label, d=d, dec=ka[5], a='|'.join(map(str, ka)), b='|'.join(map(str, kb)),
                f1_a=pooled(ca, full), f1_b=pooled(cb, full), diff_all=pooled(ca, full) - pooled(cb, full),
                lo_all=np.percentile(da, 2.5), hi_all=np.percentile(da, 97.5),
                late_a=pooled(cl, np.arange(len(late))), late_b=pooled(cm, np.arange(len(late))),
                diff_late=pooled(cl, np.arange(len(late))) - pooled(cm, np.arange(len(late))),
                lo_late=np.percentile(dl, 2.5), hi_late=np.percentile(dl, 97.5),
                steps_a_better=int((diff_steps > 0).sum()), steps_b_better=int((diff_steps < 0).sum()), wilcoxon_p=p)

def K(model, fs, pol, d, dec, cut=34): return (model, fs, pol, d, cut, dec)
avail = set(mean_f1.index)
rows = []
def add(fam, label, ka, kb, d):
    if ka in avail and kb in avail: rows.append(compare(fam, label, ka, kb, d))

for d in (1, 2, 3, 5):
    for dec in ('fix', 'psa'):
        for pol in ('every', 'slide', 'per3', 'per5', 'psi', 'dpm', 'oracle'):
            add('policy_vs_static_' + dec, pol + '_vs_static', K('LGBM', 'TXSR', pol, d, dec), K('LGBM', 'TXSR', 'static', d, dec), d)
        add('dpm_vs_every_' + dec, 'dpm_vs_every', K('LGBM', 'TXSR', 'dpm', d, dec), K('LGBM', 'TXSR', 'every', d, dec), d)
        add('oracle_vs_dpm_' + dec, 'oracle_vs_dpm', K('LGBM', 'TXSR', 'oracle', d, dec), K('LGBM', 'TXSR', 'dpm', d, dec), d)
    for pol in ('static', 'every', 'slide', 'per3', 'per5', 'psi', 'dpm', 'oracle'):
        add('psa_vs_fix', pol + '_psa_vs_fix', K('LGBM', 'TXSR', pol, d, 'psa'), K('LGBM', 'TXSR', pol, d, 'fix'), d)
    for pol in ('static', 'every', 'dpm'):
        for dec in ('fix', 'psa'):
            add('txsr_vs_tx_' + dec, pol + '_txsr_vs_tx', K('LGBM', 'TXSR', pol, d, dec), K('LGBM', 'TX', pol, d, dec), d)
for pol in ('static', 'every', 'dpm'):
    for dec in ('fix', 'psa'):
        add('ctx_effect_' + dec, pol + '_txsr_vs_txs', K('LGBM', 'TXSR', pol, 1, dec), K('LGBM', 'TXS', pol, 1, dec), 1)
        add('ctx_effect_' + dec, pol + '_txr_vs_tx', K('LGBM', 'TXR', pol, 1, dec), K('LGBM', 'TX', pol, 1, dec), 1)
        add('ctx_effect_' + dec, pol + '_txs_vs_tx', K('LGBM', 'TXS', pol, 1, dec), K('LGBM', 'TX', pol, 1, dec), 1)
for d in (1,):
    for m in ('XGB', 'RF'):
        for dec in ('fix',):
            add('policy_vs_static_' + m, 'every_vs_static_' + m, K(m, 'TXSR', 'every', d, dec), K(m, 'TXSR', 'static', d, dec), d)
            add('policy_vs_static_' + m, 'dpm_vs_static_' + m, K(m, 'TXSR', 'dpm', d, dec), K(m, 'TXSR', 'static', d, dec), d)

D = pd.DataFrame(rows)

def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj

D['holm_p'] = np.nan
for _, g in D.groupby('family'):
    D.loc[g.index, 'holm_p'] = holm(g.wilcoxon_p.values)
D.to_csv(os.path.join(RES, 'stream_significance.csv'), index=False)
pd.set_option('display.width', 250)
print(D[['family', 'comparison', 'd', 'diff_all', 'lo_all', 'hi_all', 'diff_late', 'lo_late', 'hi_late', 'steps_a_better', 'steps_b_better', 'wilcoxon_p', 'holm_p']].round(3).to_string())
