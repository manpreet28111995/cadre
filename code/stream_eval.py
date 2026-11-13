import os, glob, json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from config import RES

D = os.path.join(RES, 'stream')
rows, step_rows, trace_rows = [], [], []
for f in sorted(glob.glob(os.path.join(D, '*.npz'))):
    name = os.path.basename(f)[:-4]
    meta = json.load(open(f[:-4] + '.json'))
    if name.endswith('_test'): continue
    z = np.load(f); y = z['y']; st = z['te_step']; cut = meta['cut']
    n_seed = z['p'].shape[0]
    for s in range(n_seed):
        p = z['p'][s]
        pr_auc = average_precision_score(y, p); roc = roc_auc_score(y, p)
        late = st >= 43; early = ~late
        pr_late = average_precision_score(y[late], p[late]) if y[late].sum() > 0 else np.nan
        roc_late = roc_auc_score(y[late], p[late]) if 0 < y[late].sum() < late.sum() else np.nan
        roc_early = roc_auc_score(y[early], p[early]) if 0 < y[early].sum() < early.sum() else np.nan
        pr_early = average_precision_score(y[early], p[early]) if y[early].sum() > 0 else np.nan
        for dec in ('fix', 'psa'):
            fl = z[dec][s].astype(int)
            def cnt(m):
                tp = int(((fl[m] == 1) & (y[m] == 1)).sum()); fp = int(((fl[m] == 1) & (y[m] == 0)).sum()); fn = int(((fl[m] == 0) & (y[m] == 1)).sum()); tn = int(((fl[m] == 0) & (y[m] == 0)).sum())
                return tp, fp, fn, tn
            def f1(c): return 2 * c[0] / (2 * c[0] + c[1] + c[2]) if (2 * c[0] + c[1] + c[2]) > 0 else 0.0
            def mcc(c):
                dd = np.sqrt(float(c[0] + c[1]) * (c[0] + c[2]) * (c[3] + c[1]) * (c[3] + c[2])); return (c[0] * c[3] - c[1] * c[2]) / dd if dd > 0 else 0.0
            allm = np.ones(len(y), bool); ca = cnt(allm); ce = cnt(early); cl = cnt(late); cr = cnt(st >= 47)
            rows.append(dict(config=name, model=meta['model'], fs=meta['fs'], policy=meta['policy'], d=meta['d'], cut=cut, seed=int(z['seeds'][s]), dec=dec,
                             f1=f1(ca), precision=ca[0] / max(ca[0] + ca[1], 1), recall=ca[0] / max(ca[0] + ca[2], 1), mcc=mcc(ca), f1_early=f1(ce), f1_late=f1(cl),
                             recall_late=cl[0] / max(cl[0] + cl[2], 1), precision_late=cl[0] / max(cl[0] + cl[1], 1), f1_rec=f1(cr),
                             pr_auc=pr_auc, roc_auc=roc, pr_auc_early=pr_early, pr_auc_late=pr_late, roc_auc_early=roc_early, roc_auc_late=roc_late,
                             retrains=int(z['n_train'][s]), train_s=float(z['train_s'][s]), infer_s=float(z['infer_s'][s])))
            for t in sorted(set(st)):
                c = cnt(st == t)
                step_rows.append(dict(config=name, policy=meta['policy'], model=meta['model'], fs=meta['fs'], d=meta['d'], cut=cut, seed=int(z['seeds'][s]), dec=dec, step=int(t), tp=c[0], fp=c[1], fn=c[2], tn=c[3], f1=f1(c)))
    for t in range(cut + 1, 50):
        trace_rows.append(dict(config=name, policy=meta['policy'], d=meta['d'], step=t, retrain=float(z['retrain'][:, t].mean()), alarm=float(z['alarm'][:, t].mean()),
                               pi_hat=float(z['pi_hat'][:, t].mean()), psi=float(np.nanmean(z['psi'][:, t])) if not np.all(np.isnan(z['psi'][:, t])) else np.nan,
                               th_psi=float(np.nanmean(z['th_psi'][:, t])) if not np.all(np.isnan(z['th_psi'][:, t])) else np.nan,
                               c2=float(np.nanmean(z['c2'][:, t])) if not np.all(np.isnan(z['c2'][:, t])) else np.nan,
                               th_c2=float(np.nanmean(z['th_c2'][:, t])) if not np.all(np.isnan(z['th_c2'][:, t])) else np.nan,
                               dpm=float(np.nanmean(z['dpm'][:, t])) if not np.all(np.isnan(z['dpm'][:, t])) else np.nan,
                               dpm_ref=float(np.nanmean(z['dpm_ref'][:, t])) if not np.all(np.isnan(z['dpm_ref'][:, t])) else np.nan,
                               horizon=float(z['horizon'][:, t].mean()), tau=float(z['tau'][:, t].mean())))
R = pd.DataFrame(rows); S = pd.DataFrame(step_rows); Tr = pd.DataFrame(trace_rows)
for Z in (R, S, Tr): Z['tag'] = Z.config.str.extract(r'_c\d+(.*)$')[0].fillna('')
R.to_csv(os.path.join(RES, 'stream_metrics_per_seed.csv'), index=False)
S.to_csv(os.path.join(RES, 'stream_per_step_per_seed.csv'), index=False)
Tr.to_csv(os.path.join(RES, 'stream_traces.csv'), index=False)
keys = ['config', 'model', 'fs', 'policy', 'd', 'cut', 'dec', 'tag']
num = [c for c in R.columns if c not in keys + ['seed']]
A = R.groupby(keys)[num].agg(['mean', 'std'])
A.columns = ['_'.join(c) for c in A.columns]
A = A.reset_index(); A['n_seeds'] = R.groupby(keys)['seed'].nunique().values
A.to_csv(os.path.join(RES, 'stream_summary.csv'), index=False)
print(A[A.cut == 34][['config', 'dec', 'f1_mean', 'f1_std', 'f1_early_mean', 'f1_late_mean', 'f1_rec_mean', 'retrains_mean', 'n_seeds']].round(3).to_string())
