import numpy as np, pandas as pd, glob, os, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
from common import load_base, splits, best_f1_threshold
from config import RES, VAL_END, LAST_STEP

B = load_base(); y, t = B['y'], B['t']
tr, va, te = splits(B)
yv, yt, tv, tt = y[va], y[te], t[va], t[te]
EPS = 1e-6
def logit(p): p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))
def sig(z): return 1 / (1 + np.exp(-z))

def platt(pv):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv).reshape(-1, 1), yv)
    a, b = lr.coef_[0, 0], lr.intercept_[0]
    return lambda p: sig(a * logit(p) + b)

def em_prior(p, pi_s, iters=200):
    pi = pi_s
    for _ in range(iters):
        w1 = pi / pi_s * p; w0 = (1 - pi) / (1 - pi_s) * (1 - p)
        new = np.mean(w1 / (w1 + w0))
        if abs(new - pi) < 1e-7: pi = new; break
        pi = new
    return float(np.clip(pi, 1e-3, 0.6))

def counts(pred, yy):
    tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum()); fn = int(((pred == 0) & (yy == 1)).sum()); tn = int(((pred == 0) & (yy == 0)).sum())
    return tp, fp, fn, tn

def f1_of(tp, fp, fn): return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
def mcc_of(tp, fp, fn, tn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return (tp * tn - fp * fn) / d if d > 0 else 0.0

steps = list(range(VAL_END + 1, LAST_STEP + 1))
rows, step_rows, prior_rows = [], [], []
for f in sorted(glob.glob(os.path.join(RES, 'preds', '*.npz'))):
    name = os.path.basename(f)[:-4]; model, fs = name.split('__')
    z = np.load(f); PV, PT = z['p_val'], z['p_test']
    for s in range(PV.shape[0]):
        pv, pt = PV[s], PT[s]
        cal = platt(pv); cv, ct = cal(pv), cal(pt)
        pi_s = yv.mean()
        thr_raw, _ = best_f1_threshold(yv, pv)
        tau_v, _ = best_f1_threshold(yv, cv)
        base_logit = logit(tau_v)
        preds = {'fix': (pt >= thr_raw).astype(int)}
        est = np.zeros(len(te)); tru = np.zeros(len(te)); thr_oracle_step = np.zeros(len(te))
        allp = np.r_[cv, ct]; allt = np.r_[tv, tt]; ally = np.r_[yv, yt]
        pred_psa = np.zeros(len(te), int); pred_true = np.zeros(len(te), int); pred_orc = np.zeros(len(te), int)
        for st in steps:
            wmask = (allt >= st - 2) & (allt <= st)
            pi_hat = em_prior(allp[wmask], pi_s); pi_true = float(np.clip(ally[wmask].mean(), 1e-3, 0.6))
            m = tt == st
            pred_psa[m] = (logit(ct[m]) + logit(pi_hat) - logit(pi_s) >= base_logit).astype(int)
            pred_true[m] = (logit(ct[m]) + logit(pi_true) - logit(pi_s) >= base_logit).astype(int)
            th, _ = best_f1_threshold(yt[m], pt[m]); pred_orc[m] = (pt[m] >= th).astype(int)
            if s == 0: prior_rows.append(dict(config=name, step=st, est_prior=pi_hat, window_true_prior=pi_true, step_true_prior=float(yt[m].mean())))
        preds['psa'] = pred_psa; preds['psa_true'] = pred_true; preds['oracle_step'] = pred_orc
        th_g, _ = best_f1_threshold(yt, pt); preds['oracle_global'] = (pt >= th_g).astype(int)
        ap_all = average_precision_score(yt, pt); auc_all = roc_auc_score(yt, pt)
        late = tt >= 43; early = ~late
        ap_late = average_precision_score(yt[late], pt[late])
        for pol, pr in preds.items():
            tp, fp, fn, tn = counts(pr, yt)
            r = dict(config=name, model=model, features=fs, seed=s, policy=pol, precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                     f1=f1_of(tp, fp, fn), mcc=mcc_of(tp, fp, fn, tn), pr_auc=ap_all, roc_auc=auc_all, pr_auc_late=ap_late)
            for nm, mk in (('early', early), ('late', late)):
                a = counts(pr[mk], yt[mk]); r[f'f1_{nm}'] = f1_of(*a[:3]); r[f'precision_{nm}'] = a[0] / max(a[0] + a[1], 1); r[f'recall_{nm}'] = a[0] / max(a[0] + a[2], 1)
            rows.append(r)
            for st in steps:
                mk = tt == st; a = counts(pr[mk], yt[mk])
                step_rows.append(dict(config=name, policy=pol, seed=s, step=st, tp=a[0], fp=a[1], fn=a[2], tn=a[3], f1=f1_of(*a[:3])))
    print('evaluated', name, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, 'metrics_per_seed.csv'), index=False)
pd.DataFrame(step_rows).to_csv(os.path.join(RES, 'per_step_per_seed.csv'), index=False)
pd.DataFrame(prior_rows).to_csv(os.path.join(RES, 'prior_estimates_seed0.csv'), index=False)
R = pd.DataFrame(rows)
num = [c for c in R.columns if c not in ('config', 'model', 'features', 'seed', 'policy')]
S = R.groupby(['config', 'policy'])[num].agg(['mean', 'std'])
S.columns = ['_'.join(c) for c in S.columns]
S.reset_index().to_csv(os.path.join(RES, 'metrics_summary.csv'), index=False)

cost = []
for f in sorted(glob.glob(os.path.join(RES, 'preds', '*.npz'))):
    name = os.path.basename(f)[:-4]; z = np.load(f); meta = json.load(open(f[:-4] + '.json'))
    cost.append(dict(config=name, train_s_mean=float(z['t_train'].mean()), train_s_std=float(z['t_train'].std(ddof=1)), infer_s_mean=float(z['t_infer'].mean()),
                     params=meta.get('params', ''), n_features=meta.get('n_features', ''), hyperparams=json.dumps(meta.get('hyperparams', ''))))
pd.DataFrame(cost).to_csv(os.path.join(RES, 'cost.csv'), index=False)
