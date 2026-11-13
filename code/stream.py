import os, sys, time, json, argparse
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from common import load_base, best_f1_threshold
from config import RES, CACHE

EPS = 1e-6
NVAL = 5
LAST = 49
SHIFT_STEP = 43
ap_ = argparse.ArgumentParser()
ap_.add_argument('--model', default='LGBM')
ap_.add_argument('--fs', default='TXSR')
ap_.add_argument('--policy', default='static')
ap_.add_argument('--d', type=int, default=1)
ap_.add_argument('--seeds', default='0-4')
ap_.add_argument('--cut', type=int, default=34)
ap_.add_argument('--win', type=int, default=10)
ap_.add_argument('--tag', default='')
ap_.add_argument('--diag', action='store_true')
ap_.add_argument('--nval', type=int, default=5)
A = ap_.parse_args()
NVAL = A.nval
lo_, hi_ = A.seeds.split('-')
SEEDS = list(range(int(lo_), int(hi_) + 1))
CUT, FIRST, D = A.cut, A.cut + 1, A.d

B = load_base(); T = B['t']; Y = B['y']
caf_cols = list(B['caf_cols']); s_idx = [i for i, c in enumerate(caf_cols) if c.startswith('S_')]
if A.fs == 'TX':
    X = B['X']
else:
    Rz = np.load(os.path.join(CACHE, 'R_delay.npz'))
    parts = [B['X']]
    if 'S' in A.fs[2:]: parts.append(B['caf'][:, s_idx])
    if 'R' in A.fs[2:]: parts.append(Rz[f'R_d{D}'])
    X = np.hstack(parts)
STEP_IDX = {s: np.where(T == s)[0] for s in range(1, LAST + 1)}
LAB_IDX = {s: np.where((T == s) & (Y >= 0))[0] for s in range(1, LAST + 1)}
TE = np.concatenate([LAB_IDX[s] for s in range(FIRST, LAST + 1)])
TE_STEP = T[TE]
POS = {s: np.where(TE_STEP == s)[0] for s in range(FIRST, LAST + 1)}

def logit(p):
    p = np.clip(p, EPS, 1 - EPS); return np.log(p / (1 - p))

def sig(z): return 1 / (1 + np.exp(-z))

def f1c(pred, yy):
    tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum()); fn = int(((pred == 0) & (yy == 1)).sum())
    return (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

def em_prior(p, pi_s, iters=200):
    pi = pi_s
    for _ in range(iters):
        w1 = pi / pi_s * p; w0 = (1 - pi) / (1 - pi_s) * (1 - p)
        new = np.mean(w1 / (w1 + w0))
        if abs(new - pi) < 1e-7: pi = new; break
        pi = new
    return float(np.clip(pi, 1e-3, 0.6))

def psi(ref, cur, edges):
    a = np.bincount(np.searchsorted(edges, ref), minlength=len(edges) + 1) / len(ref)
    b = np.bincount(np.searchsorted(edges, cur), minlength=len(edges) + 1) / len(cur)
    a = np.clip(a, 1e-4, None); b = np.clip(b, 1e-4, None)
    return float(((b - a) * np.log(b / a)).sum())

def c2st(Xa, Xb, seed, n=1500):
    rng = np.random.RandomState(seed)
    ia = rng.choice(len(Xa), min(n, len(Xa)), replace=False); ib = rng.choice(len(Xb), min(n, len(Xb)), replace=False)
    Z = np.vstack([Xa[ia], Xb[ib]]); z = np.r_[np.zeros(len(ia)), np.ones(len(ib))]
    clf = lgb.LGBMClassifier(n_estimators=40, num_leaves=15, learning_rate=0.1, verbose=-1, n_jobs=2, random_state=seed)
    pr = cross_val_predict(clf, Z, z, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method='predict_proba')[:, 1]
    return float(roc_auc_score(z, pr))

def fit_model(kind, Xtr, ytr, Xva, yva, seed):
    if kind == 'LGBM':
        P = dict(objective='binary', learning_rate=0.1, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, min_child_samples=20, verbose=-1,
                 num_threads=2, seed=seed, num_leaves=31, scale_pos_weight=3, metric='average_precision')
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
        m = lgb.train(P, dtr, 600, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        return (lambda Z: m.predict(Z, num_iteration=m.best_iteration)), m
    if kind == 'XGB':
        m = xgb.XGBClassifier(tree_method='hist', n_estimators=600, learning_rate=0.1, subsample=0.8, colsample_bytree=0.7, n_jobs=2, random_state=seed,
                              eval_metric='aucpr', early_stopping_rounds=50, max_depth=8, scale_pos_weight=3)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        return (lambda Z: m.predict_proba(Z)[:, 1]), m
    m = RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=seed, max_depth=20, min_samples_leaf=1, class_weight='balanced_subsample')
    m.fit(np.nan_to_num(Xtr, nan=0.0), ytr)
    return (lambda Z: m.predict_proba(np.nan_to_num(Z, nan=0.0))[:, 1]), m

def need_null(policy): return policy in ('psi', 'static') or A.diag

def train_deployed(r, win, seed, policy):
    hi = r - NVAL; lo = 1 if win is None else max(1, hi - win + 1)
    tr = np.where((Y >= 0) & (T >= lo) & (T <= hi))[0]; va = np.where((Y >= 0) & (T > hi) & (T <= r))[0]
    t0 = time.time()
    pred, m = fit_model(A.model, X[tr], Y[tr], X[va], Y[va], seed)
    ttr = time.time() - t0
    pv = pred(X[va]); yv = Y[va]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(pv).reshape(-1, 1), yv)
    a, b = lr.coef_[0, 0], lr.intercept_[0]
    cal = lambda p: sig(a * logit(p) + b)
    cv = cal(pv); tau, f1v = best_f1_threshold(yv, cv)
    D_ = dict(pred=pred, cal=cal, tau=tau, base_logit=float(logit(tau)), pi_val=float(yv.mean()), f1_val=f1v, r=r, hi=hi, ntrain=len(tr), train_s=ttr, th_psi=None, th_c2=None)
    vs = np.concatenate([STEP_IDX[s] for s in range(hi + 1, r + 1)])
    D_['ref_scores'] = cal(pred(X[vs]))
    D_['edges'] = np.quantile(D_['ref_scores'], np.linspace(0, 1, 11)[1:-1])
    D_['ref_X'] = X[vs]
    if need_null(policy):
        nulls, nulls2 = [], []
        for s in range(hi + 1, r + 1):
            cur = np.concatenate([STEP_IDX[s]]); oth = np.concatenate([STEP_IDX[q] for q in range(hi + 1, r + 1) if q != s])
            nulls.append(psi(cal(pred(X[oth])), cal(pred(X[cur])), D_['edges']))
            if policy == 'c2st' or A.diag: nulls2.append(c2st(X[oth], X[cur], seed))
        D_['th_psi'] = float(max(nulls)); D_['th_c2'] = float(max(nulls2)) if nulls2 else None
    return D_

def run_seed(seed):
    win = A.win if A.policy == 'slide' else None
    out = dict(p=np.zeros(len(TE), np.float32), fix=np.zeros(len(TE), np.int8), psa=np.zeros(len(TE), np.int8),
               retrain=np.zeros(LAST + 1, np.int8), pi_hat=np.zeros(LAST + 1), tau=np.zeros(LAST + 1), psi=np.full(LAST + 1, np.nan), c2=np.full(LAST + 1, np.nan),
               th_psi=np.full(LAST + 1, np.nan), th_c2=np.full(LAST + 1, np.nan), dpm=np.full(LAST + 1, np.nan), dpm_ref=np.full(LAST + 1, np.nan),
               alarm=np.zeros(LAST + 1, np.int8), horizon=np.zeros(LAST + 1, np.int16), train_s=0.0, infer_s=0.0, n_train=0)
    issued = {}
    alarms = []
    if A.policy == 'oracle': alarms = [SHIFT_STEP]
    Dm = None; r_last = None; pending_per = False; dpm_checked = CUT
    for t in range(FIRST, LAST + 1):
        r = max(CUT, t - D)
        retrain = False
        if Dm is None: retrain = True
        else:
            if A.policy == 'dpm' and r > dpm_checked:
                for s in range(dpm_checked + 1, r + 1):
                    if s in issued:
                        fl, yy, f1v_issuer = issued[s]
                        if yy.sum() >= 5:
                            f1s = f1c(fl, yy); out['dpm'][s] = f1s; out['dpm_ref'][s] = f1v_issuer
                            if f1s < 0.5 * f1v_issuer: alarms.append(s); out['alarm'][t] = 1
                dpm_checked = r
            if A.policy in ('every', 'slide'): retrain = r > r_last
            elif A.policy in ('per3', 'per5'):
                k = 3 if A.policy == 'per3' else 5
                if (t - FIRST) % k == 0: pending_per = True
                if pending_per and r > r_last: retrain = True
            elif A.policy in ('psi', 'c2st', 'dpm', 'oracle'):
                if r > r_last and any((r >= a) and (r_last < a + 2) for a in alarms): retrain = True
        if retrain:
            Dm = train_deployed(r, win, seed, A.policy); r_last = r; pending_per = False
            out['retrain'][t] = 1; out['train_s'] += Dm['train_s']; out['n_train'] += 1
        out['horizon'][t] = r_last; out['tau'][t] = Dm['tau']
        t0 = time.time()
        idx = LAB_IDX[t]; p = Dm['cal'](Dm['pred'](X[idx])); out['infer_s'] += time.time() - t0
        pos = POS[t]; out['p'][pos] = p; out['fix'][pos] = (p >= Dm['tau']).astype(np.int8)
        wsteps = [s for s in (t - 2, t - 1, t) if s >= 1]
        wp = np.concatenate([Dm['cal'](Dm['pred'](X[LAB_IDX[s]])) for s in wsteps])
        pi_hat = em_prior(wp, Dm['pi_val']); out['pi_hat'][t] = pi_hat
        out['psa'][pos] = (logit(p) + logit(pi_hat) - logit(Dm['pi_val']) >= Dm['base_logit']).astype(np.int8)
        issued[t] = (out['fix'][pos].astype(int), Y[idx].astype(int), Dm['f1_val'])
        if A.policy in ('psi', 'c2st', 'static'):
            allx = STEP_IDX[t]
            v = psi(Dm['ref_scores'], Dm['cal'](Dm['pred'](X[allx])), Dm['edges']); out['psi'][t] = v; out['th_psi'][t] = Dm['th_psi']
            if A.policy == 'psi' and v > Dm['th_psi']: alarms.append(t); out['alarm'][t] = 1
        if A.policy == 'c2st' or A.diag:
            v2 = c2st(Dm['ref_X'], X[STEP_IDX[t]], seed); out['c2'][t] = v2; out['th_c2'][t] = Dm['th_c2']
            if A.policy == 'c2st' and v2 > Dm['th_c2']: alarms.append(t); out['alarm'][t] = 1
    return out

if __name__ == '__main__':
    d = os.path.join(RES, 'stream'); os.makedirs(d, exist_ok=True)
    name = f"{A.model}_{A.fs}_{A.policy}_d{D}_c{CUT}{A.tag}"
    fn = os.path.join(d, name + '.npz')
    if os.path.exists(fn): print('exists', name); sys.exit(0)
    runs = []
    t0 = time.time()
    for s in SEEDS:
        runs.append(run_seed(s))
        print(name, 'seed', s, 'retrains', runs[-1]['n_train'], 'elapsed %.0fs' % (time.time() - t0), flush=True)
    keys = runs[0].keys()
    np.savez_compressed(fn, te_idx=TE, te_step=TE_STEP, y=Y[TE], seeds=np.array(SEEDS), **{k: np.stack([r[k] for r in runs]) for k in keys})
    with open(os.path.join(d, name + '.json'), 'w') as f:
        json.dump(dict(model=A.model, fs=A.fs, policy=A.policy, d=D, cut=CUT, win=A.win if A.policy == 'slide' else None, seeds=SEEDS, wall_s=time.time() - t0), f)
