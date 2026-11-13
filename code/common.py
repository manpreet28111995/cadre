import numpy as np, os, json, time
from config import CACHE, RES, TRAIN_END, VAL_END
from sklearn.metrics import average_precision_score

def load_base():
    z = np.load(os.path.join(CACHE, 'base.npz'), allow_pickle=True)
    return {k: z[k] for k in z.files}

def feature_sets(B):
    caf_cols = list(B['caf_cols'])
    s_idx = [i for i, c in enumerate(caf_cols) if c.startswith('S_')]
    r_idx = [i for i, c in enumerate(caf_cols) if c.startswith('R_')]
    fs = {
        'TX': B['X'],
        'TX+S': np.hstack([B['X'], B['caf'][:, s_idx]]),
        'TX+S+R': np.hstack([B['X'], B['caf'][:, s_idx + r_idx]]),
        'TX+W': np.hstack([B['X'], B['wl']]),
    }
    return fs

def splits(B):
    t, y = B['t'], B['y']
    lab = y >= 0
    tr = np.where(lab & (t <= TRAIN_END))[0]
    va = np.where(lab & (t > TRAIN_END) & (t <= VAL_END))[0]
    te = np.where(lab & (t > VAL_END))[0]
    return tr, va, te

def best_f1_threshold(y, p):
    order = np.argsort(-p)
    ys = y[order]; ps = p[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    P = ys.sum()
    f1 = 2 * tp / (tp + fp + P)
    distinct = np.r_[ps[1:] != ps[:-1], True]
    f1 = np.where(distinct, f1, -1)
    i = int(np.argmax(f1))
    thr = ps[i] if i == len(ps) - 1 else 0.5 * (ps[i] + ps[i + 1])
    return float(thr), float(f1[i])

def ap(y, p):
    return float(average_precision_score(y, p))

def save_preds(name, p_val, p_test, t_train, t_infer, meta):
    d = os.path.join(RES, 'preds'); os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, name + '.npz'), p_val=np.array(p_val, dtype=np.float32), p_test=np.array(p_test, dtype=np.float32),
                        t_train=np.array(t_train), t_infer=np.array(t_infer))
    with open(os.path.join(d, name + '.json'), 'w') as f:
        json.dump(meta, f, indent=1)
