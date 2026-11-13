import numpy as np, pandas as pd, os
from config import DATA, CACHE, TRAIN_END

def load_tx():
    tf = pd.read_csv(os.path.join(DATA, 'txs_features.csv'))
    tc = pd.read_csv(os.path.join(DATA, 'txs_classes.csv'))
    df = tf.merge(tc, on='txId', how='left').sort_values('txId').reset_index(drop=True)
    y = df['class'].map({1: 1, 2: 0, 3: -1}).values.astype(np.int8)
    t = df['Time step'].values.astype(np.int16)
    feat_cols = [c for c in df.columns if c not in ('txId', 'Time step', 'class')]
    X = df[feat_cols].values.astype(np.float32)
    return df['txId'].values, t, y, X, feat_cols

def addr_edges(txid, t):
    ts = pd.Series(t, index=txid)
    a = pd.read_csv(os.path.join(DATA, 'AddrTx_edgelist.csv')).rename(columns={'input_address': 'a'})
    a['role'] = 0
    b = pd.read_csv(os.path.join(DATA, 'TxAddr_edgelist.csv')).rename(columns={'output_address': 'a'})
    b['role'] = 1
    e = pd.concat([a[['a', 'txId', 'role']], b[['a', 'txId', 'role']]], ignore_index=True).drop_duplicates()
    e['t'] = e['txId'].map(ts).astype(np.int16)
    return e

def build_causal_actor_features(txid, t, y, e):
    ylab = pd.Series(y, index=txid)
    at = e[['a', 'txId', 't']].drop_duplicates()
    at['y'] = at['txId'].map(ylab).values
    at['ill'] = ((at['y'] == 1) & (at['t'] <= TRAIN_END)).astype(np.int32)
    at['lic'] = ((at['y'] == 0) & (at['t'] <= TRAIN_END)).astype(np.int32)
    g = at.groupby(['a', 't']).agg(n_cur=('txId', 'size'), ill=('ill', 'sum'), lic=('lic', 'sum')).reset_index()
    g = g.sort_values(['a', 't']).reset_index(drop=True)
    grp = g.groupby('a', sort=False)
    g['prior_tx'] = grp['n_cur'].cumsum() - g['n_cur']
    g['prior_steps'] = grp.cumcount()
    g['first_t'] = grp['t'].transform('min')
    g['age'] = g['t'] - g['first_t']
    prev_t = grp['t'].shift(1)
    g['gap'] = (g['t'] - prev_t).fillna(0)
    g['is_new'] = (g['prior_steps'] == 0).astype(np.int8)
    g['ill_prior'] = grp['ill'].cumsum() - g['ill']
    g['lic_prior'] = grp['lic'].cumsum() - g['lic']
    m = e.merge(g[['a', 't', 'n_cur', 'prior_tx', 'prior_steps', 'age', 'gap', 'is_new', 'ill_prior', 'lic_prior']], on=['a', 't'], how='left')
    m['log_prior_tx'] = np.log1p(m['prior_tx'])
    m['log_n_cur'] = np.log1p(m['n_cur'])
    out = {}
    for role, nm in ((0, 'in'), (1, 'out')):
        s = m[m['role'] == role].groupby('txId')
        out[f'S_{nm}_n'] = s.size()
        out[f'S_{nm}_new_frac'] = s['is_new'].mean()
        out[f'S_{nm}_logprior_mean'] = s['log_prior_tx'].mean()
        out[f'S_{nm}_logprior_max'] = s['log_prior_tx'].max()
        out[f'S_{nm}_age_mean'] = s['age'].mean()
        out[f'S_{nm}_age_max'] = s['age'].max()
        out[f'S_{nm}_steps_mean'] = s['prior_steps'].mean()
        out[f'S_{nm}_gap_mean'] = s['gap'].mean()
        out[f'S_{nm}_logcur_max'] = s['log_n_cur'].max()
        out[f'S_{nm}_logcur_mean'] = s['log_n_cur'].mean()
        out[f'R_{nm}_ill_sum'] = s['ill_prior'].sum()
        out[f'R_{nm}_lic_sum'] = s['lic_prior'].sum()
        out[f'R_{nm}_ill_any'] = s['ill_prior'].max().clip(upper=1)
    s = m.groupby('txId')
    out['S_all_n'] = s['a'].nunique()
    out['S_all_new_frac'] = s['is_new'].mean()
    out['R_all_ill_sum'] = s['ill_prior'].sum()
    out['R_all_lic_sum'] = s['lic_prior'].sum()
    out['R_all_ill_frac'] = (out['R_all_ill_sum'] + 0.5) / (out['R_all_ill_sum'] + out['R_all_lic_sum'] + 1.0)
    F = pd.DataFrame(out).reindex(txid)
    F = F.fillna(0.0)
    return F

def build_leaky_wallet_features(txid, e):
    wf = pd.read_csv(os.path.join(DATA, 'wallets_features.csv'))
    wf.columns = [c.replace(' ', '_') for c in wf.columns]
    cols = [c for c in wf.columns if c not in ('address', 'Time_step')]
    wf = wf.drop_duplicates('address').set_index('address')[cols].astype(np.float32)
    m = e[['a', 'txId']].drop_duplicates().join(wf, on='a')
    m = m.dropna(subset=cols[:1])
    g = m.groupby('txId')[cols]
    mean = g.mean().add_prefix('W_mean_')
    mx = g.max().add_prefix('W_max_')
    F = pd.concat([mean, mx], axis=1).reindex(txid).fillna(0.0)
    return F

if __name__ == '__main__':
    txid, t, y, X, cols = load_tx()
    e = addr_edges(txid, t)
    caf = build_causal_actor_features(txid, t, y, e)
    wfl = build_leaky_wallet_features(txid, e)
    idx = pd.Series(np.arange(len(txid)), index=txid)
    ed = pd.read_csv(os.path.join(DATA, 'txs_edgelist.csv'))
    src = idx[ed['txId1']].values; dst = idx[ed['txId2']].values
    np.savez_compressed(os.path.join(CACHE, 'base.npz'), txid=txid, t=t, y=y, X=X, cols=np.array(cols), src=src, dst=dst,
                        caf=caf.values.astype(np.float32), caf_cols=np.array(caf.columns),
                        wl=wfl.values.astype(np.float32), wl_cols=np.array(wfl.columns))
    print('tx', X.shape, 'caf', caf.shape, 'leaky wallet', wfl.shape, 'edges', len(src))
    lab = y >= 0
    print('labelled', lab.sum(), 'illicit', (y == 1).sum())
