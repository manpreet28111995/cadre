import numpy as np, pandas as pd, os, sys
from config import CACHE, VAL_END as HIST_END
from prep import load_tx, addr_edges

def reputation_features(txid, t, y, e, d):
    ylab = pd.Series(y, index=txid)
    at = e[['a', 'txId', 't']].drop_duplicates()
    at['y'] = at['txId'].map(ylab).values
    at['ill'] = (at['y'] == 1).astype(np.int32)
    at['lic'] = (at['y'] == 0).astype(np.int32)
    g = at.groupby(['a', 't']).agg(ill=('ill', 'sum'), lic=('lic', 'sum')).reset_index().sort_values(['t', 'a']).reset_index(drop=True)
    g = g.sort_values(['a', 't']).reset_index(drop=True)
    g['cum_ill'] = g.groupby('a')['ill'].cumsum()
    g['cum_lic'] = g.groupby('a')['lic'].cumsum()
    q = g[['a', 't']].copy()
    q['tq'] = np.where(q['t'] <= HIST_END, q['t'] - d, np.maximum(HIST_END, q['t'] - d))
    q = q.sort_values('tq').reset_index()
    r = g[['a', 't', 'cum_ill', 'cum_lic']].rename(columns={'t': 'tr'}).sort_values('tr')
    mm = pd.merge_asof(q, r, left_on='tq', right_on='tr', by='a', direction='backward')
    mm = mm.sort_values('index').set_index('index')
    g['ill_prior'] = mm['cum_ill'].fillna(0).values
    g['lic_prior'] = mm['cum_lic'].fillna(0).values
    m = e.merge(g[['a', 't', 'ill_prior', 'lic_prior']], on=['a', 't'], how='left')
    out = {}
    for role, nm in ((0, 'in'), (1, 'out')):
        s = m[m['role'] == role].groupby('txId')
        out[f'R_{nm}_ill_sum'] = s['ill_prior'].sum()
        out[f'R_{nm}_lic_sum'] = s['lic_prior'].sum()
        out[f'R_{nm}_ill_any'] = s['ill_prior'].max().clip(upper=1)
    s = m.groupby('txId')
    out['R_all_ill_sum'] = s['ill_prior'].sum()
    out['R_all_lic_sum'] = s['lic_prior'].sum()
    out['R_all_ill_frac'] = (out['R_all_ill_sum'] + 0.5) / (out['R_all_ill_sum'] + out['R_all_lic_sum'] + 1.0)
    return pd.DataFrame(out).reindex(txid).fillna(0.0)

if __name__ == '__main__':
    txid, t, y, X, cols = load_tx()
    e = addr_edges(txid, t)
    ds = [int(a) for a in sys.argv[1:]] or [1, 2, 3, 5]
    res = {}
    for d in ds:
        F = reputation_features(txid, t, y, e, d)
        res[f'R_d{d}'] = F.values.astype(np.float32); res['R_cols'] = np.array(F.columns)
        print('delay', d, F.shape, 'share with any illicit history %.4f' % (F['R_all_ill_sum'] > 0).mean(), flush=True)
    np.savez_compressed(os.path.join(CACHE, 'R_delay.npz'), **res)
