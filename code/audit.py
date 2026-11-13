import numpy as np, pandas as pd, os
from config import DATA, RES

wf = pd.read_csv(os.path.join(DATA, 'wallets_features.csv'))
wf.columns = [c.replace(' ', '_') for c in wf.columns]
feat = [c for c in wf.columns if c not in ('address', 'Time_step')]
multi = wf.groupby('address')['Time_step'].transform('size') > 1
w = wf[multi]
rows = [(c, float((w.groupby('address')[c].nunique(dropna=False) == 1).mean())) for c in feat]
pd.DataFrame(rows, columns=['feature', 'share_constant_across_steps']).to_csv(os.path.join(RES, 'audit_constant_share.csv'), index=False)

first_t = wf.groupby('address')['Time_step'].transform('min')
wf['first_step'] = first_t
end_block = wf[wf['Time_step'] == wf['first_step']].groupby('Time_step')['first_block_appeared_in'].max()
wf['end_block'] = wf['Time_step'].map(end_block)
wf['future_block'] = wf['last_block_appeared_in'] > wf['end_block']
at = pd.read_csv(os.path.join(DATA, 'AddrTx_edgelist.csv')).rename(columns={'input_address': 'a'})
ta = pd.read_csv(os.path.join(DATA, 'TxAddr_edgelist.csv')).rename(columns={'output_address': 'a'})
tf = pd.read_csv(os.path.join(DATA, 'txs_features.csv'), usecols=['txId', 'Time step']).set_index('txId')['Time step']
e = pd.concat([at[['a', 'txId']], ta[['a', 'txId']]]).drop_duplicates()
e['t'] = e['txId'].map(tf)
cnt = e.groupby(['a', 't']).size().rename('n_cur').reset_index().sort_values(['a', 't'])
cnt['cum_obs'] = cnt.groupby('a')['n_cur'].cumsum()
m = wf[['address', 'Time_step', 'total_txs', 'future_block']].merge(cnt[['a', 't', 'cum_obs']], left_on=['address', 'Time_step'], right_on=['a', 't'], how='left')
m['future_count'] = m['total_txs'] > m['cum_obs']
per = m.groupby('Time_step').agg(rows=('address', 'size'), share_future_block=('future_block', 'mean'), share_future_count=('future_count', 'mean')).reset_index()
per.to_csv(os.path.join(RES, 'audit_future_by_step.csv'), index=False)
summ = {'rows': len(m), 'n_features': len(feat), 'n_features_constant_ge_99pct': int(sum(r[1] >= 0.99 for r in rows)),
        'share_rows_future_block': float(m['future_block'].mean()), 'share_rows_future_count': float(m['future_count'].mean()),
        'share_rows_future_block_steps_1_to_34': float(m[m.Time_step <= 34]['future_block'].mean()),
        'max_last_block_at_step1': float(wf[wf.Time_step == 1]['last_block_appeared_in'].max()),
        'end_block_step1': float(end_block.loc[1])}
pd.Series(summ).to_csv(os.path.join(RES, 'audit_summary.csv'), header=['value'])
print(pd.Series(summ)); print(per.head(8).to_string())
