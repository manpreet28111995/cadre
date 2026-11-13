import numpy as np, pandas as pd, os, json
from config import RES, ROOT
S = pd.read_csv(os.path.join(RES, 'metrics_summary.csv')); C = pd.read_csv(os.path.join(RES, 'cost.csv')).set_index('config')
SG = pd.read_csv(os.path.join(RES, 'significance.csv')); SP = pd.read_csv(os.path.join(RES, 'split_stats.csv')).set_index('split')
AU = pd.read_csv(os.path.join(RES, 'audit_summary.csv'), index_col=0)['value']; AC = pd.read_csv(os.path.join(RES, 'audit_constant_share.csv'))
PR = pd.read_csv(os.path.join(RES, 'prior_estimates_seed0.csv')); LR = pd.read_csv(os.path.join(RES, 'late_period_ranking.csv'))
SHORT = {'LR__TX': 'LR', 'MLP__TX': 'MLP', 'GCN__TX': 'GCN', 'SAGE__TX': 'SAGE', 'GAT__TX': 'GAT', 'RF__TX': 'RFt', 'XGB__TX': 'XGBt', 'LGBM__TX': 'LGBMt', 'LGBM__TX+S': 'LGBMs',
         'LGBM__TX+S+R': 'LGBMc', 'LGBM__TX+W': 'LGBMw', 'RF__TX+S+R': 'RFc', 'XGB__TX+S+R': 'XGBc'}
D = {}
def put(k, v): D[k] = v
metrics = ['f1', 'precision', 'recall', 'mcc', 'pr_auc', 'roc_auc', 'f1_early', 'f1_late', 'precision_late', 'recall_late', 'pr_auc_late', 'precision_early', 'recall_early']
for _, r in S.iterrows():
    if r.config not in SHORT: continue
    for m in metrics:
        put(f"{SHORT[r.config]}.{r.policy}.{m.replace('_', '-')}.m", f"{r[m + '_mean']:.3f}"); put(f"{SHORT[r.config]}.{r.policy}.{m.replace('_', '-')}.s", f"{r[m + '_std']:.3f}" if r[m + '_std'] == r[m + '_std'] else '0.000')
for c, k in SHORT.items():
    if c in C.index:
        put(f'cost.{k}.train', f"{C.loc[c, 'train_s_mean']:.1f}"); put(f'cost.{k}.infer', f"{C.loc[c, 'infer_s_mean']:.2f}"); put(f'cost.{k}.nfeat', str(C.loc[c, 'n_features']))
        if str(C.loc[c, 'params']) not in ('', 'nan'): put(f'cost.{k}.params', f"{int(float(C.loc[c, 'params'])):,}".replace(',', '{,}'))
sigmap = {'sigPolC': ('policy_psa_vs_fix', 'LGBM__TX+S+R|psa', 'LGBM__TX+S+R|fix'), 'sigPolT': ('policy_psa_vs_fix', 'LGBM__TX|psa', 'LGBM__TX|fix'), 'sigRfix': ('features_R_fix', 'LGBM__TX+S+R|fix', 'LGBM__TX|fix'),
          'sigRpsa': ('features_R_psa', 'LGBM__TX+S+R|psa', 'LGBM__TX|psa'), 'sigSfix': ('features_S_fix', 'LGBM__TX+S|fix', 'LGBM__TX|fix'), 'sigRF': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'RF__TX|psa'),
          'sigXGB': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'XGB__TX|psa'), 'sigSAGE': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'SAGE__TX|psa'), 'sigLGBMt': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'LGBM__TX|psa'),
          'sigGCN': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'GCN__TX|psa'), 'sigGAT': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'GAT__TX|psa'), 'sigMLP': ('proposed_vs_psa', 'LGBM__TX+S+R|psa', 'MLP__TX|psa')}
for k, (cmp_, a, b) in sigmap.items():
    r = SG[(SG.comparison == cmp_) & (SG.a == a) & (SG.b == b)]
    if r.empty: continue
    r = r.iloc[0]
    put(f'{k}.diff', f'{r["diff"]:+.3f}'.replace('-', '$-$')); put(f'{k}.lo', f'{r.ci_low:+.3f}'.replace('-', '$-$')); put(f'{k}.hi', f'{r.ci_high:+.3f}'.replace('-', '$-$'))
    put(f'{k}.better', str(int(r.steps_a_better))); put(f'{k}.worse', str(int(r.steps_b_better))); put(f'{k}.p', f'{r.wilcoxon_p:.3f}'); put(f'{k}.holm', f'{r.holm_p:.3f}')
put('audit.nfeat', str(int(AU['n_features']))); put('audit.nconst', str(int(AU['n_features_constant_ge_99pct']))); put('audit.share', f"{100 * AU['share_rows_future_block']:.1f}"); put('audit.sharecount', f"{100 * AU['share_rows_future_count']:.1f}")
put('audit.sharetv', f"{100 * AU['share_rows_future_block_steps_1_to_34']:.1f}"); put('audit.rows', f"{int(AU['rows']):,}".replace(',', '{,}')); put('audit.lastblock', f"{int(AU['max_last_block_at_step1']):,}".replace(',', '{,}')); put('audit.endblock', f"{int(AU['end_block_step1']):,}".replace(',', '{,}'))
for nm, k in (('train', 'train'), ('validation', 'val'), ('test', 'test'), ('test early', 'early'), ('test late', 'late')):
    put(f'split.{k}.n', f"{int(SP.loc[nm, 'labelled']):,}".replace(',', '{,}')); put(f'split.{k}.ill', f"{int(SP.loc[nm, 'illicit']):,}".replace(',', '{,}')); put(f'split.{k}.prev', f"{100 * SP.loc[nm, 'prevalence']:.1f}")
pp = PR[PR.config == 'LGBM__TX+S+R']
for nm, mk in (('early', pp.step <= 42), ('late', pp.step >= 43), ('all', pp.step > 0)):
    put(f'prior.mae.{nm}', f"{(pp[mk].est_prior - pp[mk].window_true_prior).abs().mean():.3f}")
    put(f'prior.bias.{nm}', f"{(pp[mk].est_prior - pp[mk].window_true_prior).mean():+.3f}".replace('-', '$-$'))
for _, r in LR.iterrows():
    k = SHORT.get(r.config)
    if k: put(f'late.{k}.{r.period}.roc', f'{r.roc_auc:.3f}'); put(f'late.{k}.{r.period}.pr', f'{r.pr_auc:.3f}'); put(f'late.{k}.{r.period}.base', f'{r.base_rate:.3f}')
import glob
for c, k in SHORT.items():
    fz = os.path.join(RES, 'preds', c + '.npz')
    if os.path.exists(fz):
        from common import load_base, splits
        if 'yv' not in globals():
            Bb = load_base(); _, vv, _ = splits(Bb); yv = Bb['y'][vv]
        from sklearn.metrics import average_precision_score as aps
        put(f'valap.{k}', f"{np.mean([aps(yv, p) for p in np.load(fz)['p_val']]):.3f}")
with open(os.path.join(ROOT, 'numbers.tex'), 'w') as f:
    for k, v in D.items(): f.write('\\expandafter\\gdef\\csname num@%s\\endcsname{%s}\n' % (k, v))
print(len(D), 'macros')
