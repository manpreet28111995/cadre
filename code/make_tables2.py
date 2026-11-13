import numpy as np, pandas as pd, os
from config import RES, ROOT

S = pd.read_csv(os.path.join(RES, 'stream_summary.csv'))
S['tag'] = S['tag'].fillna('')
SV = S[S.tag == '_v2']
S = S[S.tag == '']
G = pd.read_csv(os.path.join(RES, 'stream_significance.csv'))
OLD = pd.read_csv(os.path.join(RES, 'metrics_summary.csv')).set_index(['config', 'policy'])
COST = pd.read_csv(os.path.join(RES, 'cost.csv')).set_index('config')
CS = pd.read_csv(os.path.join(RES, 'stream_cost.csv')) if os.path.exists(os.path.join(RES, 'stream_cost.csv')) else None
out = os.path.join(ROOT, 'tables'); os.makedirs(out, exist_ok=True)

def get(model, fs, pol, d, dec, cut=34):
    r = S[(S.model == model) & (S.fs == fs) & (S.policy == pol) & (S.d == d) & (S.dec == dec) & (S.cut == cut)]
    return None if r.empty else r.iloc[0]

def ms(r, key, dg=3, std=True, bold=False):
    m, s = r[key + '_mean'], r[key + '_std']
    t = f'{m:.{dg}f}' + (f'\\,$\\pm$\\,{s:.{dg}f}' if (std and s == s and s > 0.0005) else '')
    return f'\\textbf{{{t}}}' if bold else t

def sg(v): return f'{v:+.3f}'.replace('-', '$-$')

main = [('LR__TX', 'Logistic regression', 'T'), ('MLP__TX', 'MLP', 'T'), ('GCN__TX', 'GCN', 'T+G'), ('SAGE__TX', 'GraphSAGE', 'T+G'), ('GAT__TX', 'GAT', 'T+G'),
        ('RF__TX', 'Random Forest', 'T'), ('XGB__TX', 'XGBoost', 'T'), ('LGBM__TX', 'LightGBM', 'T')]
rows = []
for c, nm, inp in main:
    r = OLD.loc[(c, 'fix')]
    rows.append((nm, inp, {k: (r[k + '_mean'], r[k + '_std']) for k in ('f1', 'precision', 'recall', 'mcc', 'pr_auc', 'roc_auc', 'f1_early', 'f1_late')}, COST.loc[c, 'train_s_mean']))
r = get('LGBM', 'TXSR', 'static', 1, 'fix')
rows.append(('LightGBM + context', 'T+C', {k: (r[k + '_mean'], r[k + '_std']) for k in ('f1', 'precision', 'recall', 'mcc', 'pr_auc', 'roc_auc', 'f1_early', 'f1_late')}, r.train_s_mean))
keys = ('f1', 'precision', 'recall', 'mcc', 'pr_auc', 'f1_early', 'f1_late')
best = {k: max(x[2][k][0] for x in rows) for k in keys}
L = ['\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccccr@{}}', '\\toprule', 'Model & Input & F1 & Precision & Recall & MCC & PR-AUC & F1 35--42 & F1 43--49 \\\\', '\\midrule']
for nm, inp, m, tr in rows:
    cells = []
    for k in ('f1', 'precision', 'recall', 'mcc', 'pr_auc', 'f1_early', 'f1_late'):
        mu, sd = m[k]
        t = f'{mu:.3f}' + (f'\\,$\\pm$\\,{sd:.3f}' if sd == sd and sd > 0.0005 else '')
        cells.append(f'\\textbf{{{t}}}' if abs(mu - best[k]) < 1e-9 else t)
    L.append(f"{nm} & {inp} & " + ' & '.join(cells) + ' \\\\')
    if nm in ('GAT', 'LightGBM'): L.append('\\midrule')
L[-1] = L[-1]
L += ['\\bottomrule', '\\end{tabular*}']
L = [l for i, l in enumerate(L)]
open(os.path.join(out, 'tab_bench.tex'), 'w').write('\n'.join(L) + '\n')

pols = [('static', 'Static'), ('every', 'Every step'), ('slide', 'Sliding window'), ('per3', 'Every 3 steps'), ('per5', 'Every 5 steps'), ('psi', 'Score-PSI trigger'), ('dpm', 'CADRE (DPM trigger)'), ('oracle', 'Oracle-onset trigger')]
L = ['\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccccc@{}}', '\\toprule', ' & \\multicolumn{4}{c}{$d=1$} & \\multicolumn{4}{c}{$d=5$} \\\\', '\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}',
     'Policy & F1 & F1 43--49 & PR-AUC 43--49 & Retr. & F1 & F1 43--49 & PR-AUC 43--49 & Retr. \\\\', '\\midrule']
bestc = {}
for d in (1, 5):
    for k in ('f1', 'f1_late', 'pr_auc_late'):
        bestc[(d, k)] = max(get('LGBM', 'TXSR', p, d, 'fix')[k + '_mean'] for p, _ in pols if get('LGBM', 'TXSR', p, d, 'fix') is not None)
for p, nm in pols:
    cells = []
    for d in (1, 5):
        r = get('LGBM', 'TXSR', p, d, 'fix')
        if r is None: cells += ['--'] * 4; continue
        for k in ('f1', 'f1_late', 'pr_auc_late'):
            cells.append(ms(r, k, bold=abs(r[k + '_mean'] - bestc[(d, k)]) < 1e-9))
        cells.append(f"{r.retrains_mean:.0f}")
    L.append(f'{nm} & ' + ' & '.join(cells) + ' \\\\')
    if p in ('static', 'per5', 'dpm'): L.append('\\midrule' if p != 'dpm' else '')
L = [l for l in L if l != '']
L += ['\\bottomrule', '\\end{tabular*}']
open(os.path.join(out, 'tab_policies.tex'), 'w').write('\n'.join(L) + '\n')

def pick(model, fs, pol, d, cut, tag):
    D_ = SV if tag else S
    r = D_[(D_.model == model) & (D_.fs == fs) & (D_.policy == pol) & (D_.d == d) & (D_.cut == cut) & (D_.dec == 'fix')]
    return None if r.empty else r.iloc[0]
settings = [('block', 'Feature ablation (LightGBM, $d=1$, five seeds)'),
            ('T only', 'LGBM', 'TX', 1, 34, False, 5), ('T+S', 'LGBM', 'TXS', 1, 34, False, 5), ('T+R', 'LGBM', 'TXR', 1, 34, False, 5), ('T+S+R', 'LGBM', 'TXSR', 1, 34, False, 5),
            ('block', 'Other learners (T+S+R, $d=1$, three seeds)'),
            ('XGBoost', 'XGB', 'TXSR', 1, 34, False, 3), ('Random Forest', 'RF', 'TXSR', 1, 34, False, 3),
            ('block', 'Validation buffer of two revealed steps (LightGBM, T+S+R, five seeds)'),
            ('$d=1$', 'LGBM', 'TXSR', 1, 34, True, 5), ('$d=3$', 'LGBM', 'TXSR', 3, 34, True, 5),
            ('block', 'Rolling origin (LightGBM, T+S+R, $d=1$, three seeds)'),
            ('$t_0=24$', 'LGBM', 'TXSR', 1, 24, False, 3), ('$t_0=29$', 'LGBM', 'TXSR', 1, 29, False, 3), ('$t_0=39$', 'LGBM', 'TXSR', 1, 39, False, 3)]
L = ['\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccccc@{}}', '\\toprule', ' & \\multicolumn{2}{c}{Static} & \\multicolumn{2}{c}{Every step} & \\multicolumn{3}{c}{CADRE} \\\\'.replace('lcccccc', 'lcccccc'),
     '\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}', 'Setting & F1 & F1 43--49 & F1 & F1 43--49 & F1 & F1 43--49 & Retr. \\\\', '\\midrule']
L[0] = '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lccccccc@{}}'
first = True
for row in settings:
    if row[0] == 'block':
        if not first: L.append('\\midrule')
        L.append('\\multicolumn{8}{l}{\\emph{' + row[1] + '}} \\\\'); first = False; continue
    lab, m, fs_, d_, c, v, ns = row
    cells = []
    for p in ('static', 'every', 'dpm'):
        r = pick(m, fs_, p, d_, c, v)
        if r is None: cells += ['--', '--'] + (['--'] if p == 'dpm' else []); continue
        cells += [ms(r, 'f1'), ms(r, 'f1_late')] + ([f"{r.retrains_mean:.0f}"] if p == 'dpm' else [])
    L.append(f"{lab} & " + ' & '.join(cells) + ' \\\\')
L += ['\\bottomrule', '\\end{tabular*}']
open(os.path.join(out, 'tab_robust.tex'), 'w').write('\n'.join(L) + '\n')

def sigrow(label, fam, comp, d):
    r = G[(G.family == fam) & (G.comparison == comp) & (G.d == d)]
    if r.empty: return None
    r = r.iloc[0]
    h = f'{r.holm_p:.3f}' if r.holm_p >= 0.001 else '$<$0.001'
    return f"{label} & {sg(r.diff_all)} [{sg(r.lo_all)}, {sg(r.hi_all)}] & {sg(r.diff_late)} [{sg(r.lo_late)}, {sg(r.hi_late)}] & {int(r.steps_a_better)}/{int(r.steps_b_better)} & {h} \\\\"
sigs = [('Every step vs. static, $d=1$', 'policy_vs_static_fix', 'every_vs_static', 1), ('CADRE vs. static, $d=1$', 'policy_vs_static_fix', 'dpm_vs_static', 1),
        ('CADRE vs. every step, $d=1$', 'dpm_vs_every_fix', 'dpm_vs_every', 1), ('Oracle onset vs. CADRE, $d=1$', 'oracle_vs_dpm_fix', 'oracle_vs_dpm', 1),
        ('Every step vs. static, $d=5$', 'policy_vs_static_fix', 'every_vs_static', 5), ('CADRE vs. static, $d=5$', 'policy_vs_static_fix', 'dpm_vs_static', 5),
        ('CADRE vs. every step, $d=5$', 'dpm_vs_every_fix', 'dpm_vs_every', 5),
        ('Adapted vs. fixed rule, CADRE, $d=1$', 'psa_vs_fix', 'dpm_psa_vs_fix', 1), ('Adapted vs. fixed rule, every step, $d=1$', 'psa_vs_fix', 'every_psa_vs_fix', 1),
        ('T+S+R vs. T, every step, $d=1$', 'txsr_vs_tx_fix', 'every_txsr_vs_tx', 1), ('T+S+R vs. T+S, every step, $d=1$', 'ctx_effect_fix', 'every_txsr_vs_txs', 1)]
L = ['\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcccc@{}}', '\\toprule', 'Comparison & $\\Delta$F1, steps 35--49 [95\\% CI] & $\\Delta$F1, steps 43--49 [95\\% CI] & Steps & $p_{\\mathrm{Holm}}$ \\\\', '\\midrule']
for lab, fam, comp, d in sigs:
    if fam is None: continue
    r = sigrow(lab, fam, comp, d)
    if r: L.append(r)
L += ['\\bottomrule', '\\end{tabular*}']
open(os.path.join(out, 'tab_sig2.tex'), 'w').write('\n'.join(L) + '\n')
print('tables2 written')
