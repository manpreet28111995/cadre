import numpy as np, pandas as pd, os
from config import RES, ROOT, CACHE
from common import load_base

S = pd.read_csv(os.path.join(RES, 'stream_summary.csv'))
S['tag'] = S['tag'].fillna('')
S = S[S.tag != '_diag']
G = pd.read_csv(os.path.join(RES, 'stream_significance.csv'))
PI = pd.read_csv(os.path.join(RES, 'protocol_inflation.csv'))
SH = pd.read_csv(os.path.join(RES, 'shap_importance.csv'))
SG = pd.read_csv(os.path.join(RES, 'shap_groups.csv')).set_index('group')
TR = pd.read_csv(os.path.join(RES, 'stream_traces.csv'))
D = {}
def put(k, v): D[k] = v
def neg(x): return x.replace('-', '$-$')

metrics = ['f1', 'precision', 'recall', 'mcc', 'f1_early', 'f1_late', 'f1_rec', 'recall_late', 'precision_late', 'pr_auc', 'roc_auc', 'pr_auc_late', 'roc_auc_late', 'roc_auc_early', 'pr_auc_early', 'retrains', 'train_s', 'infer_s']
for _, r in S.iterrows():
    base = (f'{r.policy}.d{int(r.d)}' if (r.model == 'LGBM' and r.fs == 'TXSR' and int(r.cut) == 34) else
            f'{r.model}{r.fs}.{r.policy}.d{int(r.d)}.c{int(r.cut)}') + r.tag.replace('_', '.')
    for m in metrics:
        k = f"{base}.{r.dec}.{m.replace('_', '-')}"
        dig = 1 if m in ('retrains',) else (1 if m == 'train_s' else 3)
        put(k + '.m', f'{r[m + "_mean"]:.{dig}f}')
        sd = r[m + '_std']
        put(k + '.s', f'{sd:.{dig}f}' if sd == sd else '0.000')
    put(f'{base}.nseeds', str(int(r.n_seeds)))

for _, r in G.iterrows():
    k = f"sg.{r.comparison.replace('_', '-')}.{r.dec}.d{int(r.d)}"
    put(k + '.diff', neg(f'{r.diff_all:+.3f}')); put(k + '.lo', neg(f'{r.lo_all:+.3f}')); put(k + '.hi', neg(f'{r.hi_all:+.3f}'))
    put(k + '.ldiff', neg(f'{r.diff_late:+.3f}')); put(k + '.llo', neg(f'{r.lo_late:+.3f}')); put(k + '.lhi', neg(f'{r.hi_late:+.3f}'))
    put(k + '.a', str(int(r.steps_a_better))); put(k + '.b', str(int(r.steps_b_better)))
    put(k + '.p', f'{r.wilcoxon_p:.3f}' if r.wilcoxon_p >= 0.001 else '$<$0.001')
    put(k + '.holm', f'{r.holm_p:.3f}' if r.holm_p >= 0.001 else '$<$0.001')

for pr in ('random', 'chronological'):
    for fs, nm in (('TX', 'TX'), ('TX+W', 'TXW')):
        x = PI[(PI.protocol == pr) & (PI.features == fs)]['f1']
        put(f'pi.{pr}.{nm}.f1.m', f'{x.mean():.3f}'); put(f'pi.{pr}.{nm}.f1.s', f'{x.std(ddof=1):.3f}')
for fs, nm in (('TX', 'TX'), ('TX+W', 'TXW')):
    a = PI[(PI.protocol == 'random') & (PI.features == fs)].sort_values('seed')['f1'].values
    b = PI[(PI.protocol == 'chronological') & (PI.features == fs)].sort_values('seed')['f1'].values
    put(f'pi.gap.{nm}', f'{(a - b).mean():.3f}')
    from scipy.stats import wilcoxon
    put(f'pi.p.{nm}', f'{wilcoxon(a, b).pvalue:.3f}')
a = PI[(PI.protocol == 'chronological') & (PI.features == 'TX')].sort_values('seed')['f1'].values
b = PI[(PI.protocol == 'chronological') & (PI.features == 'TX+W')].sort_values('seed')['f1'].values
put('pi.wallet.diff', neg(f'{(b - a).mean():+.3f}'))
put('pi.wallet.p', f'{wilcoxon(a, b).pvalue:.3f}')

tot = SG[['static_early', 'static_late', 'static_all', 'retrained_late']].sum()
for g in ('T', 'S', 'R'):
    for c in ('static_early', 'static_late', 'static_all', 'retrained_late'):
        put(f'shap.{g}.{c.replace("_", "-")}', f'{100 * SG.loc[g, c] / tot[c]:.1f}')
SH['lift'] = SH.retrained_late / SH.static_late.clip(lower=1e-9)

B = load_base(); t = B['t']; y = B['y']
put('lab.total', f"{int((y >= 0).sum()):,}".replace(',', '{,}')); put('lab.ill', f"{int((y == 1).sum()):,}".replace(',', '{,}'))
Rz = np.load(os.path.join(CACHE, 'R_delay.npz'), allow_pickle=True); rc = [str(c) for c in Rz['R_cols']]
for d in (1, 5):
    Rd = Rz[f'R_d{d}']; anyill = Rd[:, rc.index('R_all_ill_sum')] > 0; anylic = Rd[:, rc.index('R_all_lic_sum')] > 0
    for nm, mk in (('train', (t <= 29) & (y >= 0)), ('test', (t >= 35) & (y >= 0)), ('late', (t >= 43) & (y >= 0)), ('lateill', (t >= 43) & (y == 1)), ('earlyill', (t >= 35) & (t <= 42) & (y == 1))):
        put(f'cov.d{d}.{nm}.ill', f'{100 * anyill[mk].mean():.1f}'); put(f'cov.d{d}.{nm}.any', f'{100 * (anyill | anylic)[mk].mean():.1f}')

for d in (1, 2, 3, 5):
    for pol in ('dpm', 'psi', 'oracle'):
        x = TR[(TR.policy == pol) & (TR.d == d) & (TR.config.str.startswith('LGBM_TXSR_' + pol + '_d') & ~TR.config.str.contains('diag'))]
        if x.empty: continue
        a = x.groupby('step')['alarm'].mean()
        fa = [int(s) for s in a.index if a[s] > 0.5]
        put(f'alarm.{pol}.d{d}.first', str(min(fa)) if fa else 'none'); put(f'alarm.{pol}.d{d}.n', str(len(fa)))

for pol in ('psi', 'dpm'):
    for d in (1, 2, 3, 5):
        x = TR[(TR.config == f'LGBM_TXSR_{pol}_d{d}_c34') & (TR.alarm > 0)]
        st_ = sorted(int(v) for v in x.step)
        put(f'alarms.{pol}.d{d}', ', '.join(str(v) for v in st_) if st_ else 'none')
        put(f'alarms.{pol}.d{d}.pre', str(sum(v < 43 for v in st_))); put(f'alarms.{pol}.d{d}.n', str(len(st_)))
        put(f'alarms.{pol}.d{d}.first', str(st_[0]) if st_ else 'none')
        z = TR[(TR.config == f'LGBM_TXSR_{pol}_d{d}_c34') & (TR.step == 43)]
for d in (1, 2, 3, 5):
    x = TR[(TR.config == f'LGBM_TXSR_static_d{d}_c34')]
    if not x.empty and x.psi.notna().any():
        put(f'psi.d{d}.at43', f"{x[x.step == 43].psi.values[0]:.3f}"); put(f'psi.d{d}.max.pre', f"{x[x.step < 43].psi.max():.3f}"); put(f'psi.d{d}.th', f"{x.th_psi.dropna().iloc[0]:.3f}")
x = TR[TR.config == 'LGBM_TXSR_dpm_d1_c34']
put('dpm.d1.f1at43', f"{x[x.step == 43].dpm.values[0]:.3f}"); put('dpm.d1.ref43', f"{x[x.step == 43].dpm_ref.values[0]:.3f}")
g = TR[TR.config == 'LGBM_TXSR_static_d1_c34_diag']
put('c2.min', f"{g.c2.min():.3f}"); put('c2.max', f"{g.c2.max():.3f}"); put('c2.th', f"{g.th_c2.dropna().iloc[0]:.3f}")
for pol in ('static', 'every', 'dpm'):
    for d in (1,):
        x = TR[(TR.config == f'LGBM_TXSR_{pol}_d{d}_c34') & (TR.step >= 44)]
        put(f'pihat.{pol}.d{d}.late', f'{x.pi_hat.mean():.3f}'); put(f'pihat.{pol}.d{d}.min', f'{x.pi_hat.min():.3f}')
        y_ = TR[(TR.config == f'LGBM_TXSR_{pol}_d{d}_c34') & (TR.step <= 42)]
        put(f'pihat.{pol}.d{d}.early', f'{y_.pi_hat.mean():.3f}')
def val(model, fs, pol, d, cut, tag, col):
    q = S[(S.model == model) & (S.fs == fs) & (S.policy == pol) & (S.d == d) & (S.cut == cut) & (S.dec == 'fix') & (S.tag == tag)]
    return q[col].values[0]
for c in (24, 29, 39):
    put(f'roll.c{c}.diff', f"{val('LGBM', 'TXSR', 'every', 1, c, '', 'f1_mean') - val('LGBM', 'TXSR', 'static', 1, c, '', 'f1_mean'):.3f}")
put('v2.gap.d1', f"{val('LGBM', 'TXSR', 'every', 1, 34, '_v2', 'f1_late_mean') - val('LGBM', 'TXSR', 'dpm', 1, 34, '_v2', 'f1_late_mean'):.3f}")
put('v2.gap.d3', f"{val('LGBM', 'TXSR', 'every', 3, 34, '_v2', 'f1_late_mean') - val('LGBM', 'TXSR', 'dpm', 3, 34, '_v2', 'f1_late_mean'):.3f}")
put('v1.gap.d1', f"{val('LGBM', 'TXSR', 'every', 1, 34, '', 'f1_late_mean') - val('LGBM', 'TXSR', 'dpm', 1, 34, '', 'f1_late_mean'):.3f}")
CS = pd.read_csv(os.path.join(RES, 'stream_cost.csv'))
for _, r in CS.iterrows():
    k = f"cs.{r.model}{r.fs}.h{int(r.horizon)}.{r.window}"
    put(k + '.train', f'{r.train_s_mean:.1f}'); put(k + '.trains', f'{r.train_s_std:.1f}'); put(k + '.infer', f'{r.infer_ms_per_1k:.1f}')
for m, f in (('LGBM', 'TXSR'), ('XGB', 'TXSR'), ('RF', 'TXSR'), ('LGBM', 'TX')):
    q = CS[(CS.model == m) & (CS.fs == f) & (CS.window == 'all')]
    put(f'cs.{m}{f}.avg.train', f'{q.train_s_mean.mean():.1f}'); put(f'cs.{m}{f}.avg.infer', f'{q.infer_ms_per_1k.mean():.1f}')
    put(f'cs.{m}{f}.min.train', f'{q.train_s_mean.min():.1f}'); put(f'cs.{m}{f}.max.train', f'{q.train_s_mean.max():.1f}')
avg = CS[(CS.model == 'LGBM') & (CS.fs == 'TXSR') & (CS.window == 'all')].train_s_mean.mean()
for d in (1, 3, 5):
    for pol in ('static', 'every', 'slide', 'per3', 'per5', 'dpm', 'oracle', 'psi'):
        q = S[(S.model == 'LGBM') & (S.fs == 'TXSR') & (S.policy == pol) & (S.d == d) & (S.cut == 34) & (S.dec == 'fix') & (S.tag == '')]
        if not q.empty: put(f'tc.{pol}.d{d}', f'{q.retrains_mean.values[0] * avg:.0f}')
q1 = S[(S.model == 'LGBM') & (S.fs == 'TXSR') & (S.policy == 'every') & (S.d == 1) & (S.cut == 34) & (S.dec == 'fix') & (S.tag == '')].retrains_mean.values[0]
for d in (1, 2, 3, 5):
    a = S[(S.model == 'LGBM') & (S.fs == 'TXSR') & (S.policy == 'every') & (S.d == d) & (S.cut == 34) & (S.dec == 'fix') & (S.tag == '')].retrains_mean.values[0]
    b = S[(S.model == 'LGBM') & (S.fs == 'TXSR') & (S.policy == 'dpm') & (S.d == d) & (S.cut == 34) & (S.dec == 'fix') & (S.tag == '')].retrains_mean.values[0]
    put(f'saving.d{d}', f'{100 * (1 - b / a):.0f}')
with open(os.path.join(ROOT, 'numbers2.tex'), 'w') as f:
    for k, v in D.items(): f.write('\\expandafter\\gdef\\csname num@%s\\endcsname{%s}\n' % (k, v))
print(len(D), 'macros')
