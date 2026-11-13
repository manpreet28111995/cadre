import os, numpy as np, pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve, average_precision_score
from figstyle import *
from config import RES, FIG

setup()
D = os.path.join(RES, 'stream')
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.75), gridspec_kw=dict(wspace=0.5, width_ratios=[1, 1.15]))
ax = axs[0]
for pol, lab, c in (('static', 'Static', PAL['gray']), ('every', 'Every step', PAL['blue']), ('dpm', 'CADRE', PAL['orange'])):
    z = np.load(os.path.join(D, f'LGBM_TXSR_{pol}_d1_c34.npz')); y = z['y']; late = z['te_step'] >= 43
    p = z['p'][:, late].ravel(); yy = np.tile(y[late], z['p'].shape[0])
    pr, rc, _ = precision_recall_curve(yy, p)
    ax.plot(rc, pr, color=c, lw=1.5, label=f'{lab} (AP {average_precision_score(yy, p):.2f})')
base = z['y'][late].mean()
ax.axhline(base, color=PAL['gray'], lw=0.8, ls=':', label=f'Base rate ({100 * base:.1f}%)')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.legend(loc='upper right', bbox_to_anchor=(1.0, 0.99), handlelength=1.6)
ptitle(ax, '(a)', 'PR curves, steps 43-49 ($d$ = 1)', off=16)
NM = {'R_all_ill_frac': 'R: illicit address share', 'S_all_n': 'S: distinct addresses', 'num_output_addresses': 'Output addresses', 'size': 'Transaction size'}
def nice(f):
    if f in NM: return NM[f]
    if f.startswith('Local_feature_'): return 'Local feature ' + f.split('_')[-1]
    if f.startswith('Aggregate_feature_'): return 'Aggregated feature ' + f.split('_')[-1]
    return f.replace('_', ' ')
H = pd.read_csv(os.path.join(RES, 'shap_importance.csv')).sort_values('static_all', ascending=False).head(10).iloc[::-1]
CG = {'T': PAL['blue'], 'S': PAL['green'], 'R': PAL['orange']}
ax = axs[1]; y = np.arange(len(H))
ax.barh(y + 0.19, H.static_early, height=0.36, color=[CG[g] for g in H.group], zorder=2)
ax.barh(y - 0.19, H.static_late, height=0.36, color=[CG[g] for g in H.group], alpha=0.45, zorder=2)
ax.set_yticks(y); ax.set_yticklabels([nice(f) for f in H.feature]); ax.set_xlabel('Mean |SHAP value| (log-odds)'); ax.grid(axis='y', visible=False)
ax.legend(handles=[Patch(color=CG['T'], label='Transaction (T)'), Patch(color=CG['S'], label='Structural (S)'), Patch(color=CG['R'], label='Reputation (R)'),
                   Patch(color='#555555', label='Steps 35-42'), Patch(color='#555555', alpha=0.45, label='Steps 43-49')], loc='lower right', handlelength=1.1)
ptitle(ax, '(b)', 'Static model, top-10 features', off=16)
fig.savefig(os.path.join(FIG, 'fig6_curves.png'), metadata={'Software': None})
print('ok')
