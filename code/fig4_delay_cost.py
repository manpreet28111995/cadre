import os, numpy as np, pandas as pd
from figstyle import *
from config import RES, FIG

setup()
S = pd.read_csv(os.path.join(RES, 'stream_summary.csv')); S['tag'] = S['tag'].fillna('')
S = S[(S.model == 'LGBM') & (S.fs == 'TXSR') & (S.cut == 34) & (S.dec == 'fix') & (S.tag == '')]
order = [('static', 'Static'), ('every', 'Every step'), ('slide', 'Sliding window'), ('per3', 'Every 3 steps'), ('per5', 'Every 5 steps'), ('psi', 'Score-PSI trigger'), ('oracle', 'Oracle-onset trigger'), ('dpm', 'CADRE')]
ds = [1, 2, 3, 5]
def mat(col): return np.array([[S[(S.policy == p) & (S.d == d)][col].values[0] for d in ds] for p, _ in order])
M1, M2 = mat('f1_late_mean'), mat('retrains_mean')
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.9), gridspec_kw=dict(wspace=0.22, width_ratios=[1, 1]))
for ax, M, cmap, vmax, fmt, lab, tg, ttl in ((axs[0], M1, 'YlOrBr', 0.45, '{:.2f}', 'F1, steps 43-49', '(a)', 'Late-period F1'), (axs[1], M2, 'Blues', 17, '{:.0f}', 'Retrainings', '(b)', 'Retrainings over 15 test steps')):
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=vmax, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=7.2, color='white' if v / vmax > 0.62 else '#222222')
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds); ax.set_xlabel('Label delay $d$ (steps)')
    ax.set_yticks(range(len(order))); ax.set_yticklabels([n for _, n in order] if ax is axs[0] else [])
    ax.grid(False); ax.spines[:].set_visible(False); ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(ds), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which='minor', color='white', lw=1.2); ax.tick_params(which='minor', length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.outline.set_visible(False); cb.ax.tick_params(length=2)
    ptitle(ax, tg, ttl, off=16)
axs[0].get_yticklabels()[-1].set_fontweight('bold')
fig.savefig(os.path.join(FIG, 'fig4_delay_cost.png'), metadata={'Software': None})
print('ok')
