import os, numpy as np, pandas as pd
from matplotlib.lines import Line2D
from figstyle import *
from config import RES, FIG

setup()
S = pd.read_csv(os.path.join(RES, 'stream_per_step_per_seed.csv'))
S = S[(S.cut == 34) & (S.model == 'LGBM') & (S.fs == 'TXSR') & (S.tag.fillna('') == '')]
T = pd.read_csv(os.path.join(RES, 'stream_traces.csv'))
T = T[T.tag.fillna('') == '']
STY = dict(static=('Static (no retraining)', PAL['gray'], '-', 1.3, 2), every=('Retrain at every step', PAL['blue'], '-', 1.4, 3),
           slide=('Sliding window ($W$ = 10)', PAL['green'], '-', 1.4, 3), dpm=('CADRE (delayed-performance trigger)', PAL['orange'], '-', 2.0, 5))
fig, axs = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True, gridspec_kw=dict(wspace=0.07))
for ax, d, tg in ((axs[0], 1, '(a)'), (axs[1], 5, '(b)')):
    shade(ax)
    for pol, (lab, c, ls, lw, z) in STY.items():
        q = S[(S.policy == pol) & (S.d == d) & (S.dec == 'fix')]
        g = q.pivot_table(index='step', columns='seed', values='f1')
        m, s = g.mean(axis=1), g.std(axis=1, ddof=1)
        ax.plot(g.index, m, color=c, ls=ls, lw=lw, zorder=z, marker='o' if pol == 'dpm' else None, ms=2.6)
        if pol in ('dpm', 'static'): ax.fill_between(g.index, m - s, m + s, color=c, alpha=0.18, lw=0, zorder=z - 1)
    al = T[(T.config == f'LGBM_TXSR_dpm_d{d}_c34') & (T.alarm > 0)].step.values
    for a in al: ax.plot([a], [-0.045], marker='^', color=PAL['red'], ms=4.6, clip_on=False, zorder=6)
    ax.set_xlim(34.5, 49.5); ax.set_ylim(-0.02, 1.03); ax.set_xlabel('Time step')
    ax.text(38.5, 1.0, 'before the shift', fontsize=6.8, color=PAL['gray'], ha='center', va='top'); ax.text(46.0, 1.0, 'after step 43', fontsize=6.8, color=PAL['orange'], ha='center', va='top')
    ax.set_xticks(range(35, 50, 2)); ptitle(ax, tg, f'Label delay $d$ = {d}')
axs[0].set_ylabel('Illicit-class F1')
hs = [Line2D([0], [0], color=c, ls=ls, lw=lw, marker='o' if p == 'dpm' else None, ms=2.6) for p, (lab, c, ls, lw, z) in STY.items()]
ls_ = [v[0] for v in STY.values()]
hs.append(Line2D([0], [0], color=PAL['red'], marker='^', ls='', ms=4.6)); ls_.append('CADRE alarm raised')
fig.legend(hs, ls_, loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=3, handlelength=1.8, columnspacing=1.4)
fig.savefig(os.path.join(FIG, 'fig3_per_step.png'), metadata={'Software': None})
print('ok')
