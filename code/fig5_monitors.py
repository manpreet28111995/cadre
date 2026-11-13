import os, numpy as np, pandas as pd
from matplotlib.lines import Line2D
from figstyle import *
from config import RES, FIG

setup()
T = pd.read_csv(os.path.join(RES, 'stream_traces.csv'))
d1 = T[T.config == 'LGBM_TXSR_dpm_d1_c34'].sort_values('step')
st = T[T.config == 'LGBM_TXSR_static_d1_c34'].sort_values('step')
dg = T[T.config == 'LGBM_TXSR_static_d1_c34_diag'].sort_values('step')
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.45), gridspec_kw=dict(wspace=0.38))
C = PAL['navy']; R = PAL['red']
ax = axs[0]; shade(ax)
v = d1.dropna(subset=['dpm'])
ax.plot(v.step, v.dpm, color=C, marker='o', ms=3)
ax.step(v.step, 0.5 * v.dpm_ref, where='mid', color=R, lw=1.1, ls=(0, (3, 2)))
al = d1[d1.alarm > 0].step.values
ax.scatter(v[v.step.isin(al)].step, v[v.step.isin(al)].dpm, s=40, facecolor='none', edgecolor=R, lw=1.2, zorder=5)
ax.set_xlim(34.5, 49.5); ax.set_ylim(-0.03, 1.05); ax.set_xlabel('Revealed step'); ax.set_ylabel('F1 of issued alerts')
ptitle(ax, '(a)', 'Delayed-performance')
ax = axs[1]; shade(ax)
ax.plot(st.step, st.psi, color=C, marker='o', ms=3)
ax.plot(st.step, st.th_psi, color=R, lw=1.1, ls=(0, (3, 2)))
ax.set_xlim(34.5, 49.5); ax.set_ylim(0, 0.4); ax.set_xlabel('Time step'); ax.set_ylabel('Score PSI')
ptitle(ax, '(b)', 'Score (label-free)')
ax = axs[2]; shade(ax)
ax.plot(dg.step, dg.c2, color=C, marker='o', ms=3)
ax.plot(dg.step, dg.th_c2, color=R, lw=1.1, ls=(0, (3, 2)))
ax.set_xlim(34.5, 49.5); ax.set_ylim(0.96, 1.004); ax.set_xlabel('Time step'); ax.set_ylabel('Domain-classifier AUC')
ptitle(ax, '(c)', 'Feature (label-free)')
for a in axs: a.set_xticks(range(35, 50, 4))
hs = [Line2D([0], [0], color=C, marker='o', ms=3), Line2D([0], [0], color=R, ls=(0, (3, 2)), lw=1.1), Line2D([0], [0], marker='o', ls='', mfc='none', mec=R, ms=6, mew=1.2), Line2D([0], [0], color=PAL['orange'], ls=(0, (3, 2)), lw=0.8)]
fig.legend(hs, ['Monitored statistic', 'Alarm level', 'Alarm raised', 'Shift at step 43'], loc='lower center', bbox_to_anchor=(0.5, -0.17), ncol=4, handlelength=1.8, columnspacing=1.6)
fig.savefig(os.path.join(FIG, 'fig5_monitors.png'), metadata={'Software': None})
print('ok')
