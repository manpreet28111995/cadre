import os, numpy as np, pandas as pd
from figstyle import *
from config import RES, FIG
from common import load_base

setup()
B = load_base(); t = B['t']; y = B['y']
steps = np.arange(1, 50)
lic = np.array([((t == s) & (y == 0)).sum() for s in steps]); ill = np.array([((t == s) & (y == 1)).sum() for s in steps])
prev = ill / np.maximum(lic + ill, 1)
au = pd.read_csv(os.path.join(RES, 'audit_future_by_step.csv'))
pi = pd.read_csv(os.path.join(RES, 'protocol_inflation.csv'))

fig, axs = plt.subplots(2, 2, figsize=(W2, 4.0), gridspec_kw=dict(hspace=0.5, wspace=0.28))
def spans(ax):
    ax.axvspan(0.5, 29.5, color='#EEF3F8', zorder=0, lw=0); ax.axvspan(29.5, 34.5, color='#E8F4EC', zorder=0, lw=0); ax.axvspan(34.5, 49.5, color='#F7F2E8', zorder=0, lw=0)
    for x, s_ in ((15, 'train'), (32, 'val.'), (42, 'test')): ax.text(x, 1.01, s_, transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=6.8, color=PAL['gray'])
ax = axs[0, 0]
spans(ax)
ax.bar(steps, lic, color=PAL['sky'], width=0.85, label='Licit', zorder=2)
ax.bar(steps, ill, bottom=lic, color=PAL['orange'], width=0.85, label='Illicit', zorder=2)
ax.set_xlim(0.5, 49.5); ax.set_ylim(0, 2500); ax.set_xlabel('Time step'); ax.set_ylabel('Labelled transactions'); ax.grid(axis='x', visible=False)
ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.98), ncol=2, handlelength=1.0, columnspacing=1.0)
tag(ax, '(a)', dx=-0.2, dy=1.08)
ax = axs[0, 1]
spans(ax)
ax.plot(steps, 100 * prev, color=PAL['navy'], lw=1.2, marker='o', ms=2.4, zorder=3)
ax.axvline(43, color=PAL['red'], lw=0.9, ls=(0, (3, 2)), zorder=4)
ax.annotate('step 43', xy=(43, 36), xytext=(38.2, 36), fontsize=6.8, color=PAL['red'], va='center', ha='right', arrowprops=dict(arrowstyle='-', color=PAL['red'], lw=0.6))
ax.set_xlim(0.5, 49.5); ax.set_ylim(0, 40); ax.set_xlabel('Time step'); ax.set_ylabel('Illicit share of labelled (%)'); ax.grid(axis='x', visible=False)
tag(ax, '(b)', dx=-0.2, dy=1.08)
ax = axs[1, 0]
spans(ax)
ax.bar(au['Time_step'], 100 * au['share_future_block'], color=PAL['red'], width=0.85, zorder=2)
ax.set_xlabel('Time step'); ax.set_ylabel('Wallet rows with later-step\ninformation (%)'); ax.set_xlim(0.5, 49.5); ax.set_ylim(0, 34); ax.grid(axis='x', visible=False)
tag(ax, '(c)', dx=-0.2, dy=1.08)
ax = axs[1, 1]
order = [('random', 'TX'), ('chronological', 'TX'), ('random', 'TX+W'), ('chronological', 'TX+W')]
pos = [0, 1, 2.6, 3.6]; colr = [PAL['sky'], PAL['blue'], PAL['pink'], '#8C3F73']
for (pr, fs), x, c in zip(order, pos, colr):
    d = pi[(pi.protocol == pr) & (pi.features == fs)]['f1']
    ax.bar(x, d.mean(), yerr=d.std(ddof=1), color=c, width=0.85, capsize=2, error_kw=dict(lw=0.8), zorder=2)
    ax.text(x, d.mean() + d.std(ddof=1) + 0.012, f'{d.mean():.3f}', ha='center', va='bottom', fontsize=6.8)
ax.set_xticks(pos); ax.set_xticklabels(['Random', 'Chrono-\nlogical', 'Random', 'Chrono-\nlogical'])
ax.text(0.5, -0.36, 'Transaction features', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.2)
ax.text(3.1, -0.36, '+ wallet features', transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.2)
ax.set_ylim(0.5, 1.08); ax.set_xlim(-0.7, 4.3); ax.set_ylabel('Illicit-class F1 (10 seeds)'); ax.grid(axis='x', visible=False)
tag(ax, '(d)', dx=-0.2, dy=1.08)
fig.savefig(os.path.join(FIG, 'fig2_data_audit.png'), metadata={'Software': None})
print('ok')
