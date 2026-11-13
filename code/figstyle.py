import matplotlib as mpl
import matplotlib.pyplot as plt

PAL = dict(blue='#0072B2', orange='#D55E00', green='#009E73', pink='#CC79A7', sky='#56B4E9', yellow='#E69F00', gray='#7A7A7A', black='#222222', red='#B2182B', navy='#1B3A5C')
W1 = 3.35
W2 = 6.95

def setup():
    mpl.rcParams.update({
        'font.family': 'Liberation Sans', 'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8.2, 'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
        'legend.fontsize': 6.9, 'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'lines.linewidth': 1.4, 'lines.markersize': 3.6, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'savefig.dpi': 450,
        'axes.grid': True, 'grid.color': '#E6E6E6', 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'legend.frameon': False, 'pdf.fonttype': 42,
        'axes.titlelocation': 'left', 'axes.titlepad': 4, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03, 'mathtext.default': 'regular'})

def tag(ax, s, dx=-0.0, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.6, fontweight='bold', va='bottom', ha='left')

def shade(ax, x0=34.5, x1=42.5, x2=49.5):
    ax.axvspan(x0, x1, color='#F4F4F4', zorder=0, lw=0)
    ax.axvspan(x1, x2, color='#FBE9DF', zorder=0, lw=0)
    ax.axvline(x1, color=PAL['orange'], lw=0.8, ls=(0, (3, 2)), zorder=1)

def ptitle(ax, tg, title, dy=1.03, off=16):
    ax.text(0, dy, tg, transform=ax.transAxes, fontsize=8.6, fontweight='bold', va='bottom', ha='left')
    ax.annotate(title, xy=(0, dy), xycoords='axes fraction', xytext=(off, 0), textcoords='offset points', fontsize=8, va='bottom', ha='left')
