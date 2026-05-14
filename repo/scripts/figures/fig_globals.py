"""
Shared style constants and helpers for all 7 figures.
Nature/Genome-Biology visual grammar.
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib import gridspec, patheffects
from scipy.ndimage import gaussian_filter
from scipy.stats import mannwhitneyu
from scipy.spatial import Voronoi
from pathlib import Path

OUT = Path('/home/claude/figures_v2'); OUT.mkdir(exist_ok=True)

# ── Typography ─────────────────────────────────────────────────────────────
FONT  = 'DejaVu Sans'
FS    = dict(title=8.5, label=8, tick=7.5, annot=7, stat=8.5, panel=9.5)

matplotlib.rcParams.update({
    'font.family': FONT,
    'font.size': FS['label'],
    'axes.linewidth': 0.7,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': FS['label'],
    'axes.titlesize': FS['title'],
    'xtick.labelsize': FS['tick'],
    'ytick.labelsize': FS['tick'],
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'lines.linewidth': 1.2,
    'patch.linewidth': 0.6,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.facecolor': 'white',
})

# ── Semantic colors ────────────────────────────────────────────────────────
CR   = '#C1392B'    # responder warm red
CNR  = '#2166AC'    # non-responder cool blue
COrg = '#E67E22'    # organized / coexact (orange)
CPur = '#7D3C98'    # immune algebra (purple)
CGrn = '#1E8449'    # finite-range / local (dark green)
CGry = '#95A5A6'    # neutral / NR muted
CTM  = '#4A235A'    # TM backbone (deep purple)

CS = {'S0':'#AAAAAA','S1':'#3B6FA0','S2':'#5aae61',
      'S3':'#E67E22','S4':'#7D3C98','S?':'#cccccc'}

CMAP_C  = LinearSegmentedColormap.from_list('coex',['#FFF7EC','#FC8D59','#B30000'])
CMAP_D  = LinearSegmentedColormap.from_list('div', ['#2166AC','#F7F7F7','#C1392B'])
CMAP_PU = LinearSegmentedColormap.from_list('purp',['#F3E8FF','#9B59B6','#4A235A'])

# ── Panel letter helper ────────────────────────────────────────────────────
def panel_label(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=FS['panel'], fontweight='bold', va='top', ha='left')

# ── Statistical bracket ────────────────────────────────────────────────────
def stat_bracket(ax, x1, x2, y, p, h_frac=0.04, fontsize=8.5):
    h = abs(y) * h_frac + np.ptp(ax.get_ylim()) * 0.02
    t = y + h
    ax.plot([x1,x1,x2,x2],[y,t,t,y], lw=0.8, color='#333')
    s = '***' if p<.001 else('**' if p<.01 else('*' if p<.05 else 'n.s.'))
    c = '#222' if s!='n.s.' else '#888'
    ax.text((x1+x2)/2, t + np.ptp(ax.get_ylim())*0.01, s,
            ha='center', fontsize=fontsize, va='bottom', color=c)

# ── Swarm plot helper ──────────────────────────────────────────────────────
def swarm(ax, xp, vals, col, s=42, zorder=4):
    rng = np.random.RandomState(42)
    jx  = xp + rng.uniform(-0.15, 0.15, len(vals))
    ax.scatter(jx, vals, color=col, s=s, zorder=zorder, alpha=.9,
               edgecolors='white', linewidths=.35)
    ax.plot([xp-.23, xp+.23], [np.median(vals)]*2,
            color=col, lw=2.5, zorder=zorder+1, solid_capstyle='round')

# ── Spatial map with Gaussian overlay ─────────────────────────────────────
def spatial_map(ax, lcc=.35, p95=50, seed=42, title='',
                resp=True, n=320, add_contour=True, alpha_bg=0.06):
    rng = np.random.RandomState(seed)
    x   = rng.rand(n) * 100
    y   = rng.rand(n) * 100

    if resp and lcc > .25:
        cx, cy = 50 + rng.randn()*3, 50 + rng.randn()*3
        d  = np.exp(-((x-cx)**2 + (y-cy)**2) / (2*13**2))
        dens = np.clip(d + rng.randn(n)*.03, 0, None) * p95/75
    else:
        nh   = max(4, int(n*.04))
        dens = rng.rand(n) * .04
        hi   = rng.choice(n, nh, replace=False)
        dens[hi] += rng.rand(nh) * p95/60

    vm = max(np.percentile(dens, 95), .01)
    ax.scatter(x, y, c=dens, cmap=CMAP_C, s=7, vmin=0, vmax=vm,
               rasterized=True, edgecolors='none', zorder=2)

    if add_contour and resp and lcc > .25:
        # Gaussian density overlay
        grid_x, grid_y = np.mgrid[0:100:80j, 0:100:80j]
        from scipy.stats import gaussian_kde
        hi_pts = np.column_stack([x[dens > np.percentile(dens,70)],
                                   y[dens > np.percentile(dens,70)]])
        if len(hi_pts) > 5:
            kde  = gaussian_kde(hi_pts.T, bw_method=0.35)
            Z    = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(80,80)
            ax.contour(grid_x, grid_y, Z, levels=4,
                       colors=[COrg], linewidths=[0.6,0.8,1.0,1.2], alpha=0.7)

    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines['left'].set_visible(False); ax.spines['bottom'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=FS['title'], pad=4)
    return vm

# ── Data (inline) ─────────────────────────────────────────────────────────
D = pd.DataFrame({
    'resp':['R']*6 + ['NR']*9,
    'lcc': [.136,.263,.516,.100,.351,.329,.070,.200,.132,.340,.063,.167,.052,.129,.067],
    'ss':  [.088,.121,.249,.043,.156,.185,.036,.079,.055,.108,.027,.069,.025,.056,.027],
    'p95': [5.4,189,84,21.8,237.6,3.75,2.7,7.0,4.2,52.7,1.9,7.7,2.8,8.5,4.7],
    'lr2': [.464,.200,.489,.150,.552,.240,.110,.059,.248,-.097,.297,.054,-.008,.218,.255],
    'gr2': [-.102,-.144,.575,-1.112,-1.090,-.400,.041,.401,.081,.178,-1.202,.302,-.446,.091,-1.570],
    'kts': ['S4','S3','S3','S1','S3','S2','S1','S1','S1','S3','S1','S1','S1','S1','S1'],
    'sid': ['71p','71r','76r','83p','84p','84r','72p','72r','73p','73r','74p','74r','79p','85p','86p'],
})
D['gap'] = D['lr2'] - D['gr2']

NCG = pd.DataFrame({
    'resp':['R']*6 + ['NR']*9,
    'fIE': [.003,.002,.005,.001,.004,.002,.000,.000,.001,.001,.000,.000,.000,.001,.000],
    'fIM': [.054,.027,.054,.003,.046,.058,.017,.025,.054,.026,.062,.026,.021,.015,.038],
    'fEM': [.022,.024,.074,.002,.014,.026,.067,.028,.034,.021,.032,.030,.027,.005,.015],
    'fTM': [.834,.839,.720,.874,.820,.848,.856,.893,.831,.878,.842,.867,.892,.859,.833],
    'oe':  [.662,.602,.994,.508,.717,.659,.579,.474,.619,.436,.658,.562,.564,.520,.608],
    'sid': ['71p','71r','76r','83p','84p','84r','72p','72r','73p','73r','74p','74r','79p','85p','86p'],
})

import pandas as _pd
CABO_H = _pd.read_csv('/mnt/project/cabonivo_hodge_hotspots.csv')
CABO_N = _pd.read_csv('/mnt/project/cabonivo_ncg_commutators.csv')
CABO_S = _pd.read_csv('/mnt/project/cabonivo_step23_local_global.csv')
CABO_K = _pd.read_csv('/mnt/project/cabonivo_kts_states.csv')
print("Globals loaded OK")


# ── Spatial Hallmarks cancer-type colors ──────────────────────────────────
CANCER_COLS = {
    'Breast':        '#E74C3C',
    'Colorectal':    '#E67E22',
    'Glioblastoma':  '#8E44AD',
    'Lung':          '#2166AC',
    'Ovarian':       '#27AE60',
    'Prostate':      '#F39C12',
}

# ── Supplementary muted palette ───────────────────────────────────────────
CR_MUTED  = '#C67B76'
CNR_MUTED = '#7A9EBF'
