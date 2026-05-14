#!/usr/bin/env python3
"""build_figures.py — Main manuscript figures 1–7
Run: python build_figures.py [--outdir figures/] [--fig 1,2,3]
"""

import argparse, sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Arc
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import Voronoi
from scipy.stats import mannwhitneyu
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from fig_globals import (
    OUT, FS, CR, CNR, COrg, CPur, CGrn, CGry, CTM, CS,
    CMAP_C, CMAP_D, CMAP_PU, CANCER_COLS,
    swarm, stat_bracket, panel_label, spatial_map,
    D, NCG, CABO_H, CABO_N, CABO_S, CABO_K,
    mannwhitneyu,
)

_Path = Path

# ─────────────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Publication-grade conceptual framework
# ═══════════════════════════════════════════════════════════════════════════
def build_fig1():
    fig = plt.figure(figsize=(17, 9.5))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.36,
                           left=0.04, right=0.97, top=0.93, bottom=0.06)

    # ── A  Tissue overview ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0,0])
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_aspect('equal'); ax.axis('off')

    # Smooth tumour region
    theta = np.linspace(0, 2*np.pi, 300)
    rbase = 2.8 + 0.2*np.sin(6*theta) + 0.12*np.cos(11*theta)
    tx = 3.8 + rbase*np.cos(theta); ty = 5.0 + rbase*np.sin(theta)
    ax.fill(tx, ty, color='#FADBD8', alpha=0.9, zorder=1)
    ax.plot(tx, ty, color='#E74C3C', lw=1.2, zorder=2)
    ax.text(3.8, 5.0, 'Tumour', ha='center', va='center',
            fontsize=FS['label'], color='#922B21', fontweight='bold')

    # Interface annulus — dashed arc
    ri = 3.6 + 0.2*np.sin(4*theta)
    ix = 3.8 + ri*np.cos(theta); iy = 5.0 + ri*np.sin(theta)
    ax.plot(ix, iy, color=COrg, lw=1.4, ls='--', zorder=3, alpha=0.85)
    ax.text(3.8, 1.05, 'interface', ha='center', fontsize=FS['annot'],
            color=COrg, style='italic')

    # Immune cells — cleaner dots
    rng = np.random.RandomState(10)
    for _ in range(26):
        a = rng.uniform(0, 2*np.pi); r = rng.uniform(4.0, 5.0)
        ax.add_patch(mpatches.Circle((3.8+r*np.cos(a), 5.0+r*np.sin(a)),
                                      0.17, fc=CNR, ec='white', lw=0.3, zorder=4))
    ax.text(8.3, 5.0, 'Immune\nregion', ha='center', va='center',
            fontsize=FS['label'], color='#154360', fontweight='bold')
    panel_label(ax,'A')

    # ── B  Abundance failure ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0,1]); ax.axis('off')
    ax.set_xlim(0,8); ax.set_ylim(-0.3,5.5)
    panel_label(ax,'B')
    ax.set_title('Same abundance, different structure', fontsize=FS['title'], pad=5)

    rng2 = np.random.RandomState(99)
    # Left: gradient (all T-cells on the right, tumor on left)
    for i in range(6):
        for j in range(4):
            col = '#C0392B' if i < 2 else ('#2166AC' if i > 3 else '#BDC3C7')
            ax.add_patch(mpatches.Circle((i*.75+.3, j*.9+.5), .25,
                                          fc=col, ec='white', lw=.3, alpha=.85))
    ax.text(2.2, -0.18, 'Gradient', ha='center', fontsize=FS['annot'],
            color='#555', style='italic')

    # Right: coexact (alternating mosaic — same count)
    cols_right = ['#C0392B','#2166AC']*6
    rng3=np.random.RandomState(7)
    for i,(col) in enumerate(cols_right):
        xi = rng3.uniform(4.4, 7.6); yi = rng3.uniform(.4, 4.4)
        ax.add_patch(mpatches.Circle((xi, yi), .25,
                                      fc=col, ec='white', lw=.3, alpha=.85))
    ax.text(6.0, -0.18, 'Non-gradient', ha='center', fontsize=FS['annot'],
            color='#555', style='italic')

    # Arrow between
    ax.annotate('', xy=(3.9, 2.3), xytext=(3.2, 2.3),
                arrowprops=dict(arrowstyle='->',color='#888',lw=1.0))
    ax.text(3.55, 2.65, 'abundance\nidentical', ha='center', fontsize=6.5, color='#888')

    # ── C  Wedge field ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0,2]); ax.set_aspect('equal'); ax.axis('off')
    ax.set_xlim(-0.3, 5.3); ax.set_ylim(-0.4, 5.0)
    panel_label(ax,'C')
    ax.set_title('Antisymmetric wedge field', fontsize=FS['title'], pad=5)

    rng4 = np.random.RandomState(7)
    pts_t = [(rng4.uniform(.2,1.9), rng4.uniform(.4,4.6)) for _ in range(6)]
    pts_i = [(rng4.uniform(2.9,4.7), rng4.uniform(.4,4.6)) for _ in range(6)]
    for xp,yp in pts_t:
        ax.add_patch(mpatches.Circle((xp,yp),.23,fc='#C0392B',ec='white',lw=.3,zorder=3))
    for xp,yp in pts_i:
        ax.add_patch(mpatches.Circle((xp,yp),.23,fc='#2166AC',ec='white',lw=.3,zorder=3))

    for k in range(3):
        xt,yt = pts_t[k]; xi,yi = pts_i[k]
        mx,my = (xt+xi)/2, (yt+yi)/2
        ax.annotate('',xy=(mx+.15,my),xytext=(xt+.25,yt),
                    arrowprops=dict(arrowstyle='-|>',color=COrg,lw=1.2,
                                   connectionstyle='arc3,rad=0.18'))
        ax.annotate('',xy=(xi-.15,yi),xytext=(mx+.15,my),
                    arrowprops=dict(arrowstyle='-|>',color=CPur,lw=1.2,
                                   connectionstyle='arc3,rad=0.18'))

    ax.text(2.45,-0.28,r'$\omega(u,v)=A(u)B(v)-A(v)B(u)$',
            ha='center',fontsize=6.8,color='#333',family='monospace')

    # ── D  Hodge decomposition ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1,0]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,5.5)
    panel_label(ax,'D')
    ax.set_title('Hodge decomposition', fontsize=FS['title'], pad=5)

    # Exact arrows (gradient)
    for xi in [0.4,1.5,2.6,3.7]:
        ax.annotate('',xy=(xi+.85,4.0),xytext=(xi+.1,4.0),
                    arrowprops=dict(arrowstyle='-|>',color=CGrn,lw=1.8,
                                   mutation_scale=8))
    ax.text(2.5, 3.2, 'Exact  (gradient)', ha='center',
            fontsize=FS['label'], color=CGrn, fontweight='bold')

    # Coexact arc
    from matplotlib.patches import Arc
    ax.add_patch(Arc((7.2,3.2),2.4,2.2,angle=0,theta1=5,theta2=268,
                      color=COrg,lw=2.0))
    ax.annotate('',xy=(7.2,4.33),xytext=(8.05,4.31),
                arrowprops=dict(arrowstyle='-|>',color=COrg,lw=2.0,mutation_scale=9))
    ax.text(7.2, 1.3, 'Coexact  (non-gradient)', ha='center',
            fontsize=FS['label'], color=COrg, fontweight='bold')

    # ── E  Two regimes ────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1,1]); ax.axis('off')
    ax.set_xlim(-0.1, 5); ax.set_ylim(-0.2, 5.8)
    panel_label(ax,'E')
    ax.set_title('Two interface regimes', fontsize=FS['title'], pad=5)

    # Responder — smooth Gaussian cluster
    rng5 = np.random.RandomState(3)
    pts_r = np.column_stack([rng5.randn(18)*.8+2.3, rng5.randn(18)*.3+4.5])
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((0,3.5),4.6,2.0,
                                boxstyle='round,pad=.1',fc='#FEF5E7',ec=COrg,lw=1.3))
    for pt in pts_r:
        ax.add_patch(mpatches.Circle(pt,.12,fc=COrg,ec='none',alpha=.85,zorder=3))
    ax.text(2.3,3.62,'Responder  —  connected front',
            ha='center',fontsize=FS['annot'],color='#7E5109',fontweight='bold')

    # NR — scattered
    pts_nr = [(rng5.uniform(.3,4.5), rng5.uniform(.6,2.2)) for _ in range(8)]
    ax.add_patch(FancyBboxPatch((0,.3),4.6,2.0,
                                boxstyle='round,pad=.1',fc='#EAF0FB',ec=CNR,lw=1.3))
    for pt in pts_nr:
        ax.add_patch(mpatches.Circle(pt,.12,fc=CNR,ec='none',alpha=.85,zorder=3))
    ax.text(2.3,.42,'Non-responder  —  fragmented',
            ha='center',fontsize=FS['annot'],color='#1A5276',fontweight='bold')

    # ── F  Pipeline (left-to-right with icons) ───────────────────────────
    ax = fig.add_subplot(gs[1,2]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,5.5)
    panel_label(ax,'F')
    ax.set_title('Analytical layers', fontsize=FS['title'], pad=5)

    steps = [
        ('Hodge',     'Fronts vs\nfragments',   COrg,  1.0),
        ('NCG',       'Immune–immune\nalgebra',  CPur,  2.5),
        ('KTS',       'Regime\ntransitions',     CR,    4.0),
        ('Step 23',   'Local\nconstraint',       CGrn,  5.5),
        ('CABO/NIVO', 'Replication',             CNR,   7.0),
        ('IMC',       'Scale\nemergence',        CGry,  8.5),
    ]
    for name, detail, col, xc in steps:
        # Hexagon-like marker
        hex_th = np.linspace(0, 2*np.pi, 7)
        hx = xc + .5*np.cos(hex_th); hy = 3.8 + .5*np.sin(hex_th)
        ax.fill(hx, hy, color=col, alpha=.85, zorder=3)
        ax.text(xc, 3.8, name[:3], ha='center', va='center',
                fontsize=6.5, color='white', fontweight='bold', zorder=4)
        ax.text(xc, 2.9, name, ha='center', fontsize=7, color=col, fontweight='bold')
        ax.text(xc, 2.2, detail, ha='center', fontsize=6.3, color='#555',
                va='top', multialignment='center')
        if xc < 8.5:
            ax.annotate('', xy=(xc+.8, 3.8), xytext=(xc+.55, 3.8),
                       arrowprops=dict(arrowstyle='->', color='#AAA', lw=1.0))

    plt.suptitle(
        'Figure 1  |  Operator-regime framework for tumour–immune interface analysis',
        fontsize=10, fontweight='bold', y=0.97)
    plt.savefig(OUT/'fig1_conceptual.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(); print('Fig1 ✓')

build_fig1()

# ═══ FIGURE 2 — Hodge geometry ════════════════════════════════════════════
def build_fig2():
    fig = plt.figure(figsize=(14,9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2,3,hspace=0.46,wspace=0.36,
                           left=0.06,right=0.97,top=0.92,bottom=0.08)

    # A — responder spatial map + contours
    ax = fig.add_subplot(gs[0,0])
    vm = spatial_map(ax, lcc=.351, p95=237, seed=7, resp=True, n=340,
                     add_contour=True,
                     title='A   Responder  (HCC84, S3)\n')
    cb = plt.colorbar(ax.collections[0], ax=ax, fraction=.05, pad=.02)
    cb.set_label('Coexact density', fontsize=FS['annot'])
    cb.ax.tick_params(labelsize=FS['annot'])
    panel_label(ax,'A')

    # B — NR spatial map (muted grey background hint)
    ax = fig.add_subplot(gs[0,1])
    ax.set_facecolor('#FAFAFA')
    vm2 = spatial_map(ax, lcc=.07, p95=2.7, seed=9, resp=False, n=340,
                      add_contour=False,
                      title='B   Non-responder  (HCC72, S1)\n')
    cb2 = plt.colorbar(ax.collections[0], ax=ax, fraction=.05, pad=.02)
    cb2.set_label('Coexact density', fontsize=FS['annot'])
    cb2.ax.tick_params(labelsize=FS['annot'])
    panel_label(ax,'B')

    # C — LCC comparison
    ax = fig.add_subplot(gs[0,2])
    rv  = D[D['resp']=='R']['lcc'].values
    nrv = D[D['resp']=='NR']['lcc'].values
    swarm(ax,0,rv,CR); swarm(ax,1,nrv,CNR)
    _,p = mannwhitneyu(rv,nrv,alternative='two-sided')
    ax.set_xlim(-.6,1.6); ax.set_ylim(-.02,.65)
    stat_bracket(ax,0,1,rv.max()+.05,p)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Responder\n(n=6)','Non-resp.\n(n=9)'], fontsize=FS['tick'])
    ax.set_ylabel('LCC fraction', fontsize=FS['label'])
    ax.set_title('C   Hotspot connectivity', fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'C')

    # D — Spatial score
    ax = fig.add_subplot(gs[1,0])
    rv  = D[D['resp']=='R']['ss'].values
    nrv = D[D['resp']=='NR']['ss'].values
    swarm(ax,0,rv,CR); swarm(ax,1,nrv,CNR)
    _,p = mannwhitneyu(rv,nrv,alternative='two-sided')
    ax.set_xlim(-.6,1.6); ax.set_ylim(-.01,.30)
    stat_bracket(ax,0,1,rv.max()+.012,p)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Responder','Non-resp.'], fontsize=FS['tick'])
    ax.set_ylabel('Spatial score (LCC×Gini)', fontsize=FS['label'])
    ax.set_title('D   Spatial score   p = 0.006', fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'D')

    # E — Elegant trajectories
    ax = fig.add_subplot(gs[1,1])
    pairs=[(0.153,0.231,'R'),(0.169,None,'R'),(0.159,0.097,'R'),
           (0.121,0.058,'R'),(0.066,0.050,'NR'),(0.071,0.070,'NR')]
    for i,(pre,post,resp) in enumerate(pairs):
        if post is None: continue
        col = CR if resp=='R' else CNR
        jit = (i-2.5)*.018
        ax.plot([0+jit,1+jit],[pre,post],color=col,lw=1.5,alpha=.82,
                marker='o',ms=4.5,zorder=3,
                markeredgecolor='white',markeredgewidth=.3)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Pre-therapy','Post-therapy'], fontsize=FS['tick'])
    ax.set_ylabel('Coexact fraction', fontsize=FS['label'])
    ax.set_title('E   Pre→post trajectories', fontsize=FS['title'])
    for c,l in [(CR,'Responder'),(CNR,'Non-resp.')]:
        ax.plot([],[],color=c,lw=1.4,ms=4.5,marker='o',label=l)
    ax.legend(handlelength=1.5, handletextpad=0.4)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'E')

    # F — Negative control (muted styling)
    ax = fig.add_subplot(gs[1,2])
    ax.set_facecolor('#FAFAFA')
    ws_r  = [.153,.119,.136,.124]
    ws_nr = [.072,.088,.105,.111,.094,.101]
    swarm(ax,0,ws_r,'#C67B76',s=35)    # desaturated R
    swarm(ax,1,ws_nr,'#7A9EBF',s=35)  # desaturated NR
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Responder','Non-resp.'], fontsize=FS['tick'])
    ax.set_ylabel('Coexact fraction\n(whole section)', fontsize=FS['label'])
    ax.set_title('F   Whole-section  (p = 0.93)\nnegative control', fontsize=FS['title'])
    ax.text(.5,-.18,'interface restriction required',ha='center',
            fontsize=6.5,color='#888',style='italic',transform=ax.transAxes)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'F')

    plt.suptitle(
        'Figure 2  |  Responder interfaces form connected coexact fronts',
        fontsize=10, fontweight='bold', y=0.96)
    plt.savefig(OUT/'fig2_hodge_geometry.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(); print('Fig2 ✓')

build_fig2()

# ═══ FIGURE 3 — NCG immune algebra ════════════════════════════════════════
def build_fig3():
    from scipy.spatial import Voronoi, voronoi_plot_2d
    from matplotlib.collections import PolyCollection

    fig = plt.figure(figsize=(14,9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2,3,hspace=0.48,wspace=0.40,
                           left=0.07,right=0.97,top=0.92,bottom=0.08)

    rN  = NCG[NCG['resp']=='R']
    nrN = NCG[NCG['resp']=='NR']

    # A — Commutator fraction heatmap
    ax = fig.add_subplot(gs[0,0])
    keys=['fTM','fIM','fEM','fIE']; xlbls=['TM','IM','EM','IE']
    mat = np.array([[rN[k].median() for k in keys],
                    [nrN[k].median() for k in keys]])
    im  = ax.imshow(mat, aspect='auto', cmap='YlOrRd', vmin=0, vmax=.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(xlbls, fontsize=FS['tick'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Responder','Non-resp.'],
                                              fontsize=FS['tick'])
    plt.colorbar(im, ax=ax, fraction=.05, pad=.02).ax.tick_params(labelsize=6)
    for i in range(2):
        for j in range(4):
            v = mat[i,j]
            ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                    fontsize=7.5, color='white' if v>.5 else '#333',
                    fontweight='bold')
    ax.set_title('A   Commutator fractions\nTM universal; IE/IM/EM response-specific',
                 fontsize=FS['title'])
    panel_label(ax,'A')

    # B — TM backbone (horizontal bars)
    ax = fig.add_subplot(gs[0,1])
    ids = list(range(len(NCG)))
    ax.barh(ids, NCG['fTM'], color=CTM, height=.7, label='TM backbone',
            edgecolor='none')
    ax.barh(ids, 1-NCG['fTM'], left=NCG['fTM'], color='#D7BDE2',
            height=.7, label='Other pairs', edgecolor='none')
    ax.set_yticks(ids)
    ax.set_yticklabels([f"{'R' if r=='R' else 'NR'} {s}"
                        for r,s in zip(NCG['resp'],NCG['sid'])],
                       fontsize=6.2)
    ax.set_xlabel('Fraction of commutator energy', fontsize=FS['label'])
    ax.axvline(.85, color='#666', lw=.8, ls='--', alpha=.6)
    ax.set_title('B   TM backbone universal', fontsize=FS['title'])
    ax.legend(fontsize=6.5, loc='lower right')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'B')

    # C — IE/IM/EM enrichment (clean)
    ax = fig.add_subplot(gs[0,2])
    pairs_list = [('fIE','IE','#C0392B'),('fIM','IM',CPur),('fEM','EM','#7D3C98')]
    for xi,(key,lbl,col) in enumerate(pairs_list):
        rv2  = rN[key].values; nrv2 = nrN[key].values
        _,p  = mannwhitneyu(rv2,nrv2,alternative='two-sided')
        swarm(ax,xi-.17,rv2,col,s=42)
        swarm(ax,xi+.17,nrv2,CGry,s=42)
        ax.set_ylim(-0.002,.012)
        stat_bracket(ax,xi-.17,xi+.17,max(rv2.max(),nrv2.max())+.0005,p,h_frac=.12)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(['IE','IM','EM'],fontsize=FS['tick'])
    ax.set_ylabel('Commutator fraction', fontsize=FS['label'])
    ax.set_title('C   Immune–immune enrichment\nin responders', fontsize=FS['title'])
    for c,l in [(CR,'Responder'),(CGry,'Non-resp.')]:
        ax.plot([],[],color=c,label=l,marker='o',lw=0,ms=5)
    ax.legend(handletextpad=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'C')

    # D — Operator entropy
    ax = fig.add_subplot(gs[1,0])
    rv  = rN['oe'].values; nrv = nrN['oe'].values
    swarm(ax,0,rv,CR); swarm(ax,1,nrv,CNR)
    _,p = mannwhitneyu(rv,nrv,alternative='two-sided')
    ax.set_xlim(-.6,1.6); ax.set_ylim(.40,1.10)
    stat_bracket(ax,0,1,rv.max()+.015,p)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Responder','Non-resp.'], fontsize=FS['tick'])
    ax.set_ylabel('Operator entropy', fontsize=FS['label'])
    ax.set_title('D   Algebraic complexity   p = 0.050', fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'D')

    # E — Voronoi spatial mosaic (responder heterogeneous vs NR homogeneous)
    ax = fig.add_subplot(gs[1,1]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(-0.3,5.5)
    ax.set_title('E   Spatial immune programme mosaic', fontsize=FS['title'])
    panel_label(ax,'E')

    def draw_voronoi_mosaic(ax, cx_range, col_func, n=28, seed=3, y_offset=0):
        rng = np.random.RandomState(seed)
        pts = np.column_stack([rng.uniform(*cx_range,n),
                               rng.uniform(.5+y_offset, 4.5+y_offset, n)])
        # Bounded Voronoi using mirror points
        mirror = np.vstack([pts,
                            pts + [0, 10], pts - [0, 10],
                            pts + [15, 0], pts - [15, 0]])
        vor = Voronoi(mirror)
        regions, vertices = vor.regions, vor.vertices
        for i in range(n):
            ri = vor.point_region[i]
            region = regions[ri]
            if -1 in region or len(region)==0: continue
            poly = vertices[region]
            # clip to ax bounds
            from matplotlib.patches import Polygon as MPoly
            patch = MPoly(poly, closed=True,
                          fc=col_func(rng), ec='white', lw=.5, alpha=.85)
            ax.add_patch(patch)

    # Responder: heterogeneous colours
    def resp_col(rng): return '#E74C3C' if rng.rand()>.55 else '#2E86C1'
    def nr_col(rng):   return '#7F8C8D'

    ax.add_patch(mpatches.FancyBboxPatch((.05,3.1),9.5,2.05,
                 boxstyle='round,pad=.1',fc='#FEF9E7',ec=COrg,lw=1.2))
    draw_voronoi_mosaic(ax,(0.3,9.5),resp_col,n=28,seed=3,y_offset=2.6)
    ax.text(5,3.22,'Responder: heterogeneous T-cell / exhaustion mosaic',
            ha='center',fontsize=7,color='#7E5109',fontweight='bold')

    ax.add_patch(mpatches.FancyBboxPatch((.05,.3),9.5,2.05,
                 boxstyle='round,pad=.1',fc='#EAF0FB',ec=CNR,lw=1.2))
    draw_voronoi_mosaic(ax,(0.3,9.5),nr_col,n=28,seed=8,y_offset=-.2)
    ax.text(5,.42,'Non-resp.: homogeneous immune distribution',
            ha='center',fontsize=7,color='#1A5276',fontweight='bold')

    # F — HCC5NR dissection
    ax = fig.add_subplot(gs[1,2])
    h5  = CABO_N[CABO_N['sample_id']=='HCC5NR'].iloc[0]
    rC  = CABO_N[CABO_N['response']=='Responder']
    met = ['frac_TM','frac_IE','frac_IM','frac_EM','operator_entropy']
    xlb = ['TM','IE','IM','EM','Entr.']
    h5v = [float(h5[m]) for m in met]
    rm  = [float(rC[m].median()) for m in met]
    x2  = np.arange(len(met)); w = .32
    ax.bar(x2-w/2, h5v, width=w, color=CGry, alpha=.9, label='HCC5NR', ec='white')
    ax.bar(x2+w/2, rm,  width=w, color=CR,   alpha=.75, label='R median', ec='white')
    ax.set_xticks(x2); ax.set_xticklabels(xlb, fontsize=FS['tick'])
    ax.set_ylabel('Value', fontsize=FS['label'])
    ax.set_title('F   HCC5NR: organized but TM-dominated', fontsize=FS['title'])
    ax.legend(fontsize=7, handlelength=1.2)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'F')

    plt.suptitle(
        'Figure 3  |  Immune–immune algebra discriminates response\n'
        'TM backbone is universal; IE/IM/EM enrichment is response-specific',
        fontsize=10, fontweight='bold', y=0.96)
    plt.savefig(OUT/'fig3_ncg_algebra.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.close(); print('Fig3 ✓')

build_fig3()

# ═══ FIGURE 4 — KTS transitions with weighted Sankey ═════════════════════
def build_fig4():
    from collections import Counter
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(14,9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2,3,hspace=0.48,wspace=0.38,
                           left=0.06,right=0.97,top=0.92,bottom=0.07)

    # A — State definitions (clean minimal)
    ax = fig.add_subplot(gs[0,0]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'A')
    ax.set_title('Operator-regime states', fontsize=FS['title'])
    sdefs=[('S0','Absent interface','#AAAAAA',9.0),
           ('S1','Fragmented   low organization','#3B6FA0',7.2),
           ('S2','Connected-concentrated','#5aae61',5.4),
           ('S3','Amplified high-intensity front','#E67E22',3.6),
           ('S4','Residual concentrated','#7D3C98',1.8)]
    for name,desc,col,yy in sdefs:
        ax.add_patch(FancyBboxPatch((.2,yy-.52),9.3,1.04,
                                    boxstyle='round,pad=.08',fc=col+'30',
                                    ec=col,lw=1.3))
        ax.text(1.1,yy,name,fontsize=8.5,fontweight='bold',
                color=col,va='center')
        ax.text(3.0,yy,desc,fontsize=7.5,color='#333',va='center')

    # B — Representative spatial maps
    ax = fig.add_subplot(gs[0,1])
    inA = ax.inset_axes([.02,.08,.44,.86])
    spatial_map(inA,lcc=.351,p95=237,seed=7,resp=True,n=220,
                title='S3  Responder',add_contour=True)
    inB = ax.inset_axes([.52,.08,.44,.86])
    inB.set_facecolor('#FAFAFA')
    spatial_map(inB,lcc=.07,p95=2.7,seed=9,resp=False,n=220,
                title='S1  Non-resp.',add_contour=False)
    ax.axis('off'); ax.set_title('B   Representative interfaces',fontsize=FS['title'])
    panel_label(ax,'B')

    # C — Weighted Sankey
    ax = fig.add_subplot(gs[0,2]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,11)
    panel_label(ax,'C')
    ax.set_title('Pre → Post transitions  (n=11 patients)',fontsize=FS['title'])
    ax.text(1.5,10.5,'Pre',ha='center',fontsize=FS['label'],fontweight='bold')
    ax.text(8.5,10.5,'Post',ha='center',fontsize=FS['label'],fontweight='bold')

    transitions=[
        ('S0',9.7,'S0',9.7,'R',1),('S3',8.4,'S4',8.4,'R',1),
        ('S3',7.1,'S0',7.1,'R',1),('S?',5.8,'S1',5.8,'R',1),
        ('S2',4.5,'S3',4.5,'R',1),
        ('S1',9.2,'S1',9.2,'NR',1),('S3',7.8,'S1',7.8,'NR',1),
        ('S1',6.5,'S1',6.5,'NR',1),('S?',5.2,'S1',5.2,'NR',1),
        ('S?',3.9,'S1',3.9,'NR',1),('S?',2.6,'S1',2.6,'NR',1)]

    for sp,yp,sq,yq,resp,_ in transitions:
        col = CR if resp=='R' else CNR
        lw  = 2.2 if resp=='NR' else 1.8

        # State boxes
        for (st,yy,xc) in [(sp,yp,.2),(sq,yq,7.25)]:
            ax.add_patch(FancyBboxPatch((xc,yy-.28),2.2,.56,
                         boxstyle='round,pad=.04',fc=CS[st]+'50',
                         ec=CS[st],lw=1.1))
            ax.text(xc+1.1,yy,st,ha='center',va='center',
                    fontsize=7.5,color=CS[st],fontweight='bold')

        # Flow line
        ax.annotate('',xy=(7.25,yq),xytext=(2.4,yp),
                    arrowprops=dict(arrowstyle='-|>',color=col,
                                   lw=lw,alpha=.80,mutation_scale=7))

    # Highlight NR → S1 cluster
    ax.add_patch(FancyBboxPatch((6.9,2.2),2.8,7.4,
                 boxstyle='round,pad=.1',fc='none',ec=CNR,lw=1.8,ls='--',alpha=.5))
    ax.text(8.3,1.9,'S1 convergence',ha='center',fontsize=7,color=CNR,fontweight='bold')

    # D — Post-therapy distribution
    ax = fig.add_subplot(gs[1,0])
    r_post=['S0','S4','S0','S1','S3']; nr_post=['S1','S1','S1','S1','S1','S1']
    for grp,xoff in [(r_post,-.22),(nr_post,.22)]:
        cnts=Counter(grp); tot=len(grp); bot=0
        for st in ['S0','S1','S2','S3','S4']:
            h=cnts.get(st,0)/tot
            if h>0:
                ax.bar([xoff],h,bottom=bot,width=.38,color=CS[st],ec='white',lw=.4)
                if h>.09:
                    ax.text(xoff,bot+h/2,st,ha='center',va='center',
                            fontsize=8,fontweight='bold',color='white')
                bot+=h
    ax.set_xticks([-.22,.22])
    ax.set_xticklabels(['Resp.\n(n=5)','Non-resp.\n(n=6)'],fontsize=FS['tick'])
    ax.set_ylabel('Fraction',fontsize=FS['label'])
    ax.set_title('D   Post-therapy state\nFisher p = 0.015',fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'D')

    # E — 2×2 S1 convergence heatmap
    ax = fig.add_subplot(gs[1,1])
    data=np.array([[1,4],[6,0]])
    im=ax.imshow(data,cmap='RdYlGn_r',vmin=0,vmax=6,aspect='auto')
    for i in range(2):
        for j in range(2):
            ax.text(j,i,str(data[i,j]),ha='center',va='center',
                    fontsize=18,fontweight='bold',
                    color='white' if data[i,j]>3 else '#333')
    ax.set_xticks([0,1]); ax.set_xticklabels(['In S1','Non-S1'],fontsize=FS['tick'])
    ax.set_yticks([0,1]); ax.set_yticklabels(['Resp. (n=5)','Non-resp. (n=6)'],
                                               fontsize=FS['tick'])
    ax.set_title('E   S1 convergence\nFisher p = 0.015',fontsize=FS['title'])
    panel_label(ax,'E')

    # F — HCC5NR necessary vs sufficient logic
    ax = fig.add_subplot(gs[1,2]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    ax.set_title('F   HCC5NR: necessary ≠ sufficient',fontsize=FS['title'])
    panel_label(ax,'F')

    # Two columns: what HCC5NR has vs what it lacks
    ax.text(2.8,9.5,'Has',ha='center',fontsize=8,fontweight='bold',color='#1E8449')
    ax.text(7.5,9.5,'Lacks',ha='center',fontsize=8,fontweight='bold',color='#C0392B')
    ax.axvline(5.1,ymin=0.05,ymax=0.93,color='#DDD',lw=1,ls='-')

    has_items=['Spatial score 0.50','LCC = 0.85\n(connected)','S4 state']
    lack_items=['IE commutator\n< all responders','Operator entropy\n< all responders',
                'Finite-range\nconstraint']
    for i,item in enumerate(has_items):
        y=8.2-i*2.0
        ax.add_patch(FancyBboxPatch((.2,y-.5),4.5,.9,
                     boxstyle='round,pad=.06',fc='#EAFAF1',ec=CGrn,lw=.9))
        ax.text(2.45,y,item,ha='center',va='center',fontsize=7.2,color='#1E8449')
    for i,item in enumerate(lack_items):
        y=8.2-i*2.0
        ax.add_patch(FancyBboxPatch((5.3,y-.5),4.3,.9,
                     boxstyle='round,pad=.06',fc='#FDEDEC',ec='#C0392B',lw=.9))
        ax.text(7.45,y,item,ha='center',va='center',fontsize=7.2,color='#C0392B')
    ax.text(5.0,1.0,'→  Non-responder (stable disease)',
            ha='center',fontsize=7.5,color='#666',style='italic')

    plt.suptitle(
        'Figure 4  |  KTS transitions: all non-responders converge to S1\n'
        'Spatial organisation is necessary but not sufficient (HCC5NR)',
        fontsize=10,fontweight='bold',y=0.96)
    plt.savefig(OUT/'fig4_kts_transitions.png',dpi=300,
                bbox_inches='tight',facecolor='white')
    plt.close(); print('Fig4 ✓')

build_fig4()

# ═══ FIGURE 5 — Finite-range organisation ════════════════════════════════
def build_fig5():
    fig = plt.figure(figsize=(14,9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2,3,hspace=0.46,wspace=0.40,
                           left=0.07,right=0.97,top=0.92,bottom=0.09)

    # A — Local prediction schematic
    ax = fig.add_subplot(gs[0,0]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'A')
    ax.set_title('Local k-NN prediction', fontsize=FS['title'])
    rng=np.random.RandomState(2); cx,cy=5,5.5
    pts=[(cx+rng.randn()*2.1,cy+rng.randn()*2.1) for _ in range(12)]
    for xp,yp in pts:
        d=np.sqrt((xp-cx)**2+(yp-cy)**2)
        c=CMAP_C(1-d/4)
        ax.add_patch(mpatches.Circle((xp,yp),.32,fc=c,ec='white',lw=.5,zorder=3))
    ax.add_patch(mpatches.Circle((cx,cy),.38,fc='#333',ec='white',lw=1,zorder=4))
    ax.text(cx,cy,'i',ha='center',va='center',fontsize=9,color='white',
            fontweight='bold',zorder=5)
    for xp,yp in pts[:5]:
        ax.annotate('',xy=(cx-.1,cy),xytext=(xp,yp),
                    arrowprops=dict(arrowstyle='->',color='#AAA',lw=.8,alpha=.7))
    ax.text(5,.6,'Neighbours predict i accurately\nLocal R² = 0.35–0.84',
            ha='center',fontsize=7,color='#555',style='italic',va='bottom')

    # B — Gibbs overshoot (smoother)
    ax = fig.add_subplot(gs[0,1])
    x = np.linspace(0,10,400)
    ytrue  = np.exp(-((x-5)**2)/4.5)*4 + .3
    yglobal= (np.sin(x*np.pi/10)*2.8 + 0.7*np.cos(x*1.9*np.pi/10) + .5)
    ax.fill_between(x,ytrue,yglobal,where=yglobal>ytrue,
                    alpha=.18,color='#C0392B',label='Overshoot')
    ax.fill_between(x,ytrue,yglobal,where=yglobal<ytrue,
                    alpha=.12,color='#2166AC',label='Undershoot')
    ax.plot(x,ytrue,'k-',lw=2.0,label='True coexact density',zorder=4)
    ax.plot(x,yglobal,'--',color='#C0392B',lw=1.6,label='5-mode reconstruction',zorder=4)
    ax.axhline(ytrue.mean(),color='#888',lw=.8,ls=':',alpha=.7)
    ax.text(8.2,ytrue.mean()+.3,'mean',fontsize=6.5,color='#888')
    ax.set_xlim(0,10); ax.set_xlabel('Spatial position',fontsize=FS['label'])
    ax.set_ylabel('Coexact density',fontsize=FS['label'])
    ax.set_title('B   Global eigenmode overshoot\nGibbs-like failure  →  R² < 0',
                 fontsize=FS['title'])
    ax.legend(fontsize=6.5,handlelength=1.2)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'B')

    # C — Local vs global scatter
    ax = fig.add_subplot(gs[0,2])
    r  = D[D['resp']=='R']; nr = D[D['resp']=='NR']
    ax.scatter(r['gr2'],r['lr2'],color=CR,s=70,zorder=4,
               edgecolors='white',lw=.4,label='Responder')
    ax.scatter(nr['gr2'],nr['lr2'],color=CNR,s=70,zorder=4,
               edgecolors='white',lw=.4,label='Non-resp.')
    lim=(-1.7,.75)
    ax.plot(lim,lim,'k--',lw=.7,alpha=.25)
    ax.axvline(0,color='#CCC',lw=.7,ls=':')
    ax.axhline(0,color='#CCC',lw=.7,ls=':')
    # Shade responder zone
    ax.fill_betweenx([-0.15,.90],lim[0],0,alpha=.07,color='#C0392B',zorder=0)
    ax.text(-0.9,.82,'Locally\nconstrained\n(R zone)',ha='center',fontsize=7,
            color=CR,style='italic')
    ax.text(.35,.45,'Globally\ncoherent\n(HCC5NR)',ha='center',fontsize=7,
            color=CNR,style='italic',alpha=.8)
    ax.set_xlabel('Global R²',fontsize=FS['label'])
    ax.set_ylabel('Local R²',fontsize=FS['label'])
    ax.set_title('C   Local vs global R²',fontsize=FS['title'])
    ax.legend(fontsize=7,handlelength=1.2)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'C')

    # D — KTS hierarchy bar chart
    ax = fig.add_subplot(gs[1,0])
    sts=['S4','S3','S2','S1']
    meds=[D[D['kts']==s]['lr2'].median() for s in sts]
    errs=[D[D['kts']==s]['lr2'].sem() for s in sts]
    bars=ax.bar(sts,meds,color=[CS[s] for s in sts],
                width=.55,ec='white',lw=.4,zorder=3)
    ax.errorbar(sts,meds,yerr=errs,fmt='none',
                color='#333',capsize=3.5,lw=1.0,zorder=4)
    ax.axhline(0,color='#DDD',lw=.6,ls='--')
    ax.set_ylabel('Median local R²',fontsize=FS['label'])
    ax.set_title('D   KTS hierarchy\nS4 > S3 > S2 > S1',fontsize=FS['title'])
    for b,v in zip(bars,meds):
        ax.text(b.get_x()+b.get_width()/2, v+.015,
                f'{v:.3f}',ha='center',fontsize=7.5,color='#333')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'D')

    # E — Geometric schematic: front vs ring
    ax = fig.add_subplot(gs[1,1]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'E')
    ax.set_title('Finite-range front vs exclusion ring',fontsize=FS['title'])

    # Left: responder compact cluster
    import matplotlib.patheffects as pe
    cluster=mpatches.Ellipse((2.5,5.5),3.5,3.5,fc='#FDE8D8',ec=COrg,lw=2.0)
    ax.add_patch(cluster)
    rng5=np.random.RandomState(6)
    pts_c=np.column_stack([rng5.randn(10)*.7+2.5,rng5.randn(10)*.7+5.5])
    for xp,yp in pts_c:
        ax.add_patch(mpatches.Circle((xp,yp),.22,fc=COrg,ec='none',alpha=.8,zorder=3))
    ax.text(2.5,3.5,'Finite-range\nfront',ha='center',fontsize=7.5,
            color=COrg,fontweight='bold')
    ax.text(2.5,2.7,'local R²↑   global R²↓',ha='center',
            fontsize=7,color='#888',style='italic')
    ax.text(1.0,9.4,'Responder',ha='center',fontsize=8,fontweight='bold',color=CR)

    # Right: globally coherent ring
    ring=mpatches.Circle((7.5,5.5),2.8,fc='none',ec=CNR,lw=2.0,ls='--')
    ax.add_patch(ring)
    ring2=mpatches.Circle((7.5,5.5),.25,fc='white',ec='none')
    ax.add_patch(ring2)
    ax.text(7.5,5.5,'Ring',ha='center',va='center',fontsize=8,
            color=CNR,fontweight='bold')
    ax.text(7.5,2.3,'Globally coherent\nlocal R²≈global R²',ha='center',
            fontsize=7,color=CNR,style='italic')
    ax.text(7.3,9.4,'HCC5NR',ha='center',fontsize=8,fontweight='bold',color=CNR)

    # F — Local R² comparison
    ax = fig.add_subplot(gs[1,2])
    rv  = D[D['resp']=='R']['lr2'].values
    nrv = D[D['resp']=='NR']['lr2'].values
    swarm(ax,0,rv,CR); swarm(ax,1,nrv,CNR)
    _,p = mannwhitneyu(rv,nrv,alternative='two-sided')
    ax.set_xlim(-.6,1.6); ax.set_ylim(-.18,.65)
    stat_bracket(ax,0,1,rv.max()+.04,p)
    ax.axhline(0,color='#DDD',lw=.6,ls='--')
    ax.set_xticks([0,1])
    ax.set_xticklabels(['Responder','Non-resp.'],fontsize=FS['tick'])
    ax.set_ylabel('Local R² (k-NN prediction)',fontsize=FS['label'])
    ax.set_title('F   Local predictability\np = 0.033',fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'F')

    plt.suptitle(
        'Figure 5  |  Responder interfaces are locally constrained, not globally periodic',
        fontsize=10,fontweight='bold',y=0.96)
    plt.savefig(OUT/'fig5_finite_range.png',dpi=300,
                bbox_inches='tight',facecolor='white')
    plt.close(); print('Fig5 ✓')

build_fig5()

# ═══ FIGURE 6 — External CABO/NIVO validation ═════════════════════════════
def build_fig6():
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(14,9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2,3,hspace=0.48,wspace=0.40,
                           left=0.07,right=0.97,top=0.92,bottom=0.08)

    # A — Icon-based concordance matrix
    ax = fig.add_subplot(gs[0,0]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'A')
    ax.set_title('Discovery → CABO/NIVO concordance',fontsize=FS['title'])

    rows=[
        ('Coexact fraction',    '→','directional', '#F0B27A'),
        ('Spatial score',       '→','directional', '#F0B27A'),
        ('TM backbone',         '✓','confirmed',   '#82E0AA'),
        ('IE commutator',       '✓✓','replicated', '#27AE60'),
        ('Operator entropy',    '✓✓','replicated', '#27AE60'),
        ('KTS S3 enrichment',   '✓', 'replicated', '#82E0AA'),
        ('Organized NR (HCC5NR)','✦','new finding','#AED6F1'),
    ]
    for i,(metric,sym,status,bg) in enumerate(rows):
        y=9.1-i*1.28
        ax.add_patch(FancyBboxPatch((.1,y-.50),9.6,.96,
                     boxstyle='round,pad=.04',fc=bg+'77',ec=bg,lw=.8))
        ax.text(.6,y,sym,fontsize=12,va='center',fontweight='bold',
                color='#1A5276' if '✓' in sym else '#856404')
        ax.text(1.8,y,metric,fontsize=7.5,va='center',color='#333')
        ax.text(8.0,y,status,fontsize=7.2,va='center',ha='center',
                color='#1A5276',fontweight='bold' if '✓✓' in sym else 'normal')

    # B — Replicated Hodge metrics
    ax = fig.add_subplot(gs[0,1])
    cr  = CABO_H[CABO_H['response']=='Responder']
    cnr = CABO_H[CABO_H['response']=='Non_Responder']
    for xi,(key,lbl) in enumerate([('coexact_fraction','Coexact\nfrac.'),
                                    ('spatial_score','Spatial\nscore'),
                                    ('lcc_fraction','LCC')]):
        swarm(ax,xi-.17,cr[key].values,CR,s=55)
        swarm(ax,xi+.17,cnr[key].values,CNR,s=55)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['Coexact\nfrac.','Spatial\nscore','LCC'],fontsize=FS['tick'])
    ax.set_title('B   Hodge metrics  (directional)',fontsize=FS['title'])
    for c,l in [(CR,'Responder'),(CNR,'Non-resp.')]:
        ax.plot([],[],color=c,label=l,marker='o',lw=0,ms=5)
    ax.legend(fontsize=7,handletextpad=0.3)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'B')

    # C — NCG replication
    ax = fig.add_subplot(gs[0,2])
    crn = CABO_N[CABO_N['response']=='Responder']
    cnn = CABO_N[CABO_N['response']=='Non_Responder']
    for xi,(key,col) in enumerate([('frac_IE',CPur),('operator_entropy',CR)]):
        rv2=crn[key].values; nrv2=cnn[key].values
        _,p=mannwhitneyu(rv2,nrv2,alternative='greater')
        swarm(ax,xi-.15,rv2,col,s=65)
        swarm(ax,xi+.15,nrv2,CGry,s=65)
        ax.set_ylim(-0.01,max(rv2.max(),nrv2.max())*1.35)
        stat_bracket(ax,xi-.15,xi+.15,max(rv2.max(),nrv2.max())+rv2.max()*.04,p,h_frac=.14)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['IE fraction','Operator entropy'],fontsize=FS['tick'])
    ax.set_ylabel('Value',fontsize=FS['label'])
    ax.set_title('C   NCG replication  (p = 0.029 each)',fontsize=FS['title'])
    for c,l in [(CPur,'Responder'),(CGry,'Non-resp.')]:
        ax.plot([],[],color=c,label=l,marker='o',lw=0,ms=5)
    ax.legend(fontsize=7)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'C')

    # D — External KTS distribution
    ax = fig.add_subplot(gs[1,0])
    from collections import Counter
    for grp,xoff in [(CABO_K[CABO_K['response']=='Responder'],-.22),
                     (CABO_K[CABO_K['response']=='Non_Responder'],.22)]:
        cnts=Counter(grp['kts_state'].values); tot=len(grp); bot=0
        for st in ['S0','S1','S2','S3','S4']:
            h=cnts.get(st,0)/tot
            if h>0:
                ax.bar([xoff],h,bottom=bot,width=.38,color=CS[st],ec='white',lw=.4)
                if h>.1:
                    ax.text(xoff,bot+h/2,st,ha='center',va='center',
                            fontsize=8,fontweight='bold',color='white')
                bot+=h
    ax.set_xticks([-.22,.22])
    ax.set_xticklabels(['Resp.\n(n=4)','Non-resp.\n(n=3)'],fontsize=FS['tick'])
    ax.set_ylabel('Fraction',fontsize=FS['label'])
    ax.set_title('D   External KTS distribution',fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'D')

    # E — HCC5NR in local vs global space (clean scatter)
    ax = fig.add_subplot(gs[1,1])
    for _,row in CABO_S.iterrows():
        col = CR if row['response']=='Responder' else CNR
        mrkr = 'D' if row['sample_id']=='HCC5NR' else 'o'
        sz   = 130 if row['sample_id']=='HCC5NR' else 65
        ax.scatter(row['global_r2'],row['local_r2'],
                   color=col,s=sz,zorder=4,marker=mrkr,
                   edgecolors='#333' if mrkr=='D' else 'white',lw=.8)
        ax.text(row['global_r2']+.02,row['local_r2'],
                row['sample_id'].replace('HCC',''),fontsize=7.5,color=col)
    lm=(-1.3,.9)
    ax.plot(lm,lm,'k--',lw=.7,alpha=.25)
    ax.axvline(0,color='#CCC',lw=.7,ls=':')
    ax.axhline(0,color='#CCC',lw=.7,ls=':')
    ax.fill_betweenx([-0.1,.9],lm[0],0,alpha=.07,color=CR,zorder=0)
    ax.text(-0.7,.82,'Locally\nconstrained\n(R zone)',ha='center',fontsize=7,
            color=CR,style='italic')
    ax.text(.36,.38,'Globally\ncoherent',ha='center',fontsize=7,
            color=CNR,style='italic')
    ax.set_xlabel('Global R²',fontsize=FS['label'])
    ax.set_ylabel('Local R²',fontsize=FS['label'])
    ax.set_title('E   HCC5NR: globally coherent organized NR',fontsize=FS['title'])
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'E')

    # F — Causal hierarchy (necessary conditions)
    ax = fig.add_subplot(gs[1,2]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'F')
    ax.set_title('Necessary conditions for response regime',fontsize=FS['title'])

    conds=[
        ('Spatial organisation','LCC, spatial score','Necessary but\nnot sufficient','#F0B27A'),
        ('Immune–immune algebra','IE, IM, EM>','Required','#C0392B'),
        ('Finite-range constraint','local R² >> global R²','Required','#C0392B'),
        ('S3 / S4 state','+ appropriate algebra','Associated','#AED6F1'),
    ]
    for i,(t,d,st,col) in enumerate(conds):
        y=8.8-i*2.3
        ax.add_patch(FancyBboxPatch((.2,y-.85),9.3,1.65,
                     boxstyle='round,pad=.1',fc=col+'55',ec=col,lw=1.3))
        ax.text(5,y+.35,t,ha='center',fontsize=8.5,fontweight='bold',color='#222')
        ax.text(5,y-.05,d,ha='center',fontsize=7.5,color='#444')
        ax.text(5,y-.55,st,ha='center',fontsize=7,color='#555',style='italic')

    plt.suptitle(
        'Figure 6  |  External CABO/NIVO replication: immune–immune algebra confirmed\n'
        'HCC5NR dissects necessary vs sufficient conditions',
        fontsize=10,fontweight='bold',y=0.96)
    plt.savefig(OUT/'fig6_external_validation.png',dpi=300,
                bbox_inches='tight',facecolor='white')
    plt.close(); print('Fig6 ✓')

build_fig6()

# ═══ FIGURE 7 — IMC mesoscopic emergence ══════════════════════════════════
def build_fig7():
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(14,9))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(2,3,hspace=0.46,wspace=0.40,
                           left=0.07,right=0.97,top=0.92,bottom=0.08)

    EWIN_COL = '#FFF3CD'   # emergence window highlight
    EWIN_EC  = '#F0B27A'

    # A — IMC cell-level schematic
    ax = fig.add_subplot(gs[0,0]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'A')
    ax.set_title('Single-cell level\n(discrete puncta)',fontsize=FS['title'])
    rng=np.random.RandomState(11)
    for _ in range(28):
        ax.add_patch(mpatches.Circle(
            (rng.uniform(.8,4.2),rng.uniform(1.5,8.5)),
            .22,fc='#C0392B',ec='white',lw=.3,alpha=.78,zorder=3))
    for _ in range(24):
        ax.add_patch(mpatches.Circle(
            (rng.uniform(5.8,9.2),rng.uniform(1.5,8.5)),
            .18,fc='#2166AC',ec='white',lw=.3,alpha=.78,zorder=3))
    ax.text(2.5,.3,'Tumour',ha='center',fontsize=7.5,color='#C0392B',fontweight='bold')
    ax.text(7.5,.3,'Immune',ha='center',fontsize=7.5,color='#2166AC',fontweight='bold')
    ax.text(5,9.5,'→ No coherent coexact fronts',ha='center',
            fontsize=7,color='#555',style='italic')
    ax.axvline(5.0,ymin=.08,ymax=.94,color='#DDD',lw=1.2,ls='--')

    # B — Pseudospot coarse-graining grid
    ax = fig.add_subplot(gs[0,1]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'B')
    ax.set_title('Pseudospot coarse-graining\n(80px ≈ 55µm)',fontsize=FS['title'])
    rng2=np.random.RandomState(6)
    for xi in range(4):
        for yi in range(4):
            x0=xi*2.2+.8; y0=yi*2.2+.8
            t_frac=(4-xi)/5
            mixed=abs(t_frac-.5)<.22
            col='#F8C471' if mixed else('#FADBD8' if xi<2 else '#D6EAF8')
            ec='#E67E22' if mixed else('#922B21' if xi<2 else '#1A5276')
            ax.add_patch(FancyBboxPatch((x0,y0),1.9,1.9,
                         boxstyle='round,pad=.06',fc=col,ec=ec,lw=1.0))
            lbl='⊕' if mixed else('T' if xi<2 else 'I')
            fc='#E67E22' if mixed else('#922B21' if xi<2 else '#1A5276')
            ax.text(x0+.95,y0+.95,lbl,ha='center',va='center',
                    fontsize=11,color=fc,fontweight='bold')
    ax.text(5,.15,'⊕ = interface bin (tumour adj. immune)',
            ha='center',fontsize=7,color='#E67E22',style='italic')

    # C — LCC scale sweep (phase-transition style)
    ax = fig.add_subplot(gs[0,2])
    scales=[40,60,80,100,120,160,240]
    lcc_r =[.74,.82,.87,.84,.79,.71,.60]
    lcc_nr=[.86,.84,.80,.79,.75,.68,.60]

    ax.axvspan(72,108,alpha=.22,color=EWIN_COL,zorder=0,
               label='Emergence window')
    ax.plot(scales,lcc_r,'o-',color=CR,lw=2.2,ms=7,zorder=3,
            label='Responder',markeredgecolor='white',markeredgewidth=.4)
    ax.plot(scales,lcc_nr,'s--',color=CNR,lw=1.8,ms=6,zorder=3,
            label='Non-resp.',markeredgecolor='white',markeredgewidth=.4)

    ax.set_xlabel('Pseudospot size (px)',fontsize=FS['label'])
    ax.set_ylabel('LCC fraction',fontsize=FS['label'])
    ax.set_title('C   LCC emergence across scales',fontsize=FS['title'])
    ax.set_ylim(.52,.96)
    ax.legend(fontsize=7,handlelength=1.4)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'C')

    # D — Effect-size peak
    ax = fig.add_subplot(gs[1,0])
    eff=[abs(lcc_r[i]-lcc_nr[i]) for i in range(len(scales))]
    cols2=[EWIN_COL if 65<=s<=110 else '#D5D8DC' for s in scales]
    bars=ax.bar(range(len(scales)),eff,color=cols2,ec=[EWIN_EC if 65<=s<=110
                else '#AAA' for s in scales],width=.65,lw=.8)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels([str(s) for s in scales],fontsize=FS['tick'])
    ax.set_xlabel('Pseudospot size (px)',fontsize=FS['label'])
    ax.set_ylabel('|LCC_R − LCC_NR|',fontsize=FS['label'])
    ax.set_title('D   Response-signal peak\nat emergence window',fontsize=FS['title'])
    ax.axvspan(.5,2.5,alpha=.20,color=EWIN_COL,zorder=0)
    ax.text(1.5,max(eff)*.80,'Peak\n~80px',ha='center',
            fontsize=7.5,color='#7E5109',fontweight='bold')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'D')

    # E — Valid ROI collapse
    ax = fig.add_subplot(gs[1,1])
    n_valid=[35,48,62,44,31,18,3]
    ax.fill_between([40,72],[0,0],[70,70],alpha=.15,color='#EBF5FB',label='Sub-meso.')
    ax.fill_between([72,108],[0,0],[70,70],alpha=.22,color=EWIN_COL,label='Emergence')
    ax.fill_between([108,240],[0,0],[70,70],alpha=.15,color='#EAF0FB',label='Over-coarse')
    ax.plot(scales,n_valid,'ko-',lw=2.0,ms=7,zorder=4,
            markeredgecolor='white',markeredgewidth=.4,label='Valid ROIs')
    ax.set_xlabel('Pseudospot size (px)',fontsize=FS['label'])
    ax.set_ylabel('Valid ROIs (n≥20 interface)',fontsize=FS['label'])
    ax.set_title('E   Interface collapses above ~120px',fontsize=FS['title'])
    ax.legend(fontsize=7,handlelength=1.2)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    panel_label(ax,'E')

    # F — Three-scale schematic (clean hierarchy)
    ax = fig.add_subplot(gs[1,2]); ax.axis('off')
    ax.set_xlim(0,10); ax.set_ylim(0,10)
    panel_label(ax,'F')
    ax.set_title('Scale-dependent emergence',fontsize=FS['title'])

    tiers=[
        ('~10 µm\nSingle cells','Microscopic','Discrete puncta\nNo coherent fronts',
         '#F4D03F','#922B21',2),
        ('55–70 µm\n≈ Visium spot','Mesoscopic\n(emergence window)','Coherent operator fronts\nResponse signal',
         EWIN_COL,'#7E5109',4),
        ('>200 µm\nMacro','Macroscopic','Interface disappears\nOver-homogenised',
         '#EAF0FB','#1A5276',6),
    ]
    for (scale,tier,detail,bg,ec,y_top) in tiers:
        h=1.6; y=y_top+.2
        ax.add_patch(FancyBboxPatch((.3,y),9.2,h,
                     boxstyle='round,pad=.08',fc=bg,ec=ec,lw=1.4))
        ax.text(2.0,y+h/2,scale,ha='center',va='center',
                fontsize=7.5,fontweight='bold',color=ec)
        ax.text(5.0,y+h/2+.28,tier,ha='center',va='center',
                fontsize=8,color='#222',fontweight='bold')
        ax.text(5.0,y+h/2-.32,detail,ha='center',va='center',
                fontsize=7,color='#555',style='italic')

    ax.annotate('',xy=(5,4.4),xytext=(5,2.1),
               arrowprops=dict(arrowstyle='->',color='#888',lw=1.2))
    ax.annotate('',xy=(5,6.6),xytext=(5,4.35),
               arrowprops=dict(arrowstyle='->',color='#888',lw=1.2))
    ax.text(5.4,3.2,'coarse-\ngraining',fontsize=6.5,color='#999',style='italic')
    ax.text(5.4,5.5,'coarse-\ngraining',fontsize=6.5,color='#999',style='italic')

    plt.suptitle(
        'Figure 7  |  IMC cross-modality stress test: operator regime emerges at mesoscopic scale\n'
        'Signal absent at cell level; coherent at ~80px; collapses above ~120px',
        fontsize=10,fontweight='bold',y=0.96)
    plt.savefig(OUT/'fig7_imc_scale.png',dpi=300,
                bbox_inches='tight',facecolor='white')
    plt.close(); print('Fig7 ✓')

build_fig7()
print('\nAll 7 v2 figures complete.')


# ── CLI driver ────────────────────────────────────────────────────────────
BUILDERS = {
    1: build_fig1,
    2: build_fig2,
    3: build_fig3,
    4: build_fig4,
    5: build_fig5,
    6: build_fig6,
    7: build_fig7,
}

FIGURE_LABELS = {
    1: 'Conceptual framework',
    2: 'Hodge geometry — responder interfaces form connected coexact fronts',
    3: 'NCG immune algebra — immune–immune commutators discriminate response',
    4: 'KTS transitions — all non-responders converge to S1',
    5: 'Finite-range organisation — locally constrained, not globally periodic',
    6: 'External CABO/NIVO validation — immune algebra replication + HCC5NR',
    7: 'IMC mesoscopic scale emergence',
}


def parse_args():
    ap = argparse.ArgumentParser(
        description='Generate main manuscript figures 1–7')
    ap.add_argument('--outdir', default='figures/',
                    help='Output directory (default: figures/)')
    ap.add_argument('--dpi', type=int, default=300,
                    help='Output DPI (default: 300)')
    ap.add_argument('--fig', default=None,
                    help='Comma-separated figure numbers to build '
                         '(default: all). Example: --fig 1,3,5')
    return ap.parse_args()


def main():
    args = parse_args()
    out  = _Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Override global OUT path
    import fig_globals as _fg
    _fg.OUT = out

    # Determine which figures to build
    if args.fig:
        nums = [int(n.strip()) for n in args.fig.split(',')]
    else:
        nums = list(BUILDERS.keys())

    print(f"\nBuilding {len(nums)} figure(s) → {out}/\n")
    for n in nums:
        if n not in BUILDERS:
            print(f"  [skip] Figure {n} not defined")
            continue
        print(f"  Figure {n}: {FIGURE_LABELS[n]}")
        BUILDERS[n]()

    print(f"\nDone. {len(nums)} figure(s) saved to {out}/")


if __name__ == '__main__':
    main()
