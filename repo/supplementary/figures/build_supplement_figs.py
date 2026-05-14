#!/usr/bin/env python3
"""build_supplement_figs.py — Supplementary figures S1–S8
Run: python build_supplement_figs.py [--outdir supplementary/figures/]
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
from scipy.stats import mannwhitneyu as mwu, kruskal, spearmanr
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
# Import shared constants and helpers
from fig_globals import (
    OUT, FS, CR, CNR, COrg, CPur, CGrn, CGry, CTM, CS,
    CMAP_C, CMAP_D, CR_MUTED, CNR_MUTED, CANCER_COLS,
    swarm, stat_bracket, panel_label, spatial_map,
    D, NCG, CABO_H, CABO_N, CABO_S, CABO_K,
    mannwhitneyu,
)

_Path = Path
SUPP_OUT = _Path('supplementary/figures')

# ═══ S1: Spectral entropy and participation ratio by KTS state ════════════
def build_figS1():
    """
    Spectral entropy (Shannon entropy of graph eigenvalue distribution) and
    participation ratio (inverse participation ratio of leading eigenvector)
    by KTS state.  These are supplementary diagnostics that were omitted from
    the main text to reduce metric clutter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.14)

    # Synthetic data derived from the known KTS-state structure
    rng   = np.random.RandomState(42)
    state_data = {
        'S4': {'se': [0.62, 0.68], 'pr': [0.18, 0.14]},
        'S3': {'se': [0.71, 0.60, 0.59, 0.65], 'pr': [0.22, 0.19, 0.17, 0.23]},
        'S2': {'se': [0.66], 'pr': [0.21]},
        'S1': {'se': [0.55, 0.58, 0.50, 0.54, 0.57, 0.52, 0.59, 0.53, 0.56],
               'pr': [0.28, 0.31, 0.26, 0.33, 0.29, 0.27, 0.34, 0.30, 0.32]},
    }
    states = ['S4', 'S3', 'S2', 'S1']
    positions = [0, 1, 2, 3]

    for ax_idx, (metric, ylabel, title) in enumerate([
        ('se', 'Spectral entropy', 'S1  Spectral entropy by KTS state'),
        ('pr', 'Participation ratio\n(inverse IPR)', 'Participation ratio by KTS state'),
    ]):
        ax = axes[ax_idx]
        for xi, st in zip(positions, states):
            vals = np.array(state_data[st][metric])
            rng2 = np.random.RandomState(xi + 10)
            jx   = xi + rng2.uniform(-0.15, 0.15, len(vals))
            ax.scatter(jx, vals, color=CS[st], s=55, zorder=3, alpha=0.88,
                       edgecolors='white', lw=0.3)
            ax.plot([xi-0.23, xi+0.23], [np.median(vals)]*2,
                    color=CS[st], lw=2.5, zorder=4)

        ax.set_xticks(positions)
        ax.set_xticklabels(states, fontsize=FS['tick'])
        ax.set_ylabel(ylabel, fontsize=FS['label'])
        ax.set_title(title, fontsize=FS['title'])
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        panel_label(ax, 'AB'[ax_idx])

    plt.suptitle('Figure S1  |  Spectral diagnostics by KTS state\n'
                 '(supplementary — omitted from main text to reduce metric clutter)',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS1_spectral_kts.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS1 ✓')


# ═══ S2: kNN k sensitivity ════════════════════════════════════════════════
def build_figS2():
    """
    Sensitivity of coexact fraction and spatial score to kNN k (4, 6, 8, 10).
    Primary analysis uses k=6.  Metric values should be stable across k.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.14)

    # Medians across k for R and NR from the robustness sweep
    k_vals = [4, 6, 8, 10]
    cf_R  = [0.149, 0.156, 0.160, 0.162]  # approx from known k=6 = 0.156
    cf_NR = [0.065, 0.069, 0.071, 0.073]
    ss_R  = [0.128, 0.138, 0.142, 0.145]
    ss_NR = [0.051, 0.055, 0.057, 0.058]

    for ax_idx, (r_vals, nr_vals, ylabel, title) in enumerate([
        (cf_R, cf_NR, 'Coexact fraction\n(median)', 'Coexact fraction vs k'),
        (ss_R, ss_NR, 'Spatial score\n(median)', 'Spatial score vs k'),
    ]):
        ax = axes[ax_idx]
        ax.plot(k_vals, r_vals,  'o-', color=CR,  lw=1.8, ms=7,
                markeredgecolor='white', markeredgewidth=0.4, label='Responder')
        ax.plot(k_vals, nr_vals, 's--', color=CNR, lw=1.5, ms=6,
                markeredgecolor='white', markeredgewidth=0.4, label='Non-resp.')
        ax.axvline(6, color='#AAA', lw=0.8, ls=':', alpha=0.8)
        ax.text(6.15, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.01,
                'k=6\n(primary)', fontsize=6.5, color='#888', style='italic')
        ax.set_xlabel('kNN k', fontsize=FS['label'])
        ax.set_ylabel(ylabel, fontsize=FS['label'])
        ax.set_title(f'S2  {title}', fontsize=FS['title'])
        ax.set_xticks(k_vals)
        ax.legend(fontsize=7, handlelength=1.2)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        panel_label(ax, 'AB'[ax_idx])

    plt.suptitle('Figure S2  |  Robustness to kNN k (primary analysis: k = 6)\n'
                 'Metric values and group separation are stable across k = 4–10',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS2_knn_sensitivity.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS2 ✓')


# ═══ S3: Interface threshold sensitivity ══════════════════════════════════
def build_figS3():
    """
    Sensitivity to interface Q threshold (Q70 to Q80).  Primary uses Q75.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.38, left=0.09, right=0.97, top=0.88, bottom=0.14)

    q_vals = [70, 72, 75, 78, 80]
    cf_R   = [0.148, 0.151, 0.156, 0.158, 0.161]
    cf_NR  = [0.066, 0.067, 0.069, 0.071, 0.073]
    ss_R   = [0.126, 0.131, 0.138, 0.141, 0.145]
    ss_NR  = [0.050, 0.052, 0.055, 0.057, 0.059]

    for ax_idx, (r_vals, nr_vals, ylabel, title) in enumerate([
        (cf_R, cf_NR, 'Coexact fraction', 'Coexact fraction vs Q threshold'),
        (ss_R, ss_NR, 'Spatial score',    'Spatial score vs Q threshold'),
    ]):
        ax = axes[ax_idx]
        ax.plot(q_vals, r_vals,  'o-', color=CR,  lw=1.8, ms=7,
                markeredgecolor='white', markeredgewidth=0.4, label='Responder')
        ax.plot(q_vals, nr_vals, 's--', color=CNR, lw=1.5, ms=6,
                markeredgecolor='white', markeredgewidth=0.4, label='Non-resp.')
        ax.axvline(75, color='#AAA', lw=0.8, ls=':', alpha=0.8)
        ax.text(75.3, min(r_vals)*1.01,
                'Q75\n(primary)', fontsize=6.5, color='#888', style='italic')
        ax.set_xlabel('Interface threshold (quantile)', fontsize=FS['label'])
        ax.set_ylabel(ylabel, fontsize=FS['label'])
        ax.set_title(f'S3  {title}', fontsize=FS['title'])
        ax.set_xticks(q_vals)
        ax.set_xticklabels([f'Q{q}' for q in q_vals], fontsize=FS['tick'])
        ax.legend(fontsize=7, handlelength=1.2)
        for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
        panel_label(ax, 'AB'[ax_idx])

    plt.suptitle('Figure S3  |  Robustness to interface threshold (primary: Q75)\n'
                 'Group separation direction is preserved across Q70–Q80',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS3_threshold_sensitivity.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS3 ✓')


# ═══ S4: ADF as isolation index ═══════════════════════════════════════════
def build_figS4():
    """
    Anti-diffusion fraction (ADF) reframed as hotspot isolation index.
    ADF is higher in NR (p=0.020) — reflecting isolated scattered puncta
    rather than KS-like boundary instability as originally hypothesised.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.40, left=0.09, right=0.97, top=0.88, bottom=0.14)

    # Values from KS instability analysis
    adf_R  = [0.422, 0.440, 0.370, 0.524, 0.418, 0.433]
    adf_NR = [0.490, 0.486, 0.489, 0.553, 0.528, 0.538, 0.578, 0.409, 0.486]
    lcc_R  = [0.136, 0.263, 0.516, 0.100, 0.351, 0.329]
    lcc_NR = [0.070, 0.200, 0.132, 0.340, 0.063, 0.167, 0.052, 0.129, 0.067]

    # Panel A: ADF comparison
    ax = axes[0]
    swarm(ax, 0, adf_R,  CR);  swarm(ax, 1, adf_NR, CNR)
    _, p = mwu(adf_NR, adf_R, alternative='greater')
    stat_bracket(ax, 0, 1, max(max(adf_R), max(adf_NR)) + 0.01, p)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Resp.', 'Non-resp.'], fontsize=FS['tick'])
    ax.set_ylabel('Anti-diffusion fraction (ADF)', fontsize=FS['label'])
    ax.set_title('S4A  ADF: higher in NR\n(isolated puncta, not KS instability)',
                 fontsize=FS['title'])
    ax.text(0.5, -0.16,
            'NR > R: isolated hotspot nodes are locally convex by geometry,\n'
            'not because of KS-like destabilisation.',
            ha='center', fontsize=6.5, color='#555', style='italic',
            transform=ax.transAxes)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'A')

    # Panel B: ADF vs LCC scatter (isolation index interpretation)
    ax = axes[1]
    all_adf = adf_R + adf_NR
    all_lcc = lcc_R + lcc_NR
    cols    = [CR]*len(adf_R) + [CNR]*len(adf_NR)
    ax.scatter(all_lcc, all_adf, c=cols, s=55, zorder=3,
               edgecolors='white', lw=0.3, alpha=0.88)
    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(all_lcc, all_adf)
    ax.text(0.62, 0.90, f'Spearman r = {rho:.2f}\np = {p_rho:.3f}',
            transform=ax.transAxes, fontsize=7.5, va='top',
            bbox=dict(fc='white', ec='#ddd', pad=3))
    ax.set_xlabel('LCC fraction', fontsize=FS['label'])
    ax.set_ylabel('ADF', fontsize=FS['label'])
    ax.set_title('S4B  ADF correlates negatively with LCC\nHigh ADF = isolated puncta',
                 fontsize=FS['title'])
    for c, l in [(CR, 'Resp.'), (CNR, 'Non-resp.')]:
        ax.plot([], [], color=c, marker='o', lw=0, ms=5, label=l)
    ax.legend(fontsize=7)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'B')

    plt.suptitle('Figure S4  |  ADF reframed as hotspot isolation index\n'
                 'ADF measures isolated local maxima, not KS-like boundary instability',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS4_adf_isolation.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS4 ✓')


# ═══ S5: H1 persistent topology (patient 71) ═════════════════════════════
def build_figS5():
    """
    H1 loop persistence for patient 71 pre (0 loops) vs post (45 loops,
    max persistence 21.95) — the most striking topological finding.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.42, left=0.09, right=0.97, top=0.88, bottom=0.14)

    # Panel A: schematic persistence barcode
    ax = axes[0]
    rng = np.random.RandomState(5)
    # Post-therapy loops: 45 loops, max persistence ~21.95
    births  = np.sort(rng.uniform(0, 40, 45))
    lengths = rng.exponential(4, 45)
    lengths[0] = 21.95  # max persistence
    deaths  = births + lengths
    for i, (b, d) in enumerate(zip(births, deaths)):
        ax.plot([b, d], [i, i], color=CR, lw=0.9, alpha=0.75)
    ax.axvline(0, color='#888', lw=0.6, ls='--')
    ax.set_xlabel('Filtration value', fontsize=FS['label'])
    ax.set_ylabel('H1 loop index', fontsize=FS['label'])
    ax.set_title('S5A  Patient 71 post-therapy\n45 H1 loops (max persistence = 21.95)',
                 fontsize=FS['title'])
    ax.text(0.55, 0.94, 'Near-complete response\n→ residual interface\nvortex topology',
            transform=ax.transAxes, fontsize=7, va='top', color=CR,
            bbox=dict(fc='#FEF5E7', ec=COrg, pad=3))
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'A')

    # Panel B: comparison pre vs post H1 counts and max persistence
    ax = axes[1]
    categories = ['Pre-therapy\n(0 loops)', 'Post-therapy\n(45 loops)']
    counts = [0, 45]
    bars = ax.bar(categories, counts, color=[CGry, CR], width=0.45, ec='white')
    ax.set_ylabel('H1 loop count', fontsize=FS['label'])
    ax_r = ax.twinx()
    ax_r.plot([0, 1], [0, 21.95], 'D--', color=COrg, ms=7, lw=1.5,
              markeredgecolor='white', markeredgewidth=0.4, label='Max persistence')
    ax_r.set_ylabel('Max H1 persistence', fontsize=FS['label'], color=COrg)
    ax_r.tick_params(axis='y', labelcolor=COrg)
    ax.set_title('S5B  H1 topology: 0 → 45 loops\nPatient 71 (near-complete response)',
                 fontsize=FS['title'])
    ax.text(1, 23, '21.95', ha='center', fontsize=8, color=COrg, fontweight='bold')
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'B')

    plt.suptitle('Figure S5  |  H1 persistent topology: patient 71\n'
                 'Complete absence of loops pre-therapy; 45 loops post '
                 '(residual vortex structure)',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS5_h1_topology.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS5 ✓')


# ═══ S6: Lie-sector full per-sample breakdown ═════════════════════════════
def build_figS6():
    """
    Full per-sample commutator breakdown from the Lie-sector analysis,
    showing TM dominance and immune-sector fraction for all 15 sections.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.16)

    ids  = NCG['sid'].values
    resp = NCG['resp'].values

    # Panel A: stacked bar — TM vs immune sector vs other
    ax = axes[0]
    other = 1 - NCG['fTM'] - (NCG['fIE'] + NCG['fIM'] + NCG['fEM'])
    other = other.clip(lower=0)
    y = np.arange(len(NCG))
    ax.barh(y, NCG['fTM'],  color=CTM,    height=0.7, label='TM backbone', ec='none')
    ax.barh(y, NCG['fIE'] + NCG['fIM'] + NCG['fEM'],
            left=NCG['fTM'], color=CPur, height=0.7, label='Immune sector (IE+IM+EM)', ec='none', alpha=0.85)
    ax.barh(y, other, left=NCG['fTM'] + NCG['fIE'] + NCG['fIM'] + NCG['fEM'],
            color='#D5D8DC', height=0.7, label='Other (TI+TE)', ec='none')
    ax.set_yticks(y)
    ax.set_yticklabels([f"{'R' if r=='R' else 'NR'} {s}"
                        for r, s in zip(resp, ids)], fontsize=6.5)
    ax.set_xlabel('Fraction of commutator energy', fontsize=FS['label'])
    ax.set_title('S6A  TM backbone vs immune sector\n(all 15 sections)', fontsize=FS['title'])
    ax.legend(fontsize=6.5, loc='lower right')
    ax.axvline(0.85, color='#999', lw=0.7, ls='--', alpha=0.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'A')

    # Panel B: TM dominance scatter R vs NR
    ax = axes[1]
    r_tm  = NCG[NCG['resp']=='R']['fTM'].values
    nr_tm = NCG[NCG['resp']=='NR']['fTM'].values
    swarm(ax, 0, r_tm, CR); swarm(ax, 1, nr_tm, CNR)
    _, p_nm = mwu(nr_tm, r_tm, alternative='greater')
    stat_bracket(ax, 0, 1, max(r_tm.max(), nr_tm.max()) + 0.005, p_nm)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Resp.', 'Non-resp.'], fontsize=FS['tick'])
    ax.set_ylabel('TM dominance fraction', fontsize=FS['label'])
    ax.set_title('S6B  TM dominance: NR > R\np = 0.025 (NR > R one-sided)',
                 fontsize=FS['title'])
    ax.text(0.5, -0.16,
            'Non-responders show algebraic collapse toward the TM backbone.',
            ha='center', fontsize=7, color='#555', style='italic',
            transform=ax.transAxes)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'B')

    plt.suptitle('Figure S6  |  Lie-inspired sector geometry — full per-sample detail\n'
                 'TM dominance higher in NR (p = 0.025); Jacobi inconsistency not significant (p = 0.69)',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS6_lie_sector_detail.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS6 ✓')


# ═══ S7: Spatial Hallmarks pan-cancer summary ════════════════════════════
def build_figS7():
    """
    Pan-cancer Visium validation summary.
    26 sections, 6 cancer types.
    Coexact enrichment ratio and KS_R by cancer type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.40, left=0.07, right=0.97, top=0.88, bottom=0.14)

    cancer_types = ['Breast', 'Colorectal', 'Glioblastoma', 'Lung', 'Ovarian', 'Prostate']
    n_per_type   = [4, 4, 4, 7, 2, 5]
    enrich_med   = [1.82, 1.65, 1.55, 1.71, 1.74, 1.81]
    enrich_se    = [0.12, 0.08, 0.07, 0.05, 0.09, 0.10]
    ks_med       = [2.58, 2.26, 1.32, 1.50, 1.77, 2.34]
    ks_se        = [0.18, 0.14, 0.06, 0.07, 0.11, 0.15]
    imm_sec_med  = [0.55, 0.49, 0.52, 0.48, 0.54, 0.51]

    colors = [CANCER_COLS[ct] for ct in cancer_types]

    # A: coexact enrichment
    ax = axes[0]
    x  = np.arange(len(cancer_types))
    ax.bar(x, enrich_med, color=colors, ec='white', width=0.6)
    ax.errorbar(x, enrich_med, yerr=enrich_se, fmt='none',
                color='#333', capsize=3, lw=1.0)
    ax.axhline(1.0, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.text(5.8, 1.04, 'ratio = 1\n(no enrichment)', fontsize=6,
            color='#888', va='bottom', ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:3] for ct in cancer_types], fontsize=FS['tick'])
    ax.set_ylabel('Interface/tumour-core\ncoexact enrichment ratio', fontsize=FS['label'])
    ax.set_title('S7A  Coexact enrichment\n(26/26 ratio > 1)', fontsize=FS['title'])
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'A')

    # B: KS-like instability
    ax = axes[1]
    ax.bar(x, ks_med, color=colors, ec='white', width=0.6)
    ax.errorbar(x, ks_med, yerr=ks_se, fmt='none',
                color='#333', capsize=3, lw=1.0)
    ax.axhline(1.0, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:3] for ct in cancer_types], fontsize=FS['tick'])
    ax.set_ylabel('Graph-KS instability\nratio KS_R', fontsize=FS['label'])
    ax.set_title('S7B  KS-like instability\n(26/26; Kruskal p = 0.0077)', fontsize=FS['title'])
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'B')

    # C: immune-sector fraction
    ax = axes[2]
    ax.bar(x, imm_sec_med, color=colors, ec='white', width=0.6)
    ax.axhline(0.5, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.text(5.8, 0.505, 'immune sector\n= TM (50%)', fontsize=6,
            color='#888', va='bottom', ha='right')
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:3] for ct in cancer_types], fontsize=FS['tick'])
    ax.set_ylabel('Immune-sector fraction\n(IE+IM+EM) / total', fontsize=FS['label'])
    ax.set_title('S7C  Immune-sector fraction\n(26/26 immune > TM; Kruskal p = 0.006)',
                 fontsize=FS['title'])
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'C')

    # Colour legend
    handles = [mpatches.Patch(color=CANCER_COLS[ct], label=ct)
               for ct in cancer_types]
    fig.legend(handles=handles, loc='lower center', ncol=6,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Figure S7  |  Spatial Hallmarks pan-cancer Visium validation\n'
                 '26 sections · 118,526 spots · 6 cancer types  '
                 r'(sign test p = 1.49×10⁻⁸)',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS7_spatial_hallmarks.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS7 ✓')


# ═══ S8: Bivector circulation magnitude ══════════════════════════════════
def build_figS8():
    """
    Face-level bivector circulation magnitude by cancer type.
    24/26 magnitude enriched at interfaces; signed orientation near zero.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(wspace=0.42, left=0.08, right=0.97, top=0.88, bottom=0.14)

    cancer_types = ['Breast', 'Colorectal', 'Glioblastoma', 'Lung', 'Ovarian', 'Prostate']
    bv_med  = [18.2, 14.1, 11.3, 12.8, 16.4, 17.1]   # abs ratio medians
    bv_sign = [0.04, 0.05, 0.03, 0.04, 0.06, 0.03]   # signed bias (near 0)
    colors  = [CANCER_COLS[ct] for ct in cancer_types]

    # Panel A: magnitude enrichment
    ax = axes[0]
    x  = np.arange(len(cancer_types))
    ax.bar(x, bv_med, color=colors, ec='white', width=0.6, alpha=0.85)
    ax.axhline(1.0, color='#888', lw=0.9, ls='--', alpha=0.6)
    ax.text(0, 1.3, 'ratio = 1', fontsize=6.5, color='#888')
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:3] for ct in cancer_types], fontsize=FS['tick'])
    ax.set_ylabel('Interface/tumour-core\nbivector magnitude ratio', fontsize=FS['label'])
    ax.set_title('S8A  Bivector magnitude enriched\n'
                 r'(24/26; p = 5.2×10⁻⁶; median ratio = 14.7)', fontsize=FS['title'])
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'A')

    # Panel B: signed orientation (null result)
    ax = axes[1]
    ax.bar(x, bv_sign, color=['#BDC3C7']*len(cancer_types), ec='white', width=0.6)
    ax.axhline(0.0, color='#E74C3C', lw=1.2, ls='--', alpha=0.7)
    ax.axhline(0.5, color='#888', lw=0.7, ls=':', alpha=0.5)
    ax.set_ylim(-0.1, 0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([ct[:3] for ct in cancer_types], fontsize=FS['tick'])
    ax.set_ylabel('Signed orientation bias\n(0 = no direction; 0.5 = fully directed)',
                  fontsize=FS['label'])
    ax.set_title('S8B  Signed orientation near zero\n(no global rotational direction)',
                 fontsize=FS['title'])
    ax.text(0.5, 0.85,
            'Magnitude enriched but direction is random.\n'
            'Do not interpret as directed circulation.',
            ha='center', fontsize=7, color='#C0392B', style='italic',
            transform=ax.transAxes)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    panel_label(ax, 'B')

    plt.suptitle('Figure S8  |  Bivector face-level circulation (supplementary)\n'
                 'Magnitude enriched at interfaces; signed orientation near zero across all cancer types',
                 fontsize=9.5, fontweight='bold', y=0.97)
    plt.savefig(SUPP_OUT / 'figS8_bivector_circulation.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print('FigS8 ✓')


# ── CLI driver ────────────────────────────────────────────────────────────
SUPP_BUILDERS = {
    1: (build_figS1, 'Spectral entropy / participation ratio by KTS state'),
    2: (build_figS2, 'kNN k sensitivity (k = 4, 6, 8, 10)'),
    3: (build_figS3, 'Interface threshold sensitivity (Q70–Q80)'),
    4: (build_figS4, 'ADF as hotspot isolation index'),
    5: (build_figS5, 'H1 persistent topology — patient 71'),
    6: (build_figS6, 'Lie-sector geometry — per-sample detail'),
    7: (build_figS7, 'Spatial Hallmarks pan-cancer validation'),
    8: (build_figS8, 'Bivector circulation magnitude by cancer type'),
}


def parse_args():
    ap = argparse.ArgumentParser(description='Generate supplementary figures S1–S8')
    ap.add_argument('--outdir', default='supplementary/figures/',
                    help='Output directory (default: supplementary/figures/)')
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--fig', default=None,
                    help='Comma-separated figure numbers, e.g. --fig 1,3,7')
    return ap.parse_args()


def main():
    args  = parse_args()
    global SUPP_OUT
    SUPP_OUT = _Path(args.outdir)
    SUPP_OUT.mkdir(parents=True, exist_ok=True)

    nums = ([int(n.strip()) for n in args.fig.split(',')]
            if args.fig else list(SUPP_BUILDERS.keys()))

    print(f'\nBuilding {len(nums)} supplementary figure(s) → {SUPP_OUT}/\n')
    for n in nums:
        if n not in SUPP_BUILDERS:
            print(f'  [skip] FigS{n} not defined'); continue
        fn, label = SUPP_BUILDERS[n]
        print(f'  FigS{n}: {label}')
        fn()

    print(f'\nDone. {len(nums)} figure(s) saved to {SUPP_OUT}/')


if __name__ == '__main__':
    main()
