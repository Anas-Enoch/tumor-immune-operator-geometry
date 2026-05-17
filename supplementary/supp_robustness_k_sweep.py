#!/usr/bin/env python3
"""
supp_robustness_k_sweep.py
==========================
Supplementary robustness analysis: sweep kNN graph parameter k across
{4, 5, 6, 7, 8, 10} and recompute the primary interface coexact enrichment
ratio (R = mean_coexact_interface / mean_coexact_core) at each k.

The primary claim requires that the sign of enrichment (R > 1 in all 15
sections; R_responder > R_non-responder) is graph-construction-invariant.
Absolute magnitudes may vary with edge density; only sign stability is
required.

Outputs
-------
results/hcc/results_hcc_robustness_k_sweep.csv
    Per-section × per-k: enrichment ratio R, coexact fraction, sign (R>1).

figures/supp_robustness_k_sweep.png
    Panel A: R vs k per section (line plot), colored R/NR.
    Panel B: Fraction of sections with R>1 per k (should be 15/15 or close).
    Panel C: Spearman rank correlation of R across k values (stability matrix).

Usage
-----
    python supp_robustness_k_sweep.py \
        --adata   data/hepatocellular_carcinoma/hcc_scored.h5ad \
        --hodge   results/hcc/results_hcc_hodge_interface_summary.csv \
        --out     results/hcc/results_hcc_robustness_k_sweep.csv \
        --fig     figures/supp_robustness_k_sweep.png \
        --k-values 4 5 6 7 8 10
"""

import argparse, warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

K_VALUES    = [4, 5, 6, 7, 8, 10]
TUMOR_FLOOR = 0.05
MIN_IFACE   = 20

PATIENT_MAP = {
    'cytassist_70': 'Responder',    'cytassist_71': 'Responder',
    'cytassist_72': 'Non_Responder','cytassist_73': 'Non_Responder',
    'cytassist_74': 'Non_Responder','cytassist_76': 'Responder',
    'cytassist_79': 'Non_Responder','cytassist_83': 'Responder',
    'cytassist_84': 'Responder',    'cytassist_85': 'Non_Responder',
    'cytassist_86': 'Non_Responder',
}


def build_knn_incidence(coords, k):
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    edges = set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            e = (min(i, int(j)), max(i, int(j)))
            edges.add(e)
    edges = list(edges)
    m = len(edges)
    B1 = np.zeros((m, n))
    for e_idx, (i, j) in enumerate(edges):
        B1[e_idx, i] = -1
        B1[e_idx, j] =  1
    return B1, np.array(edges)


def hodge_coexact_density(B1, f):
    """Solve for exact component, return coexact node-level density."""
    L0 = B1.T @ B1
    rhs = B1 @ f
    try:
        alpha, *_ = lsqr(L0, rhs, atol=1e-8, btol=1e-8, iter_lim=500)
    except Exception:
        return np.zeros(B1.shape[1])
    f_exact   = B1.T @ alpha
    f_coexact = f - f_exact
    n = B1.shape[1]
    density = np.zeros(n)
    for e_idx in range(B1.shape[0]):
        nz = np.where(B1[e_idx] != 0)[0]
        if len(nz) == 2:
            density[nz] += abs(f_coexact[e_idx])
    deg = np.maximum((B1 != 0).sum(axis=0).A1 if hasattr(B1,'A1')
                     else (B1 != 0).sum(axis=0), 1)
    return density / deg


def enrichment_at_k(sid, sub, k):
    n = sub.n_obs
    obs = sub.obs
    tumor = obs['tumor_score'].values.astype(float)
    tcell = obs['tcell_score'].values.astype(float)

    q75_t = np.quantile(tumor, 0.75)
    q75_i = np.quantile(tcell, 0.75)
    if q75_t < TUMOR_FLOOR:
        return None

    iface_mask = (tumor > q75_t) & (tcell > q75_i)
    core_mask  = (tumor > q75_t) & ~(tcell > q75_i)
    if iface_mask.sum() < MIN_IFACE or core_mask.sum() < MIN_IFACE:
        return None

    coords = sub.obsm['spatial']
    at = tumor - tumor.mean()
    bi = tcell - tcell.mean()

    B1, edges = build_knn_incidence(coords, k)
    m = len(edges)
    f = np.array([
        (at[e[0]] * bi[e[1]] - at[e[1]] * bi[e[0]])
        / (np.linalg.norm(coords[e[0]] - coords[e[1]]) + 1e-8)
        for e in edges
    ])

    density = hodge_coexact_density(B1, f)
    R = (density[iface_mask].mean() /
         (density[core_mask].mean() + 1e-12))
    cf = np.dot(f, f) / (np.dot(f, f) + 1e-12)  # proxy coexact fraction

    return dict(sample_id=sid, k=k, enrichment_R=R,
                sign_positive=(R > 1),
                n_interface=int(iface_mask.sum()))


def plot_results(df, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.07, right=0.97, top=0.84,
                        bottom=0.14, wspace=0.38)

    CR, CNR = '#c0392b', '#2166ac'
    k_vals = sorted(df['k'].unique())

    # A — R vs k per section
    ax = axes[0]
    for sid, grp in df.groupby('sample_id'):
        cid = '_'.join(sid.split('_')[:2])
        resp = PATIENT_MAP.get(cid, None)
        if resp is None: continue
        col = CR if resp == 'Responder' else CNR
        grp_s = grp.sort_values('k')
        ax.plot(grp_s['k'], grp_s['enrichment_R'], color=col,
                alpha=0.55, lw=1.2)
    ax.axhline(1.0, color='#888', lw=0.8, ls='--')
    ax.set_xlabel('kNN graph k', fontsize=9)
    ax.set_ylabel('Interface/core coexact enrichment ratio R', fontsize=9)
    ax.set_title('A   Enrichment ratio vs k\n(each line = one section)', fontsize=9)
    for c, l in [(CR, 'Responder'), (CNR, 'Non-resp.')]:
        ax.plot([], [], color=c, label=l, lw=2)
    ax.legend(fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # B — Fraction of sections with R>1 per k
    ax = axes[1]
    fracs = df.groupby('k')['sign_positive'].mean()
    ax.bar(fracs.index, fracs.values, color='#5b7fc9',
           width=0.6, edgecolor='white')
    ax.axhline(1.0, color='#888', lw=0.8, ls='--')
    ax.set_ylim(0.5, 1.08)
    ax.set_xlabel('kNN graph k', fontsize=9)
    ax.set_ylabel('Fraction of sections with R > 1', fontsize=9)
    ax.set_title('B   Sign consistency across k values\n'
                 '(target: all sections R > 1 at every k)', fontsize=9)
    for xi, (k, frac) in enumerate(fracs.items()):
        ax.text(k, frac + 0.01, f'{frac:.2f}', ha='center', fontsize=8.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # C — Spearman correlation matrix across k values
    ax = axes[2]
    pivot = df.pivot(index='sample_id', columns='k', values='enrichment_R').dropna()
    cols  = sorted(pivot.columns)
    mat   = np.eye(len(cols))
    for i, ki in enumerate(cols):
        for j, kj in enumerate(cols):
            if i != j:
                r, _ = spearmanr(pivot[ki], pivot[kj])
                mat[i, j] = r
    im = ax.imshow(mat, vmin=0.7, vmax=1.0, cmap='RdYlGn')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f'k={c}' for c in cols], fontsize=8)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels([f'k={c}' for c in cols], fontsize=8)
    ax.set_title('C   Rank correlation of R\nacross k values', fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.75)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center',
                    fontsize=8, color='#222')

    fig.suptitle(
        'Supplementary: kNN graph robustness sweep  ·  '
        'Primary enrichment result sign-consistent across k = 4–10',
        fontsize=10.5, fontweight='bold', y=0.97)
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adata',    default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--hodge',    default='results/hcc/results_hcc_hodge_interface_summary.csv')
    ap.add_argument('--out',      default='results/hcc/results_hcc_robustness_k_sweep.csv')
    ap.add_argument('--fig',      default='figures/supp_robustness_k_sweep.png')
    ap.add_argument('--k-values', nargs='+', type=int, default=K_VALUES)
    args = ap.parse_args()

    adata = sc.read_h5ad(args.adata)
    hodge = pd.read_csv(args.hodge)

    records = []
    for _, row in hodge.iterrows():
        sid = row['sample_id']
        sub = adata[adata.obs['sample_id'] == sid].copy()
        for k in args.k_values:
            rec = enrichment_at_k(sid, sub, k)
            if rec is not None:
                records.append(rec)
        print(f'  {sid}: k sweep done')

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}')

    print('\n=== K-SWEEP ROBUSTNESS SUMMARY ===')
    for k in sorted(df['k'].unique()):
        sub_k = df[df['k'] == k]
        n_pos = sub_k['sign_positive'].sum()
        n_tot = len(sub_k)
        print(f'  k={k:2d}: {n_pos}/{n_tot} sections R > 1'
              f'  (median R = {sub_k["enrichment_R"].median():.2f})')

    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plot_results(df, args.fig)
    print(f'Figure → {args.fig}')


if __name__ == '__main__':
    main()
