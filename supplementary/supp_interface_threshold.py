#!/usr/bin/env python3
"""
supp_interface_threshold.py
===========================
Supplementary robustness analysis: sweep the interface annotation threshold
(quantile cutoff q for defining tumour-enriched and immune-enriched spots)
across {Q50, Q60, Q65, Q70, Q75, Q80, Q85, Q90} and re-evaluate the
primary spatial score and coexact enrichment separation between responders
and non-responders at each threshold.

This tests whether the main results depend on the specific Q75 threshold
used in the primary analysis or are robust across a reasonable range.

Outputs
-------
results/hcc/results_hcc_interface_threshold_sweep.csv
    Per-section × per-threshold: spatial_score, coexact_fraction,
    enrichment_R, n_interface.

figures/supp_interface_threshold.png
    Panel A: Spatial score Mann-Whitney p-value vs threshold.
    Panel B: Coexact enrichment R (median R vs NR) vs threshold.
    Panel C: Number of valid sections (≥20 interface nodes) vs threshold.

Usage
-----
    python supp_interface_threshold.py \
        --adata  data/hepatocellular_carcinoma/hcc_scored.h5ad \
        --hodge  results/hcc/results_hcc_hodge_interface_summary.csv \
        --out    results/hcc/results_hcc_interface_threshold_sweep.csv \
        --fig    figures/supp_interface_threshold.png
"""

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import lsqr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

K_KNN       = 6
MIN_IFACE   = 20
TUMOR_FLOOR = 0.05
THRESHOLDS  = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

PATIENT_MAP = {
    'cytassist_70': 'Responder',    'cytassist_71': 'Responder',
    'cytassist_72': 'Non_Responder','cytassist_73': 'Non_Responder',
    'cytassist_74': 'Non_Responder','cytassist_76': 'Responder',
    'cytassist_79': 'Non_Responder','cytassist_83': 'Responder',
    'cytassist_84': 'Responder',    'cytassist_85': 'Non_Responder',
    'cytassist_86': 'Non_Responder',
}


def build_knn_edges(coords, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    edges = set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            edges.add((min(i, int(j)), max(i, int(j))))
    return list(edges)


def section_at_threshold(sid, sub, q):
    n  = sub.n_obs
    obs = sub.obs
    tumor  = obs['tumor_score'].values.astype(float)
    tcell  = obs['tcell_score'].values.astype(float)
    coords = sub.obsm['spatial']

    q75_t = np.quantile(tumor, 0.75)
    if q75_t < TUMOR_FLOOR:
        return None

    q_t = np.quantile(tumor, q)
    q_i = np.quantile(tcell, q)

    iface_mask = (tumor > q_t) & (tcell > q_i)
    core_mask  = (tumor > q_t) & ~(tcell > q_i)

    if iface_mask.sum() < MIN_IFACE or core_mask.sum() < 5:
        return dict(sample_id=sid, threshold=q,
                    valid=False, n_interface=int(iface_mask.sum()))

    edges = build_knn_edges(coords, K_KNN)
    at = tumor - np.mean(tumor[iface_mask | core_mask])
    bi = tcell - np.mean(tcell[iface_mask | core_mask])

    f = np.array([
        (at[e[0]] * bi[e[1]] - at[e[1]] * bi[e[0]])
        / (np.linalg.norm(coords[e[0]] - coords[e[1]]) + 1e-8)
        for e in edges
    ])

    # Build B1 for Hodge
    m = len(edges)
    B1 = np.zeros((m, n))
    for ei, (i, j) in enumerate(edges):
        B1[ei, i] = -1; B1[ei, j] = 1
    L0 = B1.T @ B1
    try:
        alpha, *_ = lsqr(L0, B1 @ f, atol=1e-8, btol=1e-8, iter_lim=300)
    except Exception:
        return None
    f_exact   = B1.T @ alpha
    f_coexact = f - f_exact

    coexact_frac = np.dot(f_coexact, f_coexact) / (np.dot(f, f) + 1e-12)

    # Node-level coexact density
    density = np.zeros(n)
    for ei, (i, j) in enumerate(edges):
        density[i] += abs(f_coexact[ei])
        density[j] += abs(f_coexact[ei])
    deg = np.zeros(n)
    for i, j in edges:
        deg[i] += 1; deg[j] += 1
    density /= np.maximum(deg, 1)

    R = density[iface_mask].mean() / (density[core_mask].mean() + 1e-12)

    # Spatial score
    if iface_mask.sum() >= 10:
        hotspot = density > np.quantile(density[iface_mask | core_mask], 0.75)
        from scipy.sparse.csgraph import connected_components
        from scipy.sparse import csr_matrix
        rows, cols_idx = zip(*edges) if edges else ([], [])
        adj = csr_matrix(
            (np.ones(2*len(edges)),
             (list(rows)+list(cols_idx), list(cols_idx)+list(rows))),
            shape=(n, n))
        hot_nodes = np.where(hotspot)[0]
        if len(hot_nodes) > 1:
            adj_hot = adj[hot_nodes][:, hot_nodes]
            n_comp, labels = connected_components(adj_hot, directed=False)
            comp_sizes = np.bincount(labels)
            lcc = comp_sizes.max() / len(hot_nodes)
        else:
            lcc = 0.0
        if lcc > 0:
            counts = np.bincount(np.digitize(density[hotspot],
                                             np.linspace(density[hotspot].min(),
                                                         density[hotspot].max(), 10)))
            counts = counts[counts > 0].astype(float) / counts[counts > 0].sum()
            gini = 1 - (counts ** 2).sum()
        else:
            gini = 0.0
        spatial_score = lcc * gini
    else:
        spatial_score = 0.0

    cid = '_'.join(sid.split('_')[:2])
    return dict(
        sample_id=sid, cytassist_id=cid,
        threshold=q, valid=True,
        Response=PATIENT_MAP.get(cid),
        n_interface=int(iface_mask.sum()),
        coexact_fraction=float(coexact_frac),
        enrichment_R=float(R),
        spatial_score=float(spatial_score),
    )


def plot_results(df, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.08, right=0.97, top=0.84,
                        bottom=0.14, wspace=0.38)

    valid_df = df[df['valid']]
    thresholds = sorted(valid_df['threshold'].unique())

    # A — p-value of spatial score R vs NR at each threshold
    ax = axes[0]
    p_vals, sig = [], []
    for q in thresholds:
        sub = valid_df[valid_df['threshold'] == q]
        r  = sub[sub['Response'] == 'Responder']['spatial_score']
        nr = sub[sub['Response'] == 'Non_Responder']['spatial_score']
        if len(r) >= 2 and len(nr) >= 2:
            _, p = mannwhitneyu(r, nr, alternative='two-sided')
        else:
            p = 1.0
        p_vals.append(p)
        sig.append(p < 0.05)

    cols = ['#e74c3c' if s else '#95a5a6' for s in sig]
    ax.bar([f'Q{int(q*100)}' for q in thresholds], p_vals,
           color=cols, edgecolor='white')
    ax.axhline(0.05, color='#888', lw=0.9, ls='--', label='p=0.05')
    ax.set_ylabel('Mann-Whitney p-value\n(spatial score, R vs NR)', fontsize=9)
    ax.set_title('A   Spatial score significance\nacross annotation thresholds', fontsize=9)
    ax.set_xlabel('Interface annotation quantile', fontsize=9)
    ax.legend(fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    ax.axvline(thresholds.index(0.75), color='#2166ac',
               lw=1.5, ls=':', alpha=0.7, label='Primary (Q75)')

    # B — Median enrichment R per group vs threshold
    ax = axes[1]
    med_r  = valid_df[valid_df['Response']=='Responder'].groupby('threshold')['enrichment_R'].median()
    med_nr = valid_df[valid_df['Response']=='Non_Responder'].groupby('threshold')['enrichment_R'].median()
    ax.plot([f'Q{int(q*100)}' for q in med_r.index], med_r.values,
            'o-', color='#c0392b', label='Responder median', lw=2)
    ax.plot([f'Q{int(q*100)}' for q in med_nr.index], med_nr.values,
            's-', color='#2166ac', label='Non-resp. median', lw=2)
    ax.axhline(1.0, color='#888', lw=0.8, ls='--')
    ax.set_ylabel('Median enrichment ratio R', fontsize=9)
    ax.set_title('B   Enrichment ratio by group\nacross thresholds', fontsize=9)
    ax.set_xlabel('Interface annotation quantile', fontsize=9)
    ax.legend(fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # C — Valid section count vs threshold
    ax = axes[2]
    n_valid = valid_df[valid_df['valid']].groupby('threshold')['sample_id'].count()
    ax.bar([f'Q{int(q*100)}' for q in n_valid.index], n_valid.values,
           color='#5b7fc9', edgecolor='white')
    ax.axhline(15, color='#e74c3c', lw=1, ls='--', label='Full cohort (15)')
    ax.set_ylabel('Valid sections (n_interface ≥ 20)', fontsize=9)
    ax.set_title('C   Valid section count\n(interface size constraint)', fontsize=9)
    ax.set_xlabel('Interface annotation quantile', fontsize=9)
    ax.legend(fontsize=8)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
    for xi, (q, n) in enumerate(n_valid.items()):
        ax.text(xi, n + 0.2, str(int(n)), ha='center', fontsize=8.5)

    fig.suptitle(
        'Supplementary: Interface annotation threshold sensitivity\n'
        'Primary result (spatial score p=0.006) shown at Q75 (blue dotted line)',
        fontsize=10.5, fontweight='bold', y=0.97)
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adata', default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--hodge', default='results/hcc/results_hcc_hodge_interface_summary.csv')
    ap.add_argument('--out',   default='results/hcc/results_hcc_interface_threshold_sweep.csv')
    ap.add_argument('--fig',   default='figures/supp_interface_threshold.png')
    args = ap.parse_args()

    adata = sc.read_h5ad(args.adata)
    hodge = pd.read_csv(args.hodge)

    records = []
    for _, row in hodge.iterrows():
        sid = row['sample_id']
        sub = adata[adata.obs['sample_id'] == sid].copy()
        for q in THRESHOLDS:
            rec = section_at_threshold(sid, sub, q)
            if rec is not None:
                records.append(rec)
        print(f'  {sid}: threshold sweep done')

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}')

    print('\n=== THRESHOLD SENSITIVITY SUMMARY ===')
    for q in THRESHOLDS:
        sub = df[(df['threshold'] == q) & df.get('valid', pd.Series([True]*len(df)))]
        n_pos = (sub['enrichment_R'] > 1).sum() if 'enrichment_R' in sub else 0
        print(f'  Q{int(q*100)}: {n_pos}/{len(sub)} sections R>1')

    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plot_results(df, args.fig)
    print(f'Figure → {args.fig}')


if __name__ == '__main__':
    main()
