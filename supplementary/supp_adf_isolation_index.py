#!/usr/bin/env python3
"""
supp_adf_isolation_index.py
============================
Supplementary analysis: ADF (Augmented Distance Function) isolation index.

Measures how spatially isolated the coexact hotspot cluster is from the
tumour core centroid. A high isolation index indicates that high-coexact
nodes concentrate at the tumour periphery (interface zone) rather than
distributing throughout the tumour interior.

Two metrics are computed:
1. **ADF isolation index**: for each coexact hotspot node, its normalized
   distance from the tumour core centroid. High mean distance = peripherally
   confined coexact organisation.

2. **Boundary confinement score**: fraction of coexact hotspot nodes that
   fall within the annotated interface zone versus the tumour core. Measures
   whether high-coexact nodes are boundary-confined or distributed.

The pre-registered hypothesis is that responder interfaces show higher
isolation index (coexact organisation more peripherally confined) than
non-responder interfaces.

Outputs
-------
results/hcc/results_hcc_adf_isolation.csv
    Per-section: adf_isolation_index, boundary_confinement, hotspot_density_ratio.

figures/supp_adf_isolation.png
    Panel A: ADF isolation index R vs NR.
    Panel B: Boundary confinement score R vs NR.
    Panel C: Spatial map of representative section (ADF colour-coded).

Usage
-----
    python supp_adf_isolation_index.py \
        --adata  data/hepatocellular_carcinoma/hcc_scored.h5ad \
        --hodge  results/hcc/results_hcc_hodge_interface_summary.csv \
        --out    results/hcc/results_hcc_adf_isolation.csv \
        --fig    figures/supp_adf_isolation.png
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
HOTSPOT_Q   = 0.75

PATIENT_MAP = {
    'cytassist_70': 'Responder',    'cytassist_71': 'Responder',
    'cytassist_72': 'Non_Responder','cytassist_73': 'Non_Responder',
    'cytassist_74': 'Non_Responder','cytassist_76': 'Responder',
    'cytassist_79': 'Non_Responder','cytassist_83': 'Responder',
    'cytassist_84': 'Responder',    'cytassist_85': 'Non_Responder',
    'cytassist_86': 'Non_Responder',
}


def compute_coexact_density(coords, tumor, tcell, k):
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    edges = set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            edges.add((min(i, int(j)), max(i, int(j))))
    edges = list(edges)
    m = len(edges)

    at = tumor - tumor.mean()
    bi = tcell - tcell.mean()
    f = np.array([
        (at[e[0]] * bi[e[1]] - at[e[1]] * bi[e[0]])
        / (np.linalg.norm(coords[e[0]] - coords[e[1]]) + 1e-8)
        for e in edges
    ])

    B1 = np.zeros((m, n))
    for ei, (i, j) in enumerate(edges):
        B1[ei, i] = -1; B1[ei, j] = 1
    L0 = B1.T @ B1
    try:
        alpha, *_ = lsqr(L0, B1 @ f, atol=1e-8, btol=1e-8, iter_lim=300)
    except Exception:
        return np.zeros(n)
    f_exact   = B1.T @ alpha
    f_coexact = f - f_exact

    density = np.zeros(n)
    deg     = np.zeros(n)
    for ei, (i, j) in enumerate(edges):
        density[i] += abs(f_coexact[ei])
        density[j] += abs(f_coexact[ei])
        deg[i] += 1; deg[j] += 1
    return density / np.maximum(deg, 1)


def analyse_section(sid, sub):
    n   = sub.n_obs
    obs = sub.obs
    tumor  = obs['tumor_score'].values.astype(float)
    tcell  = obs['tcell_score'].values.astype(float)
    coords = sub.obsm['spatial']

    q75_t = np.quantile(tumor, 0.75)
    q75_i = np.quantile(tcell, 0.75)
    if q75_t < TUMOR_FLOOR:
        return None

    iface_mask = (tumor > q75_t) & (tcell > q75_i)
    core_mask  = (tumor > q75_t) & ~(tcell > q75_i)
    if iface_mask.sum() < MIN_IFACE or core_mask.sum() < 5:
        return None

    density = compute_coexact_density(coords, tumor, tcell, K_KNN)

    hotspot_q   = np.quantile(density, HOTSPOT_Q)
    hotspot_mask = density > hotspot_q

    # ADF isolation: mean distance of hotspot nodes from tumour core centroid
    core_centroid = coords[core_mask].mean(axis=0)
    all_dists     = np.linalg.norm(coords - core_centroid, axis=1)
    # Normalise by section diameter
    section_radius = all_dists.max() + 1e-8

    adf_hotspot = all_dists[hotspot_mask] / section_radius
    adf_all     = all_dists / section_radius
    adf_index   = float(adf_hotspot.mean() - adf_all.mean())

    # Boundary confinement: fraction of hotspot nodes in interface zone
    n_hot_iface = (hotspot_mask & iface_mask).sum()
    n_hot_core  = (hotspot_mask & core_mask).sum()
    n_hot_total = hotspot_mask.sum()
    boundary_conf = float(n_hot_iface / (n_hot_total + 1e-8))

    # Hotspot density ratio: mean density in interface vs core
    hdr = (density[iface_mask].mean() /
           (density[core_mask].mean() + 1e-12))

    cid = '_'.join(sid.split('_')[:2])
    return dict(
        sample_id=sid, cytassist_id=cid,
        Response=PATIENT_MAP.get(cid),
        adf_isolation_index=adf_index,
        boundary_confinement=boundary_conf,
        hotspot_density_ratio=float(hdr),
        n_interface=int(iface_mask.sum()),
        n_hotspot=int(n_hot_total),
        n_hotspot_iface=int(n_hot_iface),
        # Store for plotting
        _density=density.tolist(),
        _coords=coords.tolist(),
        _iface=iface_mask.tolist(),
    )


def plot_results(df, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.07, right=0.97, top=0.84,
                        bottom=0.14, wspace=0.38)
    CR, CNR = '#c0392b', '#2166ac'

    def sw(ax, xp, v, c, s=60):
        rng2 = np.random.RandomState(42)
        jx = xp + rng2.uniform(-0.15, 0.15, len(v))
        ax.scatter(jx, v, color=c, s=s, zorder=3, alpha=0.88,
                   edgecolors='white', lw=0.3)
        ax.plot([xp-0.25, xp+0.25], [np.median(v)]*2,
                color=c, lw=2.5, zorder=4)

    r  = df[df['Response'] == 'Responder']
    nr = df[df['Response'] == 'Non_Responder']

    # A — ADF isolation index
    ax = axes[0]
    sw(ax, 0, r['adf_isolation_index'].values, CR)
    sw(ax, 1, nr['adf_isolation_index'].values, CNR)
    ax.axhline(0, color='#888', lw=0.8, ls='--',
               label='No bias (index=0)')
    _, p = mannwhitneyu(r['adf_isolation_index'],
                         nr['adf_isolation_index'], alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Responder\n(n={len(r)})',
                        f'Non-resp.\n(n={len(nr)})'], fontsize=9)
    ax.set_ylabel('ADF isolation index\n(hotspot dist − mean dist; normalised)', fontsize=8.5)
    ax.set_title(f'A   ADF isolation index\np = {p:.3f}', fontsize=9)
    ax.legend(fontsize=7.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # B — Boundary confinement score
    ax = axes[1]
    sw(ax, 0, r['boundary_confinement'].values, CR)
    sw(ax, 1, nr['boundary_confinement'].values, CNR)
    _, p2 = mannwhitneyu(r['boundary_confinement'],
                          nr['boundary_confinement'], alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder', 'Non-resp.'], fontsize=9)
    ax.set_ylabel('Boundary confinement score\n(fraction of hotspots in interface zone)', fontsize=8.5)
    ax.set_title(f'B   Boundary confinement\np = {p2:.3f}', fontsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # C — Hotspot density ratio (same as main enrichment R, for comparison)
    ax = axes[2]
    sw(ax, 0, r['hotspot_density_ratio'].values, CR)
    sw(ax, 1, nr['hotspot_density_ratio'].values, CNR)
    ax.axhline(1.0, color='#888', lw=0.8, ls='--')
    _, p3 = mannwhitneyu(r['hotspot_density_ratio'],
                          nr['hotspot_density_ratio'], alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder', 'Non-resp.'], fontsize=9)
    ax.set_ylabel('Hotspot density ratio\n(interface / core)', fontsize=8.5)
    ax.set_title(f'C   Hotspot density ratio\np = {p3:.3f}', fontsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    fig.suptitle(
        'Supplementary: ADF isolation index — coexact hotspot peripheral confinement\n'
        'Tests whether responder coexact clusters are more boundary-confined than non-responder clusters',
        fontsize=10.5, fontweight='bold', y=0.97)
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adata', default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--hodge', default='results/hcc/results_hcc_hodge_interface_summary.csv')
    ap.add_argument('--out',   default='results/hcc/results_hcc_adf_isolation.csv')
    ap.add_argument('--fig',   default='figures/supp_adf_isolation.png')
    args = ap.parse_args()

    adata = sc.read_h5ad(args.adata)
    hodge = pd.read_csv(args.hodge)

    records = []
    for _, row in hodge.iterrows():
        sid = row['sample_id']
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = analyse_section(sid, sub)
        if rec is not None:
            # Drop large lists before saving
            rec_clean = {k: v for k, v in rec.items() if not k.startswith('_')}
            records.append(rec_clean)
        print(f'  {sid}: done')

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}  ({len(df)} sections)')

    print('\n=== ADF ISOLATION SUMMARY ===')
    r  = df[df['Response'] == 'Responder']
    nr = df[df['Response'] == 'Non_Responder']
    for metric in ['adf_isolation_index', 'boundary_confinement', 'hotspot_density_ratio']:
        _, p = mannwhitneyu(r[metric], nr[metric], alternative='two-sided')
        print(f'  {metric:<28}  R={r[metric].median():.3f}  '
              f'NR={nr[metric].median():.3f}  p={p:.4f}')

    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plot_results(df, args.fig)
    print(f'Figure → {args.fig}')


if __name__ == '__main__':
    main()
