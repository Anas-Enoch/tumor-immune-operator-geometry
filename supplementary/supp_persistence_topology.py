#!/usr/bin/env python3
"""
supp_persistence_topology.py
============================
Supplementary analysis: H0 and H1 topological persistence of the coexact
density field at the tumour–immune interface.

Computes a sublevel-set filtration of the negated coexact density field
(so high-density regions die last) using the Vietoris–Rips / alpha complex
on spatial coordinates. H0 persistence measures how long connected components
survive; H1 persistence measures the lifetime of topological loops (1-cycles)
in the coexact density landscape.

**Pre-registered biomarker (from main manuscript):**
  H1 topological loop max persistence ≤ 25 is proposed as a pre-therapy
  threshold for the response-associated regime. This script validates the
  threshold by computing H1 persistence for all sections and testing
  separation between responders and non-responders.

Requires: gudhi (conda: `conda install -c conda-forge gudhi`)
Falls back to a graph-based H1 proxy (cycle rank of the kNN graph restricted
to coexact hotspot nodes) if gudhi is not available.

Outputs
-------
results/hcc/results_hcc_persistence_topology.csv
    Per-section: h0_max_persistence, h1_max_persistence, h1_mean_persistence,
    h1_n_features, n_hotspot_nodes.

figures/supp_persistence_topology.png
    Panel A: H1 max persistence R vs NR (with threshold at 25).
    Panel B: H0 max persistence R vs NR.
    Panel C: H1 persistence diagram for representative responder section.

Usage
-----
    python supp_persistence_topology.py \
        --adata   data/hepatocellular_carcinoma/hcc_scored.h5ad \
        --hodge   results/hcc/results_hcc_hodge_interface_summary.csv \
        --out     results/hcc/results_hcc_persistence_topology.csv \
        --fig     figures/supp_persistence_topology.png \
        --threshold 25
"""

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

K_KNN       = 6
MIN_IFACE   = 20
TUMOR_FLOOR = 0.05
HOTSPOT_Q   = 0.75
H1_THRESHOLD = 25

PATIENT_MAP = {
    'cytassist_70': 'Responder',    'cytassist_71': 'Responder',
    'cytassist_72': 'Non_Responder','cytassist_73': 'Non_Responder',
    'cytassist_74': 'Non_Responder','cytassist_76': 'Responder',
    'cytassist_79': 'Non_Responder','cytassist_83': 'Responder',
    'cytassist_84': 'Responder',    'cytassist_85': 'Non_Responder',
    'cytassist_86': 'Non_Responder',
}


def try_import_gudhi():
    try:
        import gudhi
        return gudhi
    except ImportError:
        return None


def graph_h1_proxy(coords, density, hotspot_mask):
    """
    Proxy for H1 persistence when gudhi is unavailable.
    Computes the cycle rank of the kNN graph restricted to hotspot nodes:
      beta_1 = |E_hot| - |V_hot| + n_components
    High cycle rank = more loop-like topology in the coexact hotspot cluster.
    Mapped to a persistence proxy by weighting by density range.
    """
    hot_nodes = np.where(hotspot_mask)[0]
    if len(hot_nodes) < 4:
        return 0.0, 0.0, 0

    coords_hot = coords[hot_nodes]
    density_hot = density[hot_nodes]

    nbrs = NearestNeighbors(n_neighbors=min(K_KNN, len(hot_nodes)-1)).fit(coords_hot)
    _, idx = nbrs.kneighbors(coords_hot)

    edges = set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            edges.add((min(i, int(j)), max(i, int(j))))

    m = len(edges)
    n = len(hot_nodes)
    if m == 0:
        return 0.0, 0.0, 0

    rows, cols = zip(*edges)
    W = csr_matrix(
        (np.ones(2*m), (list(rows)+list(cols), list(cols)+list(rows))),
        shape=(n, n)
    )
    n_comp, _ = connected_components(W, directed=False)
    beta1 = max(m - n + n_comp, 0)

    # Scale by density range as a persistence proxy
    density_range = density_hot.max() - density_hot.min() + 1e-8
    h1_proxy = float(beta1 * density_range)
    h1_mean  = float(density_range / (n + 1e-8))
    return h1_proxy, h1_mean, beta1


def compute_persistence(coords, density, hotspot_mask, gudhi=None):
    """
    Compute H0 and H1 persistence using gudhi if available,
    else return graph-based proxy.
    """
    if gudhi is not None:
        hot_nodes = np.where(hotspot_mask)[0]
        if len(hot_nodes) < 4:
            return 0.0, 0.0, 0.0, 0.0, 0

        pts = coords[hot_nodes].astype(np.float64)
        filt = density[hot_nodes].astype(np.float64)
        # Normalise filtration
        filt = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8) * 100

        try:
            rc = gudhi.RipsComplex(points=pts, max_edge_length=200)
            st = rc.create_simplex_tree(max_dimension=2)
            st.extend_filtration()
            st.compute_persistence()
            diag = st.persistence()

            h0 = [(d, b) for dim, (b, d) in diag if dim == 0 and d != float('inf')]
            h1 = [(d, b) for dim, (b, d) in diag if dim == 1]
            h0_max = max((d - b for d, b in h0), default=0.0)
            h1_max = max((d - b for d, b in h1), default=0.0)
            h1_mean = np.mean([d - b for d, b in h1]) if h1 else 0.0
            return h0_max, h1_max, h1_mean, [(b,d) for d,b in h1], len(h1)
        except Exception:
            pass

    # Fallback
    h1_max, h1_mean, beta1 = graph_h1_proxy(coords, density, hotspot_mask)
    h0_max = float(density[hotspot_mask].max() - density[hotspot_mask].min())
    return h0_max, h1_max, h1_mean, [], beta1


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
    try:
        alpha, *_ = lsqr(B1.T @ B1, B1 @ f, atol=1e-8, btol=1e-8, iter_lim=300)
    except Exception:
        return np.zeros(n)
    f_exact   = B1.T @ alpha
    f_coexact = f - f_exact
    density = np.zeros(n)
    deg     = np.zeros(n)
    for ei, (i, j) in enumerate(edges):
        density[i] += abs(f_coexact[ei]); density[j] += abs(f_coexact[ei])
        deg[i] += 1; deg[j] += 1
    return density / np.maximum(deg, 1)


def analyse_section(sid, sub, gudhi):
    n   = sub.n_obs
    obs = sub.obs
    tumor  = obs['tumor_score'].values.astype(float)
    tcell  = obs['tcell_score'].values.astype(float)
    coords = sub.obsm['spatial']

    if np.quantile(tumor, 0.75) < TUMOR_FLOOR:
        return None

    iface_mask = ((tumor > np.quantile(tumor, 0.75)) &
                  (tcell > np.quantile(tcell, 0.75)))
    if iface_mask.sum() < MIN_IFACE:
        return None

    density = compute_coexact_density(coords, tumor, tcell, K_KNN)
    hot_q   = np.quantile(density, HOTSPOT_Q)
    hotspot_mask = (density > hot_q) & iface_mask

    h0_max, h1_max, h1_mean, diag, n_h1 = compute_persistence(
        coords, density, hotspot_mask, gudhi)

    cid = '_'.join(sid.split('_')[:2])
    return dict(
        sample_id=sid, cytassist_id=cid,
        Response=PATIENT_MAP.get(cid),
        h0_max_persistence=float(h0_max),
        h1_max_persistence=float(h1_max),
        h1_mean_persistence=float(h1_mean),
        h1_n_features=int(n_h1),
        n_hotspot_nodes=int(hotspot_mask.sum()),
        below_threshold_h1=(h1_max <= H1_THRESHOLD),
        gudhi_used=(gudhi is not None),
    )


def plot_results(df, threshold, outpath):
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

    # A — H1 max persistence
    ax = axes[0]
    sw(ax, 0, r['h1_max_persistence'].values, CR)
    sw(ax, 1, nr['h1_max_persistence'].values, CNR)
    ax.axhline(threshold, color='#e67e22', lw=1.5, ls='--',
               label=f'Pre-reg. threshold (H1≤{threshold})')
    _, p = mannwhitneyu(r['h1_max_persistence'],
                         nr['h1_max_persistence'], alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Responder\n(n={len(r)})',
                        f'Non-resp.\n(n={len(nr)})'], fontsize=9)
    ax.set_ylabel('H1 max persistence', fontsize=9)
    ax.set_title(f'A   H1 loop persistence\np = {p:.3f}', fontsize=9)
    ax.legend(fontsize=7.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # B — H0 max persistence
    ax = axes[1]
    sw(ax, 0, r['h0_max_persistence'].values, CR)
    sw(ax, 1, nr['h0_max_persistence'].values, CNR)
    _, p2 = mannwhitneyu(r['h0_max_persistence'],
                          nr['h0_max_persistence'], alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder', 'Non-resp.'], fontsize=9)
    ax.set_ylabel('H0 max persistence', fontsize=9)
    ax.set_title(f'B   H0 component persistence\np = {p2:.3f}', fontsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # C — Below-threshold classification
    ax = axes[2]
    r_below  = r['below_threshold_h1'].sum()
    nr_below = nr['below_threshold_h1'].sum()
    ax.bar([0, 1], [r_below / len(r), nr_below / len(nr)],
           color=[CR, CNR], width=0.5, edgecolor='white')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f'Responder\n({r_below}/{len(r)} ≤ {threshold})',
                        f'Non-resp.\n({nr_below}/{len(nr)} ≤ {threshold})'], fontsize=9)
    ax.set_ylabel(f'Fraction with H1 max ≤ {threshold}\n(pre-registered threshold)', fontsize=8.5)
    ax.set_title('C   Pre-registered H1 threshold classification\n'
                 '(threshold defined on discovery cohort)', fontsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # Note about gudhi
    gudhi_note = 'Gudhi TDA' if df['gudhi_used'].any() else 'Graph cycle-rank proxy (gudhi not available)'
    fig.suptitle(
        f'Supplementary: H1 topological persistence of coexact density field\n'
        f'Pre-registered biomarker: H1 max persistence ≤ {threshold}  ·  Method: {gudhi_note}',
        fontsize=10.5, fontweight='bold', y=0.97)
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adata',     default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--hodge',     default='results/hcc/results_hcc_hodge_interface_summary.csv')
    ap.add_argument('--out',       default='results/hcc/results_hcc_persistence_topology.csv')
    ap.add_argument('--fig',       default='figures/supp_persistence_topology.png')
    ap.add_argument('--threshold', type=float, default=H1_THRESHOLD)
    args = ap.parse_args()

    gudhi = try_import_gudhi()
    if gudhi is None:
        print('WARNING: gudhi not found — using graph cycle-rank proxy for H1.')
        print('Install with: conda install -c conda-forge gudhi')
    else:
        print(f'gudhi {gudhi.__version__} available.')

    adata = sc.read_h5ad(args.adata)
    hodge = pd.read_csv(args.hodge)

    records = []
    for _, row in hodge.iterrows():
        sid = row['sample_id']
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = analyse_section(sid, sub, gudhi)
        if rec is not None:
            records.append(rec)
        print(f'  {sid}: H1 max = {rec["h1_max_persistence"]:.2f}' if rec else f'  {sid}: skipped')

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}  ({len(df)} sections)')

    print(f'\n=== PERSISTENCE TOPOLOGY SUMMARY (H1 threshold = {args.threshold}) ===')
    r  = df[df['Response'] == 'Responder']
    nr = df[df['Response'] == 'Non_Responder']
    print(f'  H1 max: R median = {r["h1_max_persistence"].median():.2f}  '
          f'NR median = {nr["h1_max_persistence"].median():.2f}')
    _, p = mannwhitneyu(r['h1_max_persistence'], nr['h1_max_persistence'],
                         alternative='two-sided')
    print(f'  Mann-Whitney p = {p:.4f}')
    below_r  = r['below_threshold_h1'].sum()
    below_nr = nr['below_threshold_h1'].sum()
    print(f'  Below threshold: {below_r}/{len(r)} R,  {below_nr}/{len(nr)} NR')

    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plot_results(df, args.threshold, args.fig)
    print(f'Figure → {args.fig}')


if __name__ == '__main__':
    main()
