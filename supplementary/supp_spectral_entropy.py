#!/usr/bin/env python3
"""
supp_spectral_entropy.py
========================
Supplementary analysis: Zeta spectral concentration of the coexact density field.

For each section, projects the per-node coexact density signal onto eigenvectors
of the node graph Laplacian and computes the normalized Zeta functional

    Z(s) = Σ_k α_k λ_k^{-s}  /  Σ_k α_k,    α_k = <u, φ_k>²

where φ_k are eigenvectors and λ_k eigenvalues of the node Laplacian.
Z(s) > 1 indicates energy biased toward spatially smooth low-frequency modes.

A label-permutation null (1000 shuffles of node labels, preserving graph topology)
is used to test whether the global coexact signal is spectrally non-flat.

Pre-registered claim: coexact density is not spectrally flat (consistent with
locally structured interface organisation rather than random noise).

Outputs
-------
results/hcc/results_hcc_spectral_entropy.csv
    Per-section: Z(s=1), Z(s=2), spectral_gini, f_low (low-freq fraction),
    null_median_Zs1, null_p_Zs1, section-level metadata.

figures/supp_spectral_entropy.png
    Panel A: Z(s=1) vs null distribution (violin).
    Panel B: Low-frequency fraction f_low across sections, R vs NR.
    Panel C: Spectral Gini coefficient, R vs NR.

Usage
-----
    python supp_spectral_entropy.py \
        --adata   data/hepatocellular_carcinoma/hcc_scored.h5ad \
        --hodge   results/hcc/results_hcc_hodge_interface_summary.csv \
        --out     results/hcc/results_hcc_spectral_entropy.csv \
        --fig     figures/supp_spectral_entropy.png \
        --n-perm  1000 \
        --n-modes 50
"""

import argparse, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu
from sklearn.neighbors import NearestNeighbors

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

K_KNN    = 6
TUMOR_FLOOR = 0.05
MIN_IFACE   = 20
N_MODES  = 50
N_PERM   = 1000


PATIENT_MAP = {
    'cytassist_70': 'Responder',    'cytassist_71': 'Responder',
    'cytassist_72': 'Non_Responder','cytassist_73': 'Non_Responder',
    'cytassist_74': 'Non_Responder','cytassist_76': 'Responder',
    'cytassist_79': 'Non_Responder','cytassist_83': 'Responder',
    'cytassist_84': 'Responder',    'cytassist_85': 'Non_Responder',
    'cytassist_86': 'Non_Responder',
}


def build_knn(coords, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    rows, cols, data = [], [], []
    for i, row in enumerate(idx):
        for j in row[1:]:
            rows += [i, int(j)]; cols += [int(j), i]; data += [1.0, 1.0]
    n = len(coords)
    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W


def node_laplacian(W):
    """Return sparse node Laplacian L = D - W."""
    from scipy.sparse import diags
    d = np.array(W.sum(axis=1)).flatten()
    D = diags(d)
    return D - W


def zeta_functional(signal, eigvecs, eigvals, s=1.0):
    """Normalized Zeta functional Z(s)."""
    alphas = (eigvecs.T @ signal) ** 2
    total = alphas.sum()
    if total < 1e-12:
        return np.nan
    weights = alphas / total
    lam = np.maximum(eigvals, 1e-8)
    return float((weights * lam ** (-s)).sum())


def spectral_gini(alphas):
    """Gini coefficient of spectral energy distribution."""
    a = np.sort(np.abs(alphas))
    n = len(a)
    if n < 2 or a.sum() < 1e-12:
        return np.nan
    idx = np.arange(1, n + 1)
    return float((2 * (idx * a).sum()) / (n * a.sum()) - (n + 1) / n)


def analyse_section(sid, sub, n_modes, n_perm, rng):
    n = sub.n_obs
    obs = sub.obs
    tumor = obs['tumor_score'].values.astype(float)
    tcell = obs['tcell_score'].values.astype(float)

    if np.quantile(tumor, 0.75) < TUMOR_FLOOR:
        return None

    coords = sub.obsm['spatial']
    W = build_knn(coords, K_KNN)
    L = node_laplacian(W)

    # Coexact density (use pre-computed if available, else proxy from wedge)
    if 'coexact_density' in obs.columns:
        u = obs['coexact_density'].values.astype(float)
    else:
        # Proxy: wedge magnitude as coexact proxy
        a = tumor - tumor.mean()
        b = tcell - tcell.mean()
        u = np.abs(a * b)

    if u.std() < 1e-10:
        return None

    u = (u - u.mean()) / (u.std() + 1e-10)

    # Eigen-decomposition of node Laplacian
    k = min(n_modes, n - 2)
    try:
        vals, vecs = eigsh(L, k=k, which='SM', tol=1e-6)
    except Exception:
        return None

    order = np.argsort(vals)
    vals = vals[order]; vecs = vecs[:, order]
    vals = np.maximum(vals, 0)

    # Spectral metrics
    Zs1 = zeta_functional(u, vecs, vals, s=1.0)
    Zs2 = zeta_functional(u, vecs, vals, s=2.0)

    alphas = (vecs.T @ u) ** 2
    half   = len(alphas) // 2
    f_low  = alphas[:half].sum() / (alphas.sum() + 1e-12)
    gini   = spectral_gini(alphas)

    # Permutation null for Z(s=1)
    null_Zs1 = []
    for _ in range(n_perm):
        u_perm = rng.permutation(u)
        null_Zs1.append(zeta_functional(u_perm, vecs, vals, s=1.0))
    null_Zs1 = np.array(null_Zs1)
    null_med = float(np.median(null_Zs1))
    p_Zs1 = float((null_Zs1 >= Zs1).mean())

    cid = '_'.join(sid.split('_')[:2])
    return dict(
        sample_id=sid, cytassist_id=cid,
        Response=PATIENT_MAP.get(cid),
        Zs1=Zs1, Zs2=Zs2, f_low=f_low, gini=gini,
        null_med_Zs1=null_med, p_Zs1=p_Zs1,
        n_nodes=n,
    )


def plot_results(df, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')
    plt.subplots_adjust(left=0.07, right=0.97, top=0.84,
                        bottom=0.14, wspace=0.38)

    CR, CNR = '#c0392b', '#2166ac'

    def sw(ax, xp, v, c):
        rng = np.random.RandomState(42)
        jx = xp + rng.uniform(-0.15, 0.15, len(v))
        ax.scatter(jx, v, color=c, s=55, zorder=3, alpha=0.88,
                   edgecolors='white', lw=0.3)
        ax.plot([xp - 0.25, xp + 0.25], [np.median(v)] * 2,
                color=c, lw=2.5, zorder=4)

    # A — Z(s=1) vs null
    ax = axes[0]
    r = df[df['Response'] == 'Responder']['Zs1'].dropna()
    nr = df[df['Response'] == 'Non_Responder']['Zs1'].dropna()
    sw(ax, 0, r.values, CR)
    sw(ax, 1, nr.values, CNR)
    ax.axhline(1.0, color='#888', lw=0.8, ls='--', label='Flat spectrum (Z=1)')
    _, p = mannwhitneyu(r, nr, alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder\n(n=%d)' % len(r),
                        'Non-resp.\n(n=%d)' % len(nr)], fontsize=9)
    ax.set_ylabel('Z(s=1) normalized Zeta functional', fontsize=9)
    ax.set_title('A   Spectral concentration Z(s=1)\n'
                 f'R vs NR p = {p:.3f}', fontsize=9)
    ax.legend(fontsize=7.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # B — Low-frequency fraction
    ax = axes[1]
    r2 = df[df['Response'] == 'Responder']['f_low'].dropna()
    nr2 = df[df['Response'] == 'Non_Responder']['f_low'].dropna()
    sw(ax, 0, r2.values, CR)
    sw(ax, 1, nr2.values, CNR)
    ax.axhline(0.5, color='#888', lw=0.8, ls='--', label='Uniform (f_low=0.5)')
    _, p2 = mannwhitneyu(r2, nr2, alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder', 'Non-resp.'], fontsize=9)
    ax.set_ylabel('Low-frequency fraction (f_low)', fontsize=9)
    ax.set_title('B   Low-frequency spectral fraction\n'
                 f'p = {p2:.3f}', fontsize=9)
    ax.legend(fontsize=7.5)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    # C — Spectral Gini
    ax = axes[2]
    r3 = df[df['Response'] == 'Responder']['gini'].dropna()
    nr3 = df[df['Response'] == 'Non_Responder']['gini'].dropna()
    sw(ax, 0, r3.values, CR)
    sw(ax, 1, nr3.values, CNR)
    _, p3 = mannwhitneyu(r3, nr3, alternative='two-sided')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Responder', 'Non-resp.'], fontsize=9)
    ax.set_ylabel('Spectral Gini coefficient', fontsize=9)
    ax.set_title('C   Spectral energy concentration (Gini)\n'
                 f'p = {p3:.3f}', fontsize=9)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

    fig.suptitle(
        'Supplementary: Zeta spectral concentration of coexact density field\n'
        'Tests whether coexact signal is spectrally non-flat '
        '(locally structured vs random noise)',
        fontsize=10.5, fontweight='bold', y=0.97)
    plt.savefig(outpath, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adata',  default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--hodge',  default='results/hcc/results_hcc_hodge_interface_summary.csv')
    ap.add_argument('--out',    default='results/hcc/results_hcc_spectral_entropy.csv')
    ap.add_argument('--fig',    default='figures/supp_spectral_entropy.png')
    ap.add_argument('--n-perm', type=int, default=N_PERM)
    ap.add_argument('--n-modes',type=int, default=N_MODES)
    args = ap.parse_args()

    adata = sc.read_h5ad(args.adata)
    hodge = pd.read_csv(args.hodge)
    rng   = np.random.RandomState(42)

    records = []
    for _, row in hodge.iterrows():
        sid = row['sample_id']
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = analyse_section(sid, sub, args.n_modes, args.n_perm, rng)
        if rec is not None:
            records.append(rec)
        print(f'  {sid}: done')

    df = pd.DataFrame(records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}  ({len(df)} sections)')

    # Summary
    print('\n=== SPECTRAL ENTROPY SUMMARY ===')
    above_null = (df['Zs1'] > df['null_med_Zs1']).sum()
    print(f'Z(s=1) > null median: {above_null}/{len(df)} sections')
    print(f'Median Z(s=1): {df["Zs1"].median():.3f}')
    print(f'Median f_low: {df["f_low"].median():.3f}  (null ≈ 0.50)')

    Path(args.fig).parent.mkdir(parents=True, exist_ok=True)
    plot_results(df, args.fig)
    print(f'Figure → {args.fig}')


if __name__ == '__main__':
    main()
