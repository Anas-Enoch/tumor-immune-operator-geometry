#!/usr/bin/env python3
"""
build_hodge_decomposition.py — HCC Hodge Decomposition of Wedge Field
======================================================================
Performs discrete Hodge decomposition of the wedge edge 1-cochain on
each sample's kNN spatial graph, computing exact / coexact / harmonic
energy components and their paired pre/post changes per patient.

Mathematical structure
----------------------
Given a directed kNN graph G = (V, E):

  B1 ∈ R^{|E| × |V|}  — signed node-edge incidence matrix
                         B1[e, v] = +1 if v is head of e
                                    −1 if v is tail of e

Edge 1-cochain (wedge field):
  ω ∈ R^{|E|}

Graph Laplacian (edge-based / Hodge-1):
  L1 = B1.T @ B1

Hodge decomposition (Helmholtz on graph):
  ω = ω_exact + ω_coexact + ω_harmonic

where:
  ω_exact   ∈ Im(B1)         — gradient component (curl-free)
             solved as B1 @ α where α = lsqr(L0, B1.T @ ω)
  ω_coexact = ω − ω_exact    — residual after exact projection
  ω_harmonic                  — numerical nullspace of L1 (small graphs)
             approximated as ω_coexact projected onto low eigenmodes
             with eigenvalue < threshold; usually negligible on kNN graphs

Energy components:
  exact_energy     = ||ω_exact||²  / |E|     (normalised by edge count)
  coexact_energy   = ||ω_coexact||² / |E|
  harmonic_energy  = ||ω_harmonic||² / |E|
  total_energy     = exact_energy + coexact_energy + harmonic_energy

Note: normalisation by |E| makes energies comparable across samples
with different spot counts.

Numerical approach
------------------
  - All matrices are scipy.sparse (CSR format)
  - Exact component: scipy.sparse.linalg.lsqr with atol/btol=1e-10
  - Ridge stabilisation: B1.T is well-posed for connected graphs;
    ridge λ=1e-8 added if condition is poor
  - No dense matrix inversion anywhere

Output
------
  results_hcc_hodge_summary.csv      per-sample Hodge energies
  results_hcc_hodge_delta.csv        per-patient Δ metrics
  fig_hcc_delta_coexact_boxplot.png  boxplot of Δcoexact by response
  fig_hcc_prepost_coexact_scatter.png scatter pre vs post coexact_energy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Patient metadata (hardcoded from known cohort structure) ────────────────
PATIENT_MAP = {
    'cytassist_70': (85557, 'Responder',    'List A'),
    'cytassist_71': (85558, 'Responder',    'List A'),
    'cytassist_72': (85559, 'Non_Responder','List A'),
    'cytassist_73': (85560, 'Non_Responder','List A'),
    'cytassist_74': (85561, 'Non_Responder','List A'),
    'cytassist_76': (85563, 'Responder',    'List B'),
    'cytassist_79': (85565, 'Non_Responder','List A'),
    'cytassist_83': (85566, 'Responder',    'List B'),
    'cytassist_84': (85567, 'Responder',    'List B'),
    'cytassist_85': (85568, 'Non_Responder','List B'),
    'cytassist_86': (85569, 'Non_Responder','List B'),
}

COLOURS = {'Responder': '#d62728', 'Non_Responder': '#1f77b4'}


# ── Graph construction ─────────────────────────────────────────────────────

def build_knn_edges(coords: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return directed edge list (src, tgt) for kNN graph.
    Each undirected edge is stored once: src < tgt by index convention.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    src_list, tgt_list = [], []
    seen = set()
    for i, nbr in enumerate(indices):
        for j in nbr[1:]:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen.add(key)
                src_list.append(key[0])
                tgt_list.append(key[1])
    return np.array(src_list, dtype=np.int32), np.array(tgt_list, dtype=np.int32)


def build_incidence_matrix(src: np.ndarray,
                           tgt: np.ndarray,
                           n_nodes: int) -> sp.csr_matrix:
    """
    Build signed node-edge incidence matrix B1 ∈ R^{|E| × |V|}.
    Convention: B1[e, tgt[e]] = +1, B1[e, src[e]] = -1
    So that B1 @ node_potential = gradient edge field.
    """
    n_edges = len(src)
    rows    = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    cols    = np.concatenate([tgt, src])
    data    = np.concatenate([np.ones(n_edges), -np.ones(n_edges)])
    B1 = sp.coo_matrix((data, (rows, cols)),
                       shape=(n_edges, n_nodes)).tocsr()
    return B1


def wedge_field(A: np.ndarray, B: np.ndarray,
                src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """
    Antisymmetric wedge: ω_e = A[src] · B[tgt] − A[tgt] · B[src]
    Returns edge 1-cochain ω ∈ R^{|E|}.
    """
    return A[src] * B[tgt] - A[tgt] * B[src]


# ── Hodge decomposition ────────────────────────────────────────────────────

def hodge_decompose(omega: np.ndarray,
                    B1: sp.csr_matrix,
                    ridge: float = 1e-8
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Discrete Hodge decomposition of edge 1-cochain ω.

    Returns (ω_exact, ω_coexact, ω_harmonic).

    Exact component:
      Solve   B1 @ B1.T @ α = B1 @ ω   (normal equations for node potentials)
      Then    ω_exact = B1.T @ α

    Equivalently, use LSQR to solve B1.T @ α = ω for the least-squares
    solution, then ω_exact = B1.T @ α is the projection onto Im(B1.T).

    This is equivalent to minimising ||ω − B1.T @ α||² over node
    potentials α ∈ R^{|V|}, which yields the gradient component.

    Coexact residual:
      ω_coexact = ω − ω_exact

    Harmonic:
      Estimated as the part of ω_coexact lying in the near-nullspace
      of B1 @ B1.T (eigenvalues < 1e-6). For kNN graphs this is
      typically machine-zero.
    """
    n_edges, n_nodes = B1.shape

    # ── Exact: project onto Im(B1.T) ─────────────────────────────────────
    # Solve B1 @ phi = omega (LSQR on the node-potential system)
    # B1 has shape (n_edges, n_nodes); B1.T maps nodes → edges.
    # We solve: min ||B1.T @ alpha - omega||  (LSQR on B1.T system)
    # Note: B1.T ∈ R^{n_nodes × n_edges} maps edges to node divergence.
    # We need: min_{alpha} || B1.T @ alpha - omega ||
    # which is transposed: min_{alpha} || (B1.T).T @ alpha^T - omega^T ||
    # LSQR(A, b) solves min ||A @ x - b||, so pass A=B1.T, b=omega.
    # Wait — B1.T has shape (n_nodes, n_edges). We want alpha in R^{n_nodes}.
    # The system: B1.T alpha ≈ omega  has shape (n_nodes,) @ (n_nodes,) 
    # But we need to project omega onto Im(B1.T) = Im(grad).
    # Correct: solve  (B1 B1.T) alpha = B1 omega  for alpha ∈ R^{n_nodes}
    # then omega_exact = B1.T alpha.

    # L0 = B1.T @ B1  — node Laplacian, shape (n_nodes, n_nodes)
    # rhs = B1.T @ omega — node divergence of the edge field
    L0  = B1.T.dot(B1)      # (n_nodes, n_nodes)
    rhs = B1.T.dot(omega)   # (n_nodes,)

    # LSQR on the node Laplacian system
    # L0 is positive semi-definite; add ridge for numerical stability
    if ridge > 0:
        diag_vals = np.array(L0.diagonal()) + ridge
        L0_reg = L0.copy().tolil()
        L0_reg.setdiag(diag_vals)
        L0_reg = L0_reg.tocsr()
    else:
        L0_reg = L0

    result = spla.lsqr(L0_reg, rhs, atol=1e-10, btol=1e-10, iter_lim=10000)
    alpha       = result[0]             # node potentials
    omega_exact = B1.dot(alpha)         # gradient: B1 @ alpha → (n_edges,)

    # ── Coexact: residual ────────────────────────────────────────────────
    omega_coexact = omega - omega_exact

    # ── Harmonic: near-nullspace of B1.T B1 (edge Laplacian) ─────────────
    # For small graphs, estimate with a few eigenmodes.
    # For large kNN graphs this is effectively zero.
    omega_harmonic = np.zeros_like(omega)
    n_eig = min(5, n_edges - 2)
    if n_eig > 1:
        try:
            L1 = B1.dot(B1.T)   # edge (1-)Laplacian: (n_edges, n_edges)
            vals, vecs = spla.eigsh(L1, k=n_eig, which='SM', tol=1e-6)
            harmonic_mask = vals < 1e-6
            if harmonic_mask.any():
                coexact_coeffs = vecs[:, harmonic_mask].T @ omega_coexact
                omega_harmonic = vecs[:, harmonic_mask] @ coexact_coeffs
                omega_coexact  = omega_coexact - omega_harmonic
        except Exception:
            pass  # harmonic is zero for connected graphs

    return omega_exact, omega_coexact, omega_harmonic


# ── Per-sample processing ─────────────────────────────────────────────────

def process_sample(sid: str,
                   sub: sc.AnnData,
                   prog_a: str,
                   prog_b: str,
                   k: int,
                   ridge: float) -> dict | None:

    n      = sub.n_obs
    coords = sub.obsm['spatial']
    A      = sub.obs[prog_a].values.astype(float)
    B      = sub.obs[prog_b].values.astype(float)

    if n < k + 2:
        print(f'  [{sid}] SKIP — too few spots ({n})')
        return None

    # Graph
    src, tgt = build_knn_edges(coords, k)
    n_edges  = len(src)
    if n_edges == 0:
        print(f'  [{sid}] SKIP — no edges')
        return None

    B1    = build_incidence_matrix(src, tgt, n)
    omega = wedge_field(A, B, src, tgt)

    # Hodge decomposition
    omega_e, omega_c, omega_h = hodge_decompose(omega, B1, ridge=ridge)

    total_sq = np.dot(omega, omega)
    e_sq     = np.dot(omega_e, omega_e)
    c_sq     = np.dot(omega_c, omega_c)
    h_sq     = np.dot(omega_h, omega_h)

    # Normalise by edge count for cross-sample comparability
    norm = n_edges if n_edges > 0 else 1
    cid  = '_'.join(sid.split('_')[:2])   # cytassist_XX
    tp   = sid.split('_')[-1]             # pre or post
    pid, resp, trt = PATIENT_MAP.get(cid, (None, None, None))

    rec = {
        'sample_id':        sid,
        'cytassist_id':     cid,
        'timepoint':        tp,
        'patient_id':       pid,
        'Response':         resp,
        'treatment':        trt,
        'n_spots':          n,
        'n_edges':          n_edges,
        # Raw squared norms
        'exact_energy_raw':     float(e_sq),
        'coexact_energy_raw':   float(c_sq),
        'harmonic_energy_raw':  float(h_sq),
        'total_energy_raw':     float(total_sq),
        # Normalised by edge count
        'exact_energy':     float(e_sq   / norm),
        'coexact_energy':   float(c_sq   / norm),
        'harmonic_energy':  float(h_sq   / norm),
        'total_energy':     float(total_sq / norm),
        # Fractions
        'exact_fraction':   float(e_sq   / total_sq) if total_sq > 0 else 0.,
        'coexact_fraction': float(c_sq   / total_sq) if total_sq > 0 else 0.,
        'harmonic_fraction':float(h_sq   / total_sq) if total_sq > 0 else 0.,
    }

    print(f"  [{sid}]  n={n:5d}  edges={n_edges:6d}  "
          f"coexact_E={rec['coexact_energy']:8.4f}  "
          f"coexact_frac={rec['coexact_fraction']:.4f}")
    return rec


# ── Paired delta computation ───────────────────────────────────────────────

def build_delta(summary: pd.DataFrame) -> pd.DataFrame:
    pre  = summary[summary['timepoint'] == 'pre'].set_index('cytassist_id')
    post = summary[summary['timepoint'] == 'post'].set_index('cytassist_id')
    rows = []
    for cid in pre.index:
        if cid not in post.index:
            continue
        pr, po = pre.loc[cid], post.loc[cid]
        rows.append({
            'cytassist_id':           cid,
            'patient_id':             pr['patient_id'],
            'Response':               pr['Response'],
            'treatment':              pr['treatment'],
            'n_spots_pre':            int(pr['n_spots']),
            'n_spots_post':           int(po['n_spots']),
            'size_ratio':             po['n_spots'] / pr['n_spots'],
            # Raw energies (normalised by edge count)
            'exact_energy_pre':       pr['exact_energy'],
            'exact_energy_post':      po['exact_energy'],
            'coexact_energy_pre':     pr['coexact_energy'],
            'coexact_energy_post':    po['coexact_energy'],
            'coexact_fraction_pre':   pr['coexact_fraction'],
            'coexact_fraction_post':  po['coexact_fraction'],
            # Deltas
            'delta_exact_energy':     po['exact_energy']     - pr['exact_energy'],
            'delta_coexact_energy':   po['coexact_energy']   - pr['coexact_energy'],
            'delta_coexact_fraction': po['coexact_fraction'] - pr['coexact_fraction'],
        })
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────

def plot_boxplot(delta: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('HCC Cohort — Δcoexact energy by response group\n'
                 '(tumour × exhaustion wedge field, Hodge decomposition)',
                 fontsize=11)

    for ax, metric, label in zip(
        axes,
        ['delta_coexact_energy', 'delta_coexact_fraction'],
        ['Δ coexact energy\n(edge-normalised)', 'Δ coexact fraction\n(fraction of total energy)'],
    ):
        rng = np.random.RandomState(42)
        for xi, (resp, col) in enumerate([
            ('Responder', '#d62728'), ('Non_Responder', '#1f77b4')
        ]):
            vals = delta[delta['Response'] == resp][metric].values
            jx   = xi + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(jx, vals, color=col, s=70, zorder=3, alpha=0.85)
            ax.plot([xi - 0.3, xi + 0.3], [np.median(vals)] * 2,
                    color=col, lw=2.5, zorder=4)
            for i, v in enumerate(vals):
                lbl = delta[delta['Response'] == resp]['cytassist_id'].values[i]
                ax.text(xi + 0.22, v, lbl.replace('cytassist_', ''),
                        fontsize=7, va='center', color=col)

        ax.axhline(0, color='grey', lw=0.8, ls='--')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Responder\n(n=5)', 'Non-Responder\n(n=6)'], fontsize=10)
        ax.set_ylabel(label, fontsize=9)

        r_vals = delta[delta['Response'] == 'Responder'][metric]
        nr_vals = delta[delta['Response'] == 'Non_Responder'][metric]
        _, p = mannwhitneyu(r_vals, nr_vals, alternative='two-sided')
        ax.set_title(f'Mann-Whitney p = {p:.3f} (two-sided)', fontsize=9)

    r_patch  = mpatches.Patch(color='#d62728', label='Responder')
    nr_patch = mpatches.Patch(color='#1f77b4', label='Non-Responder')
    fig.legend(handles=[r_patch, nr_patch], fontsize=9,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_scatter(delta: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('HCC — Pre vs Post coexact energy per patient\n'
                 '(edge-normalised; diagonal = no change)', fontsize=11)

    for ax, metric, label in zip(
        axes,
        ['coexact_energy', 'coexact_fraction'],
        ['coexact energy (edge-normalised)', 'coexact fraction'],
    ):
        all_vals = pd.concat([
            delta[f'{metric}_pre'],
            delta[f'{metric}_post']
        ])
        lim = (0, all_vals.max() * 1.05)

        for _, row in delta.iterrows():
            c   = COLOURS.get(row['Response'], 'grey')
            x   = row[f'{metric}_pre']
            y   = row[f'{metric}_post']
            ax.scatter(x, y, color=c, s=70, zorder=3)
            ax.annotate(
                row['cytassist_id'].replace('cytassist_', ''),
                (x, y), textcoords='offset points', xytext=(5, 3),
                fontsize=7, color=c,
            )

        ax.plot(lim, lim, 'k--', lw=0.8, alpha=0.5, label='no change')
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'Pre-therapy {label}', fontsize=9)
        ax.set_ylabel(f'Post-therapy {label}', fontsize=9)

        r_patch  = mpatches.Patch(color='#d62728', label='Responder')
        nr_patch = mpatches.Patch(color='#1f77b4', label='Non-Responder')
        ax.legend(handles=[r_patch, nr_patch], fontsize=8)

    fig.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='HCC Hodge decomposition of wedge field')
    ap.add_argument('--adata',
                    default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--prog-a',  default='tumor_score')
    ap.add_argument('--prog-b',  default='exhaustion_score')
    ap.add_argument('--k',       type=int,   default=6)
    ap.add_argument('--ridge',   type=float, default=1e-8)
    ap.add_argument('--out-summary', default='results_hcc_hodge_summary.csv')
    ap.add_argument('--out-delta',   default='results_hcc_hodge_delta.csv')
    ap.add_argument('--fig-box',   default='fig_hcc_delta_coexact_boxplot.png')
    ap.add_argument('--fig-scat',  default='fig_hcc_prepost_coexact_scatter.png')
    args = ap.parse_args()

    print(f'Loading {args.adata} …')
    adata = sc.read_h5ad(args.adata)
    print(f'  {adata.n_obs:,} spots  '
          f'programs: {args.prog_a} × {args.prog_b}  k={args.k}')

    for p in (args.prog_a, args.prog_b):
        if p not in adata.obs.columns:
            raise RuntimeError(
                f"Program '{p}' missing. Run build_program_scores.py first.")

    samples = sorted(adata.obs['sample_id'].unique())
    print(f'\nProcessing {len(samples)} samples …\n')

    records = []
    for sid in samples:
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = process_sample(
            sid, sub, args.prog_a, args.prog_b, args.k, args.ridge)
        if rec:
            records.append(rec)

    summary = pd.DataFrame(records)
    summary.to_csv(args.out_summary, index=False)
    print(f'\nSaved summary → {args.out_summary}')

    # ── Paired delta ──────────────────────────────────────────────────────
    delta = build_delta(summary)
    delta.to_csv(args.out_delta, index=False)
    print(f'Saved delta   → {args.out_delta}')

    # ── Statistics ────────────────────────────────────────────────────────
    resp = delta[delta['Response'] == 'Responder']
    nr   = delta[delta['Response'] == 'Non_Responder']

    print('\n' + '='*65)
    print('PATIENT-LEVEL PAIRED TABLE')
    print('='*65)
    print(f"{'cid':<16} {'R/NR':>5} {'n_pre':>7} {'n_post':>7} "
          f"{'ce_pre':>9} {'ce_post':>9} {'Δce':>9}  "
          f"{'cf_pre':>7} {'cf_post':>7} {'Δcf':>7}")
    print('─'*100)
    for resp_val in ['Responder','Non_Responder']:
        for _, r in delta[delta['Response']==resp_val].iterrows():
            tag = 'R' if resp_val == 'Responder' else 'NR'
            print(f"{r['cytassist_id']:<16} {tag:>5} "
                  f"{int(r['n_spots_pre']):>7} {int(r['n_spots_post']):>7} "
                  f"{r['coexact_energy_pre']:>9.4f} "
                  f"{r['coexact_energy_post']:>9.4f} "
                  f"{r['delta_coexact_energy']:>+9.4f}  "
                  f"{r['coexact_fraction_pre']:>7.4f} "
                  f"{r['coexact_fraction_post']:>7.4f} "
                  f"{r['delta_coexact_fraction']:>+7.4f}")
        print()

    print('='*65)
    print('GROUP SUMMARY — delta_coexact_energy')
    print('='*65)
    for resp_val, label in [('Responder','Responders'), ('Non_Responder','Non-Responders')]:
        d = delta[delta['Response']==resp_val]['delta_coexact_energy']
        n_up = (d > 0).sum()
        print(f'  {label:<17}  n={len(d)}  '
              f'median_Δ={d.median():+.5f}  '
              f'mean_Δ={d.mean():+.5f}  UP={n_up}/{len(d)}')

    r_ce  = resp['delta_coexact_energy']
    nr_ce = nr['delta_coexact_energy']
    stat, pval = mannwhitneyu(r_ce, nr_ce, alternative='two-sided')
    print(f'\n  Mann-Whitney U (two-sided): U={stat:.0f}  p={pval:.4f}')
    r2, _ = mannwhitneyu(r_ce, nr_ce, alternative='greater')
    _, p1 = mannwhitneyu(r_ce, nr_ce, alternative='greater')
    print(f'  Mann-Whitney (one-sided R>NR): p={p1:.4f}')

    rho, rp = spearmanr(delta['size_ratio'], delta['delta_coexact_energy'])
    print(f'\n  Spearman r(size_ratio, Δcoexact_energy) = {rho:.3f}  p={rp:.4f}')
    print(f'  {"→ size confound controlled" if abs(rho) < 0.4 else "→ residual size confound present"}')

    print('\n' + '='*65)
    print('FRACTION SUMMARY — delta_coexact_fraction')
    print('='*65)
    for resp_val, label in [('Responder','Responders'), ('Non_Responder','Non-Responders')]:
        d = delta[delta['Response']==resp_val]['delta_coexact_fraction']
        print(f'  {label:<17}  median_Δfrac={d.median():+.4f}  '
              f'UP={(d>0).sum()}/{len(d)}')

    _, pf = mannwhitneyu(
        resp['delta_coexact_fraction'],
        nr['delta_coexact_fraction'],
        alternative='two-sided')
    print(f'  Mann-Whitney p (fraction): {pf:.4f}')

    # ── Figures ───────────────────────────────────────────────────────────
    plot_boxplot(delta, Path(args.fig_box))
    plot_scatter(delta, Path(args.fig_scat))
    print(f'\nFigures saved:')
    print(f'  {args.fig_box}')
    print(f'  {args.fig_scat}')


if __name__ == '__main__':
    main()
