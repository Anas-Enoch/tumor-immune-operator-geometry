#!/usr/bin/env python3
"""
build_hodge_interface.py — Interface-Restricted Hodge Decomposition
====================================================================
Restricts the wedge field and Hodge decomposition to the spatial
subgraph defined by interface-like spots and their immediate neighbors.

Motivation
----------
Whole-section Hodge averages coexact energy over the full tissue,
including bulk tumor and stroma that carry no meaningful tumor–immune
antisymmetric interaction. Post-therapy sections are 4–40× larger
than pre-therapy biopsies, further diluting the interface signal.

The interface-restricted subgraph contains only:
  1. Interface nodes (tumor_score > sample Q75 AND tcell_score > Q75)
  2. Their 1-hop neighbors in the kNN graph (border zone)

This concentrates the analysis on the biologically active zone.

Interface definition
--------------------
Same per-sample Q75 thresholds as build_interface_mask.py.
If sample Q75(tumor_score) < TUMOR_FLOOR, the sample is flagged
as low-tumor-signal (likely complete response) and its interface
coexact energy is reported as near-zero by design.

Output
------
  results_hcc_hodge_interface_summary.csv   per-sample metrics
  results_hcc_hodge_interface_delta.csv     per-patient Δ metrics
  fig_hcc_interface_coexact_boxplot.png
  fig_hcc_interface_coexact_scatter.png
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
TUMOR_FLOOR = 0.05   # samples below this Q75 are flagged as low-tumor


# ── Graph utilities ────────────────────────────────────────────────────────

def build_knn_graph(coords: np.ndarray, k: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Undirected kNN edge list, each pair stored once (src < tgt)."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    src_l, tgt_l = [], []
    seen = set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen.add(key)
                src_l.append(key[0])
                tgt_l.append(key[1])
    return np.array(src_l, np.int32), np.array(tgt_l, np.int32)


def adjacency_list(src: np.ndarray, tgt: np.ndarray,
                   n: int) -> list[list[int]]:
    """Build per-node neighbour lists."""
    adj = [[] for _ in range(n)]
    for s, t in zip(src, tgt):
        adj[s].append(int(t))
        adj[t].append(int(s))
    return adj


def interface_subgraph(interface_mask: np.ndarray,
                       adj: list[list[int]],
                       src: np.ndarray,
                       tgt: np.ndarray,
                       include_neighbors: bool = True
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect interface nodes + (optionally) their 1-hop neighbors.
    Returns:
      keep_nodes : sorted array of global node indices in subgraph
      sub_src    : edge sources re-indexed to subgraph
      sub_tgt    : edge targets re-indexed to subgraph
    """
    keep = set(np.where(interface_mask)[0].tolist())
    if include_neighbors:
        border = set()
        for v in keep:
            border.update(adj[v])
        keep.update(border)
    keep_nodes = np.array(sorted(keep), dtype=np.int32)
    node_remap = {int(v): i for i, v in enumerate(keep_nodes)}

    sub_src, sub_tgt = [], []
    for s, t in zip(src, tgt):
        if int(s) in node_remap and int(t) in node_remap:
            sub_src.append(node_remap[int(s)])
            sub_tgt.append(node_remap[int(t)])

    return keep_nodes, np.array(sub_src, np.int32), np.array(sub_tgt, np.int32)


def build_B1(src: np.ndarray, tgt: np.ndarray,
             n_nodes: int) -> sp.csr_matrix:
    """Signed node-edge incidence matrix, shape (n_edges, n_nodes)."""
    m = len(src)
    rows = np.tile(np.arange(m), 2)
    cols = np.concatenate([tgt, src])
    data = np.concatenate([np.ones(m), -np.ones(m)])
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(m, n_nodes)).tocsr()


def wedge(A: np.ndarray, B: np.ndarray,
          src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    return A[src] * B[tgt] - A[tgt] * B[src]


def hodge_decompose(omega: np.ndarray, B1: sp.csr_matrix,
                    ridge: float = 1e-8
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (omega_exact, omega_coexact)."""
    L0  = B1.T.dot(B1)
    rhs = B1.T.dot(omega)
    diag = np.array(L0.diagonal()) + ridge
    Lr = L0.tolil(); Lr.setdiag(diag); Lr = Lr.tocsr()
    alpha   = spla.lsqr(Lr, rhs, atol=1e-10, btol=1e-10,
                        iter_lim=10000)[0]
    omega_e = B1.dot(alpha)
    return omega_e, omega - omega_e


# ── Per-sample processing ──────────────────────────────────────────────────

def process_sample(sid: str, sub: sc.AnnData,
                   prog_a: str, prog_b: str,
                   k: int, ridge: float,
                   include_neighbors: bool,
                   min_interface: int = 20) -> dict:
    n      = sub.n_obs
    coords = sub.obsm['spatial']
    A      = sub.obs[prog_a].values.astype(float)
    B      = sub.obs[prog_b].values.astype(float)

    cid = '_'.join(sid.split('_')[:2])
    tp  = sid.split('_')[-1]
    pid, resp, trt = PATIENT_MAP.get(cid, (None, None, None))

    base = dict(sample_id=sid, cytassist_id=cid, timepoint=tp,
                patient_id=pid, Response=resp, treatment=trt,
                n_spots=n)

    if n < k + 2:
        print(f'  [{sid}] SKIP — too few spots ({n})')
        return {**base, 'status': 'too_few_spots',
                'n_interface': 0, 'coexact_energy': 0.,
                'coexact_fraction': 0., 'exact_energy': 0.}

    # ── Interface mask (per-sample Q75) ───────────────────────────────────
    tumor_q75  = float(np.quantile(A, 0.75))
    immune_q75 = float(np.quantile(B, 0.75))
    low_tumor  = tumor_q75 < TUMOR_FLOOR

    interface_mask = (A > tumor_q75) & (B > immune_q75)
    n_interface    = int(interface_mask.sum())

    # ── Minimum interface size check ──────────────────────────────────────
    if not low_tumor and n_interface < min_interface:
        print(f'  [{sid}] SMALL INTERFACE (n={n_interface} < {min_interface}) '
              f'— Laplacian system is numerically unstable; flagged')
        return {**base, 'status': 'small_interface',
                'tumor_q75': tumor_q75, 'n_interface': n_interface,
                'coexact_energy': np.nan, 'exact_energy': np.nan,
                'coexact_fraction': np.nan, 'total_energy': np.nan}

    if low_tumor:
        print(f'  [{sid}] LOW TUMOR SIGNAL (Q75={tumor_q75:.4f}) '
              f'— interface coexact ≈ 0 by design '
              f'(likely complete response)')
        return {**base, 'status': 'low_tumor_signal',
                'tumor_q75': tumor_q75, 'n_interface': n_interface,
                'coexact_energy': 0., 'exact_energy': 0.,
                'coexact_fraction': 0., 'total_energy': 0.}

    # ── Global kNN graph ─────────────────────────────────────────────────
    src_g, tgt_g = build_knn_graph(coords, k)
    adj          = adjacency_list(src_g, tgt_g, n)

    # ── Interface subgraph ───────────────────────────────────────────────
    keep_nodes, sub_src, sub_tgt = interface_subgraph(
        interface_mask, adj, src_g, tgt_g, include_neighbors)

    n_sub   = len(keep_nodes)
    n_edges = len(sub_src)

    if n_edges < 3 or n_sub < k + 1:
        print(f'  [{sid}] SKIP — interface subgraph too small '
              f'(nodes={n_sub}, edges={n_edges})')
        return {**base, 'status': 'subgraph_too_small',
                'tumor_q75': tumor_q75, 'n_interface': n_interface,
                'n_subgraph_nodes': n_sub,
                'coexact_energy': np.nan, 'exact_energy': np.nan,
                'coexact_fraction': np.nan, 'total_energy': np.nan}

    # Programs on subgraph
    A_sub = A[keep_nodes]
    B_sub = B[keep_nodes]
    omega = wedge(A_sub, B_sub, sub_src, sub_tgt)

    B1 = build_B1(sub_src, sub_tgt, n_sub)
    omega_e, omega_c = hodge_decompose(omega, B1, ridge)

    norm   = n_edges
    e_sq   = float(np.dot(omega_e, omega_e))
    c_sq   = float(np.dot(omega_c, omega_c))
    tot_sq = float(np.dot(omega,   omega))

    cf = c_sq / tot_sq if tot_sq > 0 else 0.

    # Interface-only coexact (nodes strictly in interface, no border)
    iface_only = np.array([
        i for i, gv in enumerate(keep_nodes)
        if interface_mask[gv]
    ])
    if len(iface_only) > 0:
        ce_iface = float(np.dot(omega_c[iface_only], omega_c[iface_only])
                         / max(len(iface_only), 1))
    else:
        ce_iface = 0.

    print(f'  [{sid}]  iface={n_interface:4d}  sub_nodes={n_sub:4d}  '
          f'edges={n_edges:5d}  '
          f'coexact_E={c_sq/norm:8.3f}  '
          f'coexact_frac={cf:.4f}  '
          f'iface_ce={ce_iface:.3f}')

    return {**base,
            'status':               'ok',
            'tumor_q75':            tumor_q75,
            'n_interface':          n_interface,
            'n_subgraph_nodes':     n_sub,
            'n_edges':              n_edges,
            'exact_energy':         e_sq / norm,
            'coexact_energy':       c_sq / norm,
            'total_energy':         tot_sq / norm,
            'coexact_fraction':     cf,
            'exact_fraction':       e_sq / tot_sq if tot_sq > 0 else 0.,
            'iface_coexact_energy': ce_iface,
            }


# ── Delta computation ──────────────────────────────────────────────────────

def build_delta(summary: pd.DataFrame) -> pd.DataFrame:
    pre  = summary[summary['timepoint'] == 'pre'].set_index('cytassist_id')
    post = summary[summary['timepoint'] == 'post'].set_index('cytassist_id')
    EXCLUDE_STATUSES = {'small_interface', 'low_tumor_signal',
                        'subgraph_too_small', 'too_few_spots'}
    rows = []
    for cid in pre.index:
        if cid not in post.index:
            continue
        pr, po = pre.loc[cid], post.loc[cid]
        pre_ok  = pr.get('status','ok') not in EXCLUDE_STATUSES
        post_ok = po.get('status','ok') not in EXCLUDE_STATUSES
        ce_pre  = pr['coexact_energy']  if (pre_ok and pd.notna(pr['coexact_energy']))  else np.nan
        ce_post = po['coexact_energy']  if (post_ok and pd.notna(po['coexact_energy'])) else np.nan
        cf_pre  = pr['coexact_fraction']if (pre_ok and pd.notna(pr['coexact_fraction']))else np.nan
        cf_post = po['coexact_fraction']if (post_ok and pd.notna(po['coexact_fraction']))else np.nan
        rows.append({
            'cytassist_id':           cid,
            'patient_id':             pr['patient_id'],
            'Response':               pr['Response'],
            'treatment':              pr['treatment'],
            'n_spots_pre':            int(pr['n_spots']),
            'n_spots_post':           int(po['n_spots']),
            'n_interface_pre':        int(pr.get('n_interface', 0)),
            'n_interface_post':       int(po.get('n_interface', 0)),
            'size_ratio':             po['n_spots'] / pr['n_spots'],
            'iface_ratio':            (po.get('n_interface',0) /
                                       max(pr.get('n_interface',1), 1)),
            'status_pre':             pr.get('status',''),
            'status_post':            po.get('status',''),
            'coexact_energy_pre':     ce_pre,
            'coexact_energy_post':    ce_post,
            'coexact_fraction_pre':   cf_pre,
            'coexact_fraction_post':  cf_post,
            'delta_coexact_energy':   ce_post - ce_pre
                if (pd.notna(ce_pre) and pd.notna(ce_post)) else np.nan,
            'delta_coexact_fraction': cf_post - cf_pre
                if (pd.notna(cf_pre) and pd.notna(cf_post)) else np.nan,
            'both_ok':               int(pre_ok and post_ok),
        })
    return pd.DataFrame(rows)


# ── Figures ────────────────────────────────────────────────────────────────

def plot_boxplot(delta: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('HCC — Interface-restricted Δcoexact energy by response\n'
                 '(tumour × exhaustion wedge field, interface subgraph only)',
                 fontsize=11)
    for ax, metric, ylabel in zip(
        axes,
        ['delta_coexact_energy', 'delta_coexact_fraction'],
        ['Δ coexact energy (interface subgraph,\nedge-normalised)',
         'Δ coexact fraction (interface subgraph)'],
    ):
        rng = np.random.RandomState(42)
        for xi, (resp, col) in enumerate([
            ('Responder','#d62728'), ('Non_Responder','#1f77b4')
        ]):
            grp  = delta[delta['Response'] == resp]
            vals = grp[metric].values
            jx   = xi + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(jx, vals, color=col, s=70, zorder=3, alpha=0.9)
            ax.plot([xi-.3, xi+.3], [np.median(vals)]*2,
                    color=col, lw=2.5, zorder=4)
            for i, (_, row) in enumerate(grp.iterrows()):
                ax.text(xi+.22, vals[i],
                        row['cytassist_id'].replace('cytassist_',''),
                        fontsize=7, va='center', color=col)

        ax.axhline(0, color='grey', lw=.8, ls='--')
        ax.set_xticks([0,1])
        ax.set_xticklabels(['Responder\n(n=5)','Non-Responder\n(n=6)'],
                           fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)

        r_v  = delta[delta['Response']=='Responder'][metric]
        nr_v = delta[delta['Response']=='Non_Responder'][metric]
        _, p = mannwhitneyu(r_v, nr_v, alternative='two-sided')
        ax.set_title(f'Mann-Whitney p = {p:.3f} (two-sided)', fontsize=9)

    fig.legend(
        handles=[mpatches.Patch(color='#d62728', label='Responder'),
                 mpatches.Patch(color='#1f77b4', label='Non-Responder')],
        fontsize=9, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.04))
    fig.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


def plot_scatter(delta: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('HCC — Pre vs Post coexact energy (interface subgraph)\n'
                 'Diagonal = no change', fontsize=11)
    for ax, pre_col, post_col, label in zip(
        axes,
        ['coexact_energy_pre',   'coexact_fraction_pre'],
        ['coexact_energy_post',  'coexact_fraction_post'],
        ['coexact energy (interface, edge-norm)', 'coexact fraction (interface)'],
    ):
        all_v = np.concatenate([delta[pre_col].values,
                                 delta[post_col].values])
        all_v = all_v[np.isfinite(all_v)]
        lim   = (0, all_v.max() * 1.08) if len(all_v) else (0,1)
        for _, row in delta.iterrows():
            c = COLOURS.get(row['Response'], 'grey')
            x, y = row[pre_col], row[post_col]
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(x, y, color=c, s=75, zorder=3)
            ax.annotate(row['cytassist_id'].replace('cytassist_',''),
                        (x,y), textcoords='offset points',
                        xytext=(5,3), fontsize=7, color=c)
        ax.plot(lim, lim, 'k--', lw=.8, alpha=.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(f'Pre-therapy {label}', fontsize=9)
        ax.set_ylabel(f'Post-therapy {label}', fontsize=9)
        ax.legend(
            handles=[mpatches.Patch(color='#d62728', label='Responder'),
                     mpatches.Patch(color='#1f77b4', label='Non-Responder')],
            fontsize=8)
    fig.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Interface-restricted Hodge decomposition for HCC')
    ap.add_argument('--adata',  default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--prog-a', default='tumor_score')
    ap.add_argument('--prog-b', default='exhaustion_score')
    ap.add_argument('--k',      type=int,   default=6)
    ap.add_argument('--ridge',  type=float, default=1e-8)
    ap.add_argument('--min-interface', type=int, default=20,
                    help='Minimum interface nodes required for analysis '
                         '(default 20; samples below are flagged and excluded)')
    ap.add_argument('--no-border', action='store_true',
                    help='Exclude 1-hop border zone; use strict interface only')
    ap.add_argument('--out-summary',
                    default='results_hcc_hodge_interface_summary.csv')
    ap.add_argument('--out-delta',
                    default='results_hcc_hodge_interface_delta.csv')
    ap.add_argument('--fig-box',
                    default='fig_hcc_interface_coexact_boxplot.png')
    ap.add_argument('--fig-scat',
                    default='fig_hcc_interface_coexact_scatter.png')
    args = ap.parse_args()
    include_neighbors = not args.no_border

    print(f'Loading {args.adata} …')
    adata = sc.read_h5ad(args.adata)
    print(f'  {adata.n_obs:,} spots  '
          f'{args.prog_a} × {args.prog_b}  k={args.k}  '
          f'border_zone={"yes" if include_neighbors else "no"}')

    for p in (args.prog_a, args.prog_b):
        if p not in adata.obs.columns:
            raise RuntimeError(f"'{p}' missing — run build_program_scores.py")

    samples = sorted(adata.obs['sample_id'].unique())
    print(f'\nProcessing {len(samples)} samples …\n')

    records = []
    for sid in samples:
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = process_sample(sid, sub, args.prog_a, args.prog_b,
                             args.k, args.ridge, include_neighbors,
                             args.min_interface)
        records.append(rec)

    summary = pd.DataFrame(records)
    summary.to_csv(args.out_summary, index=False)
    print(f'\nSaved → {args.out_summary}')

    delta = build_delta(summary)
    delta.to_csv(args.out_delta, index=False)
    print(f'Saved → {args.out_delta}')

    # ── Statistics ────────────────────────────────────────────────────────
    print('\n' + '='*70)
    print('PATIENT-LEVEL TABLE  (interface-restricted coexact)')
    print('='*70)
    print(f"{'cid':<16} {'R/NR':>4} {'iface_pre':>9} {'iface_post':>10} "
          f"{'ce_pre':>9} {'ce_post':>9} {'Δce':>9}  "
          f"{'cf_pre':>7} {'cf_post':>7} {'Δcf':>7}")
    print('─'*105)
    for resp_val in ['Responder','Non_Responder']:
        for _, r in delta[delta['Response']==resp_val].iterrows():
            tag = 'R' if resp_val=='Responder' else 'NR'
            print(f"{r['cytassist_id']:<16} {tag:>4} "
                  f"{int(r['n_interface_pre']):>9} {int(r['n_interface_post']):>10} "
                  f"{r['coexact_energy_pre']:>9.3f} "
                  f"{r['coexact_energy_post']:>9.3f} "
                  f"{r['delta_coexact_energy']:>+9.3f}  "
                  f"{r['coexact_fraction_pre']:>7.4f} "
                  f"{r['coexact_fraction_post']:>7.4f} "
                  f"{r['delta_coexact_fraction']:>+7.4f}")
        print()

    print('='*70)
    print('GROUP SUMMARY')
    print('='*70)
    for resp_val, label in [('Responder','Responders'),
                            ('Non_Responder','Non-Responders')]:
        d   = delta[delta['Response']==resp_val]['delta_coexact_energy']
        df_ = delta[delta['Response']==resp_val]['delta_coexact_fraction']
        print(f'  {label:<17}  n={len(d)}  '
              f'median_Δce={d.median():+.3f}  UP={(d>0).sum()}/{len(d)}  '
              f'median_Δcf={df_.median():+.4f}  UP_frac={(df_>0).sum()}/{len(d)}')

    # All samples (includes NaN rows)
    valid = delta[delta['both_ok'] == 1].dropna(
        subset=['delta_coexact_energy'])
    r_v  = valid[valid['Response']=='Responder']['delta_coexact_energy']
    nr_v = valid[valid['Response']=='Non_Responder']['delta_coexact_energy']

    print(f'\n  Quality-filtered pairs (both timepoints ok, '
          f'n_interface ≥ {args.min_interface}): {len(valid)}/{len(delta)}')
    print(f'  Responders (filtered):    n={len(r_v)}  '
          f'median_Δ={r_v.median():+.1f}  UP={(r_v>0).sum()}/{len(r_v)}')
    print(f'  Non-Responders (filtered):n={len(nr_v)}  '
          f'median_Δ={nr_v.median():+.1f}  UP={(nr_v>0).sum()}/{len(nr_v)}')
    if len(r_v) > 0 and len(nr_v) > 0:
        _, p2 = mannwhitneyu(r_v, nr_v, alternative='two-sided')
        _, p1 = mannwhitneyu(r_v, nr_v, alternative='greater')
        print(f'\n  Mann-Whitney (two-sided):  p = {p2:.4f}')
        print(f'  Mann-Whitney (R > NR):     p = {p1:.4f}')
    else:
        print('  Insufficient pairs for Mann-Whitney after filtering')

    # Compute only on rows where delta is defined
    _valid = delta.dropna(subset=['delta_coexact_energy','size_ratio'])
    if len(_valid) >= 3:
        rho, rp = spearmanr(_valid['size_ratio'], _valid['delta_coexact_energy'])
        rho_i, rp_i = spearmanr(_valid['iface_ratio'], _valid['delta_coexact_energy'])
    else:
        rho, rp, rho_i, rp_i = np.nan, np.nan, np.nan, np.nan
    print(f'\n  r(size_ratio,   Δce) = {rho:.3f}  p={rp:.4f}')
    print(f'  r(iface_ratio,  Δce) = {rho_i:.3f}  p={rp_i:.4f}')

    _vf   = delta.dropna(subset=['delta_coexact_fraction'])
    rf_v  = _vf[_vf['Response']=='Responder']['delta_coexact_fraction']
    nrf_v = _vf[_vf['Response']=='Non_Responder']['delta_coexact_fraction']
    if len(rf_v) > 0 and len(nrf_v) > 0:
        _, pf = mannwhitneyu(rf_v, nrf_v, alternative='two-sided')
        print(f'\n  Mann-Whitney fraction (two-sided): p = {pf:.4f}')
    else:
        print('  Insufficient pairs for fraction Mann-Whitney')

    # ── Pre-therapy baseline comparison (size-agnostic) ─────────────────
    pre_ok = summary[
        (summary['timepoint'] == 'pre') &
        (summary['status'] == 'ok') &
        (summary['n_interface'] >= args.min_interface)
    ]
    if len(pre_ok) >= 2:
        print('\n' + '='*65)
        print('PRE-THERAPY BASELINE (primary size-agnostic metric)')
        print('='*65)
        r_b  = pre_ok[pre_ok['Response']=='Responder']['coexact_fraction']
        nr_b = pre_ok[pre_ok['Response']=='Non_Responder']['coexact_fraction']
        print(f'  Responders    n={len(r_b)}  cf_pre values: '
              f'{r_b.values.round(4).tolist()}  median={r_b.median():.4f}')
        print(f'  Non-Responders n={len(nr_b)} cf_pre values: '
              f'{nr_b.values.round(4).tolist()}  median={nr_b.median():.4f}')
        if len(r_b) > 0 and len(nr_b) > 0:
            _, p_cf2 = mannwhitneyu(r_b, nr_b, alternative='two-sided')
            _, p_cf1 = mannwhitneyu(r_b, nr_b, alternative='greater')
            print(f'  Mann-Whitney (two-sided): p = {p_cf2:.4f}')
            print(f'  Mann-Whitney (R > NR):    p = {p_cf1:.4f}')
            print(f'  NOTE: Pre-therapy cf is size-agnostic and '
                  f'does not suffer from the biopsy-size confound.')

    plot_boxplot(delta, Path(args.fig_box))
    plot_scatter(delta, Path(args.fig_scat))
    print(f'\nFigures:\n  {args.fig_box}\n  {args.fig_scat}')


if __name__ == '__main__':
    main()
