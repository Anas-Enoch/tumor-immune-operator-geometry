#!/usr/bin/env python3
"""
build_ncg_commutators.py — Finite Noncommutative Interface Geometry
====================================================================
Computes the multi-field pairwise commutator matrix for each interface,
providing the operator-algebraic characterisation of each KTS state.

Mathematical structure
──────────────────────
For each interface subgraph with node graph Laplacian L ∈ ℝⁿˣⁿ and
diagonal multiplication operators M_p = diag(p_1,...,p_n) for biological
field p, the finite commutator:

    C(p,q) = M_p L M_q - M_q L M_p

has (i,j) entry:   C(p,q)_{ij} = L_{ij}(p_i q_j - q_i p_j)
                               = L_{ij} · ω_{pq}(i,j)

This is the wedge field ω_{pq} on edge (i,j), weighted by the graph
Laplacian entry. The Frobenius norm ||C(p,q)||_F is the weighted total
wedge energy between fields p and q.

Key identity
────────────
||C(p,q)||_F² = Σ_{ij} L²_{ij}(p_i q_j - q_i p_j)²

This CONNECTS to the Hodge decomposition:
    ||ω_{pq}||² = exact_energy + coexact_energy
    NCG purity   = coexact_energy / ||ω_{pq}||²  (already computed)

What is NEW
───────────
The MULTI-FIELD extension: computing C(p,q) for all pairs (p,q) across
{tumor, tcell, exhaustion, myeloid} simultaneously produces a 4×4
antisymmetric matrix of commutator norms per sample.

This reveals WHICH field pairs are most noncommutatively organised —
not just the total commutator energy from one pair.

Output metrics per sample
─────────────────────────
  comm_matrix[p,q]      ||C(p,q)||_F  for each program pair
  ncg_purity[p,q]       coexact fraction for each pair (Hodge)
  dominant_pair         the (p,q) pair with maximum commutator norm
  total_comm_energy     sum of all ||C(p,q)||_F
  operator_entropy      Shannon entropy of the commutator norm distribution
                        across all pairs — high = complex multi-pair structure,
                        low = one dominant interaction pair
  spectral_entropy_C    entropy of |eigenvalue| spectrum of C(tumor,exhaustion)

KTS state → NCG interpretation
───────────────────────────────
S0  Absent algebra — trivial commutators (no interface)
S1  Fragmented algebra — low commutator energy, gradient-dominated
S2  Connected algebra — moderate commutator, structured coexact component
S3  Amplified algebra — high commutator energy, multi-pair interactions
S4  Residual algebra  — concentrated commutator in few modes (low spectral entropy)

Outputs
───────
results_hcc_ncg_commutators.csv    per-sample commutator metrics
fig_hcc_ncg_commutator_matrix.png  heatmap of commutator norms by KTS state
"""

import argparse
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

PROGRAMS = ['tumor_score', 'tcell_score', 'exhaustion_score', 'myeloid_score']
PROG_SHORT = {'tumor_score': 'T', 'tcell_score': 'I',
              'exhaustion_score': 'E', 'myeloid_score': 'M'}
TUMOR_FLOOR = 0.05
MIN_IFACE   = 20


# ── Graph ──────────────────────────────────────────────────────────────────

def build_knn_edges(coords, k):
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    src, tgt, seen = [], [], set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            key = (min(i,int(j)), max(i,int(j)))
            if key not in seen:
                seen.add(key); src.append(key[0]); tgt.append(key[1])
    return np.array(src, np.int32), np.array(tgt, np.int32)


def build_laplacian(src, tgt, n):
    """Weighted node graph Laplacian from unweighted edges (w=1)."""
    m = len(src)
    A_data = np.ones(2*m)
    A_rows = np.concatenate([src, tgt])
    A_cols = np.concatenate([tgt, src])
    A = sp.coo_matrix((A_data,(A_rows,A_cols)), shape=(n,n)).tocsr()
    D = sp.diags(np.array(A.sum(axis=1)).ravel())
    return (D - A).toarray()            # dense; interface subgraph is small


def adj_list(src, tgt, n):
    adj = [[] for _ in range(n)]
    for s, t in zip(src, tgt):
        adj[s].append(int(t)); adj[t].append(int(s))
    return adj


def interface_subgraph(mask, adj, src, tgt, include_border=True):
    keep = set(np.where(mask)[0].tolist())
    if include_border:
        for v in list(keep): keep.update(adj[v])
    nodes  = np.array(sorted(keep), np.int32)
    remap  = {int(v): i for i, v in enumerate(nodes)}
    ss, st = [], []
    for s, t in zip(src, tgt):
        if int(s) in remap and int(t) in remap:
            ss.append(remap[int(s)]); st.append(remap[int(t)])
    return nodes, np.array(ss, np.int32), np.array(st, np.int32)


# ── Commutator computation ─────────────────────────────────────────────────

def commutator_norm(L: np.ndarray, p: np.ndarray, q: np.ndarray) -> float:
    """
    ||M_p L M_q - M_q L M_p||_F

    = ||(A·B^T - B·A^T) ⊙ L||_F  where A=p[:,None], B=q[:,None]

    Efficient formula:
    C_{ij} = L_{ij}(p_i q_j - q_i p_j)
    ||C||²_F = Σ_{ij} L²_{ij}(p_i q_j - q_i p_j)²
    """
    # Outer difference: (p_i q_j - q_i p_j) for all (i,j)
    wedge_mat = np.outer(p, q) - np.outer(q, p)  # antisymmetric (n×n)
    C         = L * wedge_mat                      # element-wise
    return float(np.sqrt(np.sum(C**2)))


def commutator_spectrum_entropy(L: np.ndarray,
                                p: np.ndarray, q: np.ndarray) -> float:
    """Shannon entropy of |eigenvalues| of C = M_p L M_q - M_q L M_p."""
    wedge_mat = np.outer(p, q) - np.outer(q, p)
    C         = L * wedge_mat
    evals     = np.linalg.eigvalsh(C)
    w = np.abs(evals)
    w = w[w > 1e-12]
    if len(w) == 0: return 0.
    w /= w.sum()
    return float(-np.dot(w, np.log(w)))


# ── Per-sample processing ──────────────────────────────────────────────────

def process_sample(sid, sub, k, kts_state):
    n      = sub.n_obs
    coords = sub.obsm['spatial']

    # Check all programs are present
    missing = [p for p in PROGRAMS if p not in sub.obs.columns]
    if missing:
        print(f"  [{sid}] SKIP — missing programs: {missing}")
        return None

    fields = {p: sub.obs[p].values.astype(float) for p in PROGRAMS}

    # Interface mask (tumor × tcell)
    A_t = fields['tumor_score']
    T_t = fields['tcell_score']
    tumor_q75  = float(np.quantile(A_t, 0.75))
    immune_q75 = float(np.quantile(T_t, 0.75))
    low_tumor  = tumor_q75 < TUMOR_FLOOR
    iface_mask = (A_t > tumor_q75) & (T_t > immune_q75)
    n_iface    = int(iface_mask.sum())

    if low_tumor or n_iface < MIN_IFACE:
        return None

    # Build interface subgraph
    src_g, tgt_g = build_knn_edges(coords, k)
    adj_g        = adj_list(src_g, tgt_g, n)
    keep_nodes, ss, st = interface_subgraph(iface_mask, adj_g, src_g, tgt_g)
    n_sub = len(keep_nodes)
    if n_sub < k + 2 or len(ss) < 4:
        return None

    # Build Laplacian on interface subgraph
    L = build_laplacian(ss, st, n_sub)

    # Subgraph field values
    fields_sub = {p: fields[p][keep_nodes] for p in PROGRAMS}

    # ── 4×4 commutator norm matrix ────────────────────────────────────────
    comm_norms = {}
    pairs = list(combinations(PROGRAMS, 2))
    for p, q in pairs:
        key = f"comm_{PROG_SHORT[p]}{PROG_SHORT[q]}"
        comm_norms[key] = commutator_norm(L, fields_sub[p], fields_sub[q])

    total_comm = float(sum(comm_norms.values()))

    # Operator entropy (distribution across pairs)
    vals = np.array(list(comm_norms.values()))
    vals_n = vals / (vals.sum() + 1e-12)
    op_entropy = float(-np.dot(vals_n[vals_n>0], np.log(vals_n[vals_n>0])))

    # Dominant pair
    dominant_pair = max(comm_norms, key=comm_norms.get)

    # Spectral entropy of C(tumor, exhaustion)
    spec_ent = commutator_spectrum_entropy(
        L, fields_sub['tumor_score'], fields_sub['exhaustion_score'])

    # NCG purity for (tumor, exhaustion): coexact / total wedge
    # (use Hodge decomposition on the primary pair)
    from scipy.sparse.linalg import lsqr
    B1_rows = np.tile(np.arange(len(ss)), 2)
    B1_cols = np.concatenate([st, ss])
    B1_data = np.concatenate([np.ones(len(ss)), -np.ones(len(ss))])
    B1 = sp.coo_matrix((B1_data,(B1_rows,B1_cols)),
                        shape=(len(ss), n_sub)).tocsr()
    omega = fields_sub['tumor_score'][ss]*fields_sub['exhaustion_score'][st] - \
            fields_sub['tumor_score'][st]*fields_sub['exhaustion_score'][ss]
    L0  = B1.T.dot(B1)
    rhs = B1.T.dot(omega)
    diag = np.array(L0.diagonal()) + 1e-8
    Lr = L0.tolil(); Lr.setdiag(diag); Lr = Lr.tocsr()
    alpha   = lsqr(Lr, rhs, atol=1e-10, btol=1e-10, iter_lim=5000)[0]
    omega_e = B1.dot(alpha); omega_c = omega - omega_e
    c_sq  = float(np.dot(omega_c, omega_c))
    tot_sq = float(np.dot(omega, omega))
    ncg_purity = float(c_sq / tot_sq) if tot_sq > 1e-12 else 0.

    cid = '_'.join(sid.split('_')[:2])
    tp  = sid.split('_')[-1]
    pid, resp, trt = PATIENT_MAP.get(cid, (None,None,None))

    rec = dict(
        sample_id     = sid,
        cytassist_id  = cid,
        timepoint     = tp,
        patient_id    = pid,
        Response      = resp,
        kts_state     = kts_state,
        n_interface   = n_iface,
        n_subgraph    = n_sub,
        **comm_norms,
        total_comm_energy  = total_comm,
        operator_entropy   = op_entropy,
        dominant_pair      = dominant_pair,
        spectral_entropy_C = spec_ent,
        ncg_purity_TE      = ncg_purity,
    )

    dominant_val = comm_norms[dominant_pair]
    print(f"  [{sid}] state={kts_state}  "
          f"total_E={total_comm:.1f}  "
          f"op_entr={op_entropy:.3f}  "
          f"ncg_purity={ncg_purity:.4f}  "
          f"dominant={dominant_pair}({dominant_val:.1f})")
    return rec


# ── Figures ────────────────────────────────────────────────────────────────

def plot_commutator_heatmap(df: pd.DataFrame, outpath: Path) -> None:
    """
    For each quality-filtered sample, show the 4×4 commutator norm matrix,
    grouped by KTS state.
    """
    pairs_labels = ['TI','TE','TM','IE','IM','EM']
    comm_cols    = [f'comm_{p}' for p in pairs_labels]
    present_cols = [c for c in comm_cols if c in df.columns]
    if not present_cols: return

    ordered_states = ['S2','S3','S4','S1','S0']
    samples_ordered = []
    for s in ordered_states:
        sub = df[df['kts_state']==s]
        samples_ordered.extend(sub['sample_id'].tolist())
    df_plot = df.set_index('sample_id').loc[
        [s for s in samples_ordered if s in df['sample_id'].values]
    ].reset_index()

    mat = df_plot[present_cols].values.astype(float)
    if mat.size == 0: return

    # Normalise row-wise (per sample) to show dominant pair pattern
    row_max = mat.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    mat_norm = mat / row_max

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df_plot)*0.4 + 1)))
    fig.suptitle('Pairwise commutator norms ||C(p,q)||_F per interface sample\n'
                 '(T=tumor, I=tcell, E=exhaustion, M=myeloid)\n'
                 'Rows sorted by KTS state (S2→S3→S4→S1)', fontsize=10)

    labels_y = [f"{r['sample_id'].replace('cytassist_','')} "
                f"({r['Response'][0] if r['Response'] else '?'}) "
                f"[{r['kts_state']}]"
                for _, r in df_plot.iterrows()]

    for ax, data, title, fmt in [
        (axes[0], mat,      'Raw commutator norms',       '.0f'),
        (axes[1], mat_norm, 'Row-normalised (dominant=1)', '.2f'),
    ]:
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=data.max())
        ax.set_xticks(range(len(present_cols)))
        ax.set_xticklabels([c.replace('comm_','') for c in present_cols],
                           fontsize=9)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(labels_y, fontsize=7.5)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)
        for i in range(len(df_plot)):
            for j, col in enumerate(present_cols):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:{fmt}}',
                            ha='center', va='center', fontsize=6.5,
                            color='white' if v > data.max()*0.6 else '#333')

    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--adata',       default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--kts-csv',     default='results_hcc_kts_states.csv')
    ap.add_argument('--k',           type=int, default=6)
    ap.add_argument('--out',         default='results_hcc_ncg_commutators.csv')
    ap.add_argument('--fig',         default='fig_hcc_ncg_commutator_matrix.png')
    args = ap.parse_args()

    print(f"Loading {args.adata} …")
    adata = sc.read_h5ad(args.adata)

    kts = pd.read_csv(args.kts_csv).set_index('sample_id')

    samples = sorted(adata.obs['sample_id'].unique())
    print(f"\nProcessing {len(samples)} samples …\n")

    records = []
    for sid in samples:
        state = kts.loc[sid, 'state'] if sid in kts.index else 'S?'
        if state in ('S0', 'S?'): continue
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = process_sample(sid, sub, args.k, state)
        if rec: records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f"\nSaved → {args.out}")

    # ── Statistics ────────────────────────────────────────────────────────
    print('\n' + '='*65)
    print('COMMUTATOR STRUCTURE BY KTS STATE')
    print('='*65)

    pair_cols = [c for c in df.columns if c.startswith('comm_')]
    for state in sorted(df['kts_state'].unique()):
        sub = df[df['kts_state']==state]
        print(f"\n  {state}  (n={len(sub)}  resp={sub['Response'].value_counts().to_dict()})")
        if not sub.empty:
            for col in pair_cols:
                print(f"    {col.replace('comm_',''):3s}  "
                      f"median={sub[col].median():8.2f}  "
                      f"max={sub[col].max():8.2f}")
            print(f"    op_entropy median={sub['operator_entropy'].median():.3f}  "
                  f"ncg_purity median={sub['ncg_purity_TE'].median():.4f}")

    print('\n' + '='*65)
    print('RESPONDER vs NON-RESPONDER — commutator structure')
    print('='*65)
    from scipy.stats import mannwhitneyu
    for col in pair_cols + ['total_comm_energy','operator_entropy','ncg_purity_TE']:
        if col not in df.columns: continue
        r  = df[df['Response']=='Responder'][col].dropna()
        nr = df[df['Response']=='Non_Responder'][col].dropna()
        if len(r)<2 or len(nr)<2: continue
        _, p = mannwhitneyu(r, nr, alternative='two-sided')
        sig = '*' if p < 0.1 else ''
        print(f"  {col:<28}  R={r.median():8.2f}  NR={nr.median():8.2f}  p={p:.4f} {sig}")

    print('\n' + '='*65)
    print('DOMINANT PAIR DISTRIBUTION')
    print('='*65)
    print(df.groupby(['Response','dominant_pair']).size().to_string())

    plot_commutator_heatmap(df, Path(args.fig))
    print(f'\nFigure → {args.fig}')


if __name__ == '__main__':
    main()
