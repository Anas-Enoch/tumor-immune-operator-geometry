#!/usr/bin/env python3
"""
build_coexact_spatial.py — Spatial Localization of Interface Coexact Structure
===============================================================================
Computes three analytical layers on the interface-restricted Hodge coexact field:

CORE LAYER — Localized coexact density
    c_i = (1/deg_i) * Σ_{e~i} |ω_coexact(e)|

SIGNED LAYER — Directional circulation analysis
    s_i = (1/deg_i) * Σ_{e~i} ω_coexact(e) * orientation(e,i)
    Measures whether coexact structure has coherent directionality.

SPECTRAL LAYER — Eigenmode decomposition of interface coexact structure
    ω_coexact = Σ_k α_k φ_k   (eigenvectors of interface 1-Laplacian)
    Distinguishes organized low-frequency modes from noisy high-frequency scatter.

All three layers use the same interface-restricted subgraph as build_hodge_interface.py.
All matrices remain sparse throughout.
"""

import argparse
from pathlib import Path
from collections import deque

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
import matplotlib.patches as mpatches

# ── Patient metadata ───────────────────────────────────────────────────────
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
COLOURS      = {'Responder': '#d62728', 'Non_Responder': '#1f77b4'}
TUMOR_FLOOR  = 0.05
MIN_IFACE    = 20


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def build_knn_edges(coords: np.ndarray, k: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    src_l, tgt_l, seen = [], [], set()
    for i, row in enumerate(idx):
        for j in row[1:]:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen.add(key)
                src_l.append(key[0]); tgt_l.append(key[1])
    return np.array(src_l, np.int32), np.array(tgt_l, np.int32)


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


def build_B1(src, tgt, n_nodes):
    """Shape: (n_edges, n_nodes)."""
    m    = len(src)
    rows = np.tile(np.arange(m), 2)
    cols = np.concatenate([tgt, src])
    data = np.concatenate([np.ones(m), -np.ones(m)])
    return sp.coo_matrix((data, (rows, cols)),
                         shape=(m, n_nodes)).tocsr()


def hodge_decompose(omega, B1, ridge=1e-8):
    """Returns (omega_exact, omega_coexact)."""
    L0  = B1.T.dot(B1)
    rhs = B1.T.dot(omega)
    diag = np.array(L0.diagonal()) + ridge
    Lr   = L0.tolil(); Lr.setdiag(diag); Lr = Lr.tocsr()
    alpha   = spla.lsqr(Lr, rhs, atol=1e-10, btol=1e-10,
                        iter_lim=10000)[0]
    omega_e = B1.dot(alpha)
    return omega_e, omega - omega_e


# ══════════════════════════════════════════════════════════════════════════════
# STATISTICS UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def gini(x: np.ndarray) -> float:
    x = np.sort(np.abs(x.ravel()))
    n = len(x)
    if n == 0 or x.sum() < 1e-12: return 0.
    idx = np.arange(1, n + 1)
    return float((2*np.dot(idx, x) - (n+1)*x.sum()) / (n*x.sum()))


def shannon_entropy(x: np.ndarray, bins: int = 20) -> float:
    x = x[x > 0]
    if len(x) == 0: return 0.
    counts, _ = np.histogram(x, bins=bins)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-np.dot(p, np.log(p)))


def largest_hotspot_fraction(density: np.ndarray,
                             src: np.ndarray, tgt: np.ndarray,
                             n: int, threshold_q: float = 0.75) -> float:
    """
    BFS connected-component analysis on hotspot nodes.
    Hotspot = density > threshold_q percentile.
    Returns fraction of hotspot nodes in the largest connected component.
    """
    thresh = np.percentile(density, threshold_q * 100)
    hot    = set(np.where(density > thresh)[0].tolist())
    if not hot: return 0.
    adj = [[] for _ in range(n)]
    for s, t in zip(src, tgt):
        if int(s) in hot and int(t) in hot:
            adj[s].append(int(t)); adj[t].append(int(s))
    visited  = set()
    max_comp = 0
    for start in hot:
        if start in visited: continue
        q = deque([start]); visited.add(start); size = 0
        while q:
            v = q.popleft(); size += 1
            for nb in adj[v]:
                if nb not in visited:
                    visited.add(nb); q.append(nb)
        max_comp = max(max_comp, size)
    return float(max_comp / len(hot))


# ══════════════════════════════════════════════════════════════════════════════
# CORE LAYER — localized coexact density
# ══════════════════════════════════════════════════════════════════════════════

def compute_core(omega_c: np.ndarray,
                 sub_src: np.ndarray, sub_tgt: np.ndarray,
                 n_sub: int) -> np.ndarray:
    """
    c_i = (1/deg_i) * Σ_{e~i} |ω_coexact(e)|  for nodes in interface subgraph.
    Returns array of length n_sub.
    """
    abs_c = np.abs(omega_c)
    density = np.zeros(n_sub, dtype=float)
    counts  = np.zeros(n_sub, dtype=float)
    np.add.at(density, sub_src, abs_c)
    np.add.at(density, sub_tgt, abs_c)
    np.add.at(counts,  sub_src, 1.)
    np.add.at(counts,  sub_tgt, 1.)
    mask = counts > 0
    density[mask] /= counts[mask]
    return density


# ══════════════════════════════════════════════════════════════════════════════
# SIGNED LAYER — directional circulation
# ══════════════════════════════════════════════════════════════════════════════

def compute_signed(omega_c: np.ndarray,
                   sub_src: np.ndarray, sub_tgt: np.ndarray,
                   n_sub: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Local sign coherence: what fraction of a node's incident edges agree in sign?

    NOTE: The Hodge coexact component satisfies B1.T @ ω_c = 0 by construction
    (it is divergence-free). Node divergence of ω_c is therefore identically zero
    everywhere — measuring divergence would always return 0.

    Instead we measure LOCAL SIGN COHERENCE:
      lsc_i = |Σ_{e~i} sign(ω_c(e))| / deg_i

    lsc_i = 1  → all edges incident to i have the same sign (coherent direction)
    lsc_i = 0  → equal CW and CCW edges cancel (vortex centre or noise)

    Also returns:
      pos_fraction_i = fraction of edges with ω_c > 0 (CCW convention)
      unsigned_density = mean |ω_c| per edge (same as core density for comparison)
    """
    sign_sum   = np.zeros(n_sub, float)   # Σ sign(ω_c)
    pos_count  = np.zeros(n_sub, float)   # Σ [ω_c > 0]
    unsigned   = np.zeros(n_sub, float)   # Σ |ω_c|
    counts     = np.zeros(n_sub, float)   # deg

    s = np.sign(omega_c)  # +1, -1, or 0

    np.add.at(sign_sum,  sub_src, s);       np.add.at(sign_sum, sub_tgt, s)
    np.add.at(pos_count, sub_src, (omega_c > 0).astype(float))
    np.add.at(pos_count, sub_tgt, (omega_c > 0).astype(float))
    np.add.at(unsigned,  sub_src, np.abs(omega_c))
    np.add.at(unsigned,  sub_tgt, np.abs(omega_c))
    np.add.at(counts,    sub_src, 1.);      np.add.at(counts, sub_tgt, 1.)

    mask = counts > 0
    coherence   = np.zeros(n_sub)
    pos_frac    = np.zeros(n_sub)
    unsigned_d  = np.zeros(n_sub)

    coherence[mask]  = np.abs(sign_sum[mask]) / counts[mask]
    pos_frac[mask]   = pos_count[mask]       / counts[mask]
    unsigned_d[mask] = unsigned[mask]        / counts[mask]

    return coherence, pos_frac, unsigned_d


# ══════════════════════════════════════════════════════════════════════════════
# SPECTRAL LAYER — eigenmode decomposition of coexact field
# ══════════════════════════════════════════════════════════════════════════════

def compute_spectral(omega_c: np.ndarray,
                     B1: sp.csr_matrix,
                     k_eigs: int = 20
                     ) -> dict:
    """
    Decompose ω_coexact in eigenmodes of the interface 1-Laplacian L1=B1@B1.T.
    L1 acts on edge space: shape (n_edges, n_edges).

    Returns dict with spectral metrics.
    """
    n_edges = B1.shape[0]
    k_use   = min(k_eigs, n_edges - 2)
    result  = dict(spectral_entropy=np.nan, participation_ratio=np.nan,
                   dominant_mode_fraction=np.nan, n_eigenmodes=0)
    if k_use < 2: return result

    try:
        L1 = B1.dot(B1.T)
        vals, vecs = spla.eigsh(L1, k=k_use, which='SM', tol=1e-6)
    except Exception:
        return result

    order = np.argsort(vals)
    vals  = vals[order];  vecs = vecs[:, order]

    # Spectral coefficients: α_k = <ω_c, φ_k>²
    alpha = np.array([float(np.dot(omega_c, vecs[:, i]))**2
                      for i in range(k_use)])

    # Only non-trivial modes (non-zero eigenvalue)
    # The null space of L1 = B1 B1^T are the tree edges (cycle-free);
    # coexact structure lives in modes with eigenvalue > 0.
    nz = vals > 1e-8
    alpha_nz = alpha[nz]
    total = float(alpha_nz.sum())
    if total < 1e-12: return result

    p = alpha_nz / total

    # Spectral entropy: high = coexact energy spread across many cycle modes (diffuse)
    #                   low  = energy concentrated in 1–2 modes (organized circulation)
    s_ent = float(-np.dot(p[p > 0], np.log(p[p > 0])))

    # Participation ratio: 1/k = one mode dominates; 1 = energy spread uniformly
    # PR = 1 / (k * IPR)  where IPR = Σα²/(Σα)²
    ipr = float(np.dot(alpha_nz, alpha_nz) / (total**2))
    pr  = float(1.0 / (len(alpha_nz) * ipr)) if ipr > 0 else np.nan

    # Dominant mode fraction: max(α_k)/Σα_k
    dom_frac = float(alpha_nz.max() / total)

    result.update(spectral_entropy=s_ent,
                  participation_ratio=pr,
                  dominant_mode_fraction=dom_frac,
                  n_nz_modes=int(nz.sum()),
                  n_eigenmodes=int(k_use))
    result.pop('low_freq_frac', None)   # removed — SM-only is always "low freq"
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SPATIAL FIGURE
# ══════════════════════════════════════════════════════════════════════════════

def plot_spatial(sid: str, coords: np.ndarray,
                 density_full: np.ndarray,
                 signed_full: np.ndarray,
                 interface_mask: np.ndarray,
                 resp: str,
                 out_path: Path) -> None:
    """
    Two-panel spatial figure:
      Left:  unsigned coexact density (c_i) — identifies hotspots
      Right: signed circulation (s_i)   — identifies directionality
    Non-interface nodes shown as grey background.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f'{sid}  [{resp}]\n'
        f'Coexact density (left) and signed circulation (right)',
        fontsize=11
    )
    x, y = coords[:, 0], coords[:, 1]
    col  = COLOURS.get(resp, 'grey')

    for ax, vals, cmap, title in [
        (axes[0], density_full, 'YlOrRd',
         'Unsigned coexact density  c_i\n(interface subgraph; hotspot = high)'),
        (axes[1], signed_full,  'YlGn',
         'Local sign coherence  lsc_i\n(high = directionally organized edges)'),
    ]:
        # Background: non-interface nodes
        bg = ~interface_mask
        ax.scatter(x[bg], y[bg], c='#eeeeee', s=6, rasterized=True, zorder=1)

        # Interface nodes coloured by metric
        iface = interface_mask
        vdata = vals[iface]
        if cmap == 'YlOrRd':
            vmin, vmax = 0., np.percentile(vdata[vdata > 0], 95) if vdata.any() else 1.
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        else:
            abs_max = max(abs(np.percentile(vdata, 1)),
                          abs(np.percentile(vdata, 99)), 1e-8)
            norm = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)

        sc = ax.scatter(x[iface], y[iface], c=vdata,
                        s=14, cmap=cmap, norm=norm,
                        rasterized=True, zorder=2)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


def plot_cohort_summary(records: list[dict], out_path: Path) -> None:
    """
    Two-panel summary:
      Left:  hotspot Gini (concentration) by response
      Right: Shannon entropy (diffuseness) by response
    """
    df  = pd.DataFrame(records)
    df  = df[df['status'] == 'ok']
    if df.empty: return

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('HCC — Spatial coexact hotspot organization\n'
                 'by response group (quality-filtered)', fontsize=11)

    rng = np.random.RandomState(42)
    for ax, metric, ylabel in [
        (axes[0], 'hotspot_gini',
         'Gini coefficient of coexact density\n(1=maximally concentrated, 0=uniform)'),
        (axes[1], 'hotspot_entropy',
         'Shannon entropy of coexact density\n(high=diffuse, low=concentrated)'),
    ]:
        for xi, (resp_val, col) in enumerate([
            ('Responder', '#d62728'), ('Non_Responder', '#1f77b4')
        ]):
            grp  = df[df['response_group'] == resp_val]
            vals = grp[metric].dropna().values
            if len(vals) == 0: continue
            jx   = xi + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(jx, vals, color=col, s=75, zorder=3, alpha=0.9)
            ax.plot([xi-.3, xi+.3], [np.median(vals)]*2,
                    color=col, lw=2.5, zorder=4)
            for i, (_, row) in enumerate(grp.iterrows()):
                ax.text(xi+.22, vals[i],
                        row['sample_id'].replace('cytassist_',''),
                        fontsize=7, va='center', color=col)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Responder', 'Non-Responder'], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)

    fig.legend(
        handles=[mpatches.Patch(color='#d62728', label='Responder'),
                 mpatches.Patch(color='#1f77b4', label='Non-Responder')],
        fontsize=9, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PER-SAMPLE PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def process_sample(sid: str, sub: sc.AnnData,
                   prog_a: str, prog_b: str, prog_immune: str,
                   k: int, ridge: float,
                   include_border: bool, k_eigs: int,
                   fig_dir: Path, h5ad_dir: Path,
                   layers: set) -> dict:

    n      = sub.n_obs
    coords = sub.obsm['spatial']
    A      = sub.obs[prog_a].values.astype(float)      # tumor
    B      = sub.obs[prog_b].values.astype(float)      # exhaustion
    T      = sub.obs[prog_immune].values.astype(float) # tcell

    cid = '_'.join(sid.split('_')[:2])
    tp  = sid.split('_')[-1]
    pid, resp, trt = PATIENT_MAP.get(cid, (None, None, None))
    base = dict(sample_id=sid, cytassist_id=cid, timepoint=tp,
                patient_id=pid, response_group=resp, treatment=trt,
                n_spots=n, status='skip')

    if n < k + 2: return base

    # Interface mask and threshold
    tumor_q75  = float(np.quantile(A, 0.75))
    immune_q75 = float(np.quantile(T, 0.75))
    low_tumor  = tumor_q75 < TUMOR_FLOOR
    iface_mask = (A > tumor_q75) & (T > immune_q75)
    n_iface    = int(iface_mask.sum())

    if low_tumor:
        print(f'  [{sid}] LOW TUMOR — complete response pattern')
        return {**base, 'status':'low_tumor', 'n_interface':n_iface}

    if n_iface < MIN_IFACE:
        print(f'  [{sid}] SMALL INTERFACE (n={n_iface})')
        return {**base, 'status':'small_interface', 'n_interface':n_iface}

    # Build global kNN graph
    src_g, tgt_g = build_knn_edges(coords, k)
    adj_g        = adj_list(src_g, tgt_g, n)

    # Interface subgraph
    keep_nodes, ss, st = interface_subgraph(iface_mask, adj_g, src_g, tgt_g,
                                            include_border)
    n_sub = len(keep_nodes)
    n_edges = len(ss)
    if n_edges < 4 or n_sub < k + 1:
        return {**base, 'status':'subgraph_too_small', 'n_interface':n_iface}

    A_sub, B_sub = A[keep_nodes], B[keep_nodes]

    # Wedge field
    omega = A_sub[ss] * B_sub[st] - A_sub[st] * B_sub[ss]

    # Hodge decomposition
    B1              = build_B1(ss, st, n_sub)
    omega_e, omega_c = hodge_decompose(omega, B1, ridge)

    # ── CORE LAYER ──────────────────────────────────────────────────────────
    density_sub = compute_core(omega_c, ss, st, n_sub)

    # Identify interface-only nodes within subgraph
    iface_in_sub = np.array([iface_mask[gv] for gv in keep_nodes])
    d_iface      = density_sub[iface_in_sub]

    h_p95  = float(np.percentile(d_iface, 95)) if len(d_iface) else 0.
    h_mean = float(d_iface.mean()) if len(d_iface) else 0.
    h_gini = gini(d_iface)
    h_entr = shannon_entropy(d_iface)
    lcc_f  = largest_hotspot_fraction(density_sub, ss, st, n_sub)

    # Wedge intensity per node (for h5ad)
    wedge_int   = np.zeros(n_sub, float)
    w_abs       = np.abs(omega)
    deg_        = np.zeros(n_sub, float)
    np.add.at(wedge_int, ss, w_abs); np.add.at(wedge_int, st, w_abs)
    np.add.at(deg_,      ss, 1.);    np.add.at(deg_,      st, 1.)
    wedge_int[deg_ > 0] /= deg_[deg_ > 0]

    # coexact/exact ratio per node
    exact_int = np.zeros(n_sub, float); exc = np.abs(omega_e)
    np.add.at(exact_int, ss, exc); np.add.at(exact_int, st, exc)
    exact_int[deg_ > 0] /= deg_[deg_ > 0]
    ce_ratio = density_sub / (exact_int + 1e-12)

    rec = {**base, 'status':'ok', 'n_interface':n_iface,
           'n_subgraph_nodes':n_sub, 'n_edges':n_edges,
           'hotspot_p95':h_p95, 'hotspot_mean':h_mean,
           'hotspot_gini':h_gini, 'hotspot_entropy':h_entr,
           'largest_component_fraction':lcc_f}

    # ── SIGNED LAYER ─────────────────────────────────────────────────────────
    s_coh = np.zeros(n_sub, float)  # default; overwritten if signed layer runs
    signed_rec = {}
    if 'signed' in layers:
        s_coh, pos_frac, s_dens = compute_signed(omega_c, ss, st, n_sub)
        coh_iface = s_coh[iface_in_sub]
        pf_iface  = pos_frac[iface_in_sub]
        # Edge-level sign distribution
        ccw_frac_edge = float((omega_c > 0).mean()) if len(omega_c) else 0.5
        cw_frac_edge  = float((omega_c < 0).mean()) if len(omega_c) else 0.5
        edge_sign_entr = shannon_entropy(np.abs(omega_c))
        signed_rec = {
            'orientation_coherence':   float(coh_iface.mean()) if len(coh_iface) else 0.,
            'pos_fraction_node':       float(pf_iface.mean()) if len(pf_iface) else 0.5,
            'ccw_fraction_edge':       ccw_frac_edge,
            'cw_fraction_edge':        cw_frac_edge,
            'cw_ccw_edge_bias':        float(ccw_frac_edge - cw_frac_edge),
            'edge_sign_entropy':       edge_sign_entr,
        }
        rec.update(signed_rec)
    else:
        s_dens = np.zeros(n_sub, float)

    # ── SPECTRAL LAYER ───────────────────────────────────────────────────────
    spec_rec = {}
    if 'spectral' in layers:
        spec_rec = compute_spectral(omega_c, B1, k_eigs)
        rec.update({f'spec_{k2}': v for k2, v in spec_rec.items()})

    print(f'  [{sid}]  iface={n_iface:4d}  gini={h_gini:.3f}  '
          f'entr={h_entr:.3f}  lcc={lcc_f:.3f}  p95={h_p95:.1f}')

    # ── SPATIAL FIGURE ───────────────────────────────────────────────────────
    if 'core' in layers:
        # Map back to full tissue (all n spots)
        density_full = np.zeros(n, float)
        signed_full  = np.zeros(n, float)
        density_full[keep_nodes] = density_sub
        signed_full[keep_nodes]  = s_coh  if 'signed' in layers else np.zeros(n_sub)

        fig_path = fig_dir / f'fig_hcc_coexact_density_{sid}.png'
        plot_spatial(sid, coords, density_full, signed_full,
                     iface_mask, resp or 'Unknown', fig_path)

    # ── SAVE H5AD ────────────────────────────────────────────────────────────
    out_h5 = h5ad_dir / f'hcc_{sid}_coexact.h5ad'
    sub_out = sub.copy()
    density_full2 = np.zeros(n, float)
    signed_full2  = np.zeros(n, float)
    wedge_full    = np.zeros(n, float)
    ce_ratio_full = np.zeros(n, float)
    iface_arr     = np.zeros(n, bool)
    density_full2[keep_nodes] = density_sub
    signed_full2[keep_nodes]  = s_coh  if 'signed' in layers else 0.
    wedge_full[keep_nodes]    = wedge_int
    ce_ratio_full[keep_nodes] = ce_ratio
    iface_arr[:] = iface_mask

    sub_out.obs['coexact_density']    = density_full2
    sub_out.obs['sign_coherence']     = signed_full2
    sub_out.obs['coexact_exact_ratio']= ce_ratio_full
    sub_out.obs['wedge_intensity']    = wedge_full
    sub_out.obs['interface_mask']     = iface_arr.astype(float)
    sub_out.write(out_h5)

    return rec


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Spatial coexact localization analysis')
    ap.add_argument('--adata',       default='data/hepatocellular_carcinoma/hcc_scored.h5ad')
    ap.add_argument('--prog-a',      default='tumor_score')
    ap.add_argument('--prog-b',      default='exhaustion_score')
    ap.add_argument('--prog-immune', default='tcell_score')
    ap.add_argument('--k',           type=int,   default=6)
    ap.add_argument('--ridge',       type=float, default=1e-8)
    ap.add_argument('--k-eigs',      type=int,   default=20)
    ap.add_argument('--layers',      nargs='+',
                    default=['core', 'signed', 'spectral'],
                    choices=['core', 'signed', 'spectral'])
    ap.add_argument('--no-border',   action='store_true')
    ap.add_argument('--fig-dir',     default='.')
    ap.add_argument('--h5ad-dir',    default='data/hepatocellular_carcinoma')
    ap.add_argument('--out-hotspot', default='results_hcc_coexact_hotspots.csv')
    ap.add_argument('--out-signed',  default='results_hcc_signed_orientation.csv')
    ap.add_argument('--out-spectral',default='results_hcc_coexact_spectrum.csv')
    ap.add_argument('--sample-id',   default=None)
    args = ap.parse_args()

    layers          = set(args.layers)
    include_border  = not args.no_border
    fig_dir         = Path(args.fig_dir);  fig_dir.mkdir(parents=True, exist_ok=True)
    h5ad_dir        = Path(args.h5ad_dir); h5ad_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading {args.adata} …')
    adata = sc.read_h5ad(args.adata)
    print(f'  {adata.n_obs:,} spots  '
          f'{args.prog_a} × {args.prog_b}  k={args.k}  '
          f'layers={sorted(layers)}')

    for p in (args.prog_a, args.prog_b, args.prog_immune):
        if p not in adata.obs.columns:
            raise RuntimeError(f"'{p}' not in .obs — run build_program_scores.py")

    samples = ([args.sample_id] if args.sample_id
               else sorted(adata.obs['sample_id'].unique()))
    print(f'\nProcessing {len(samples)} samples …\n')

    records = []
    for sid in samples:
        sub = adata[adata.obs['sample_id'] == sid].copy()
        rec = process_sample(sid, sub, args.prog_a, args.prog_b,
                             args.prog_immune, args.k, args.ridge,
                             include_border, args.k_eigs,
                             fig_dir, h5ad_dir, layers)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(args.out_hotspot, index=False)
    print(f'\nSaved → {args.out_hotspot}')

    # Signed and spectral CSVs
    if 'signed' in layers:
        signed_cols = ['sample_id','patient_id','response_group','timepoint',
                       'n_interface','signed_mean','orientation_coherence',
                       'cw_fraction','ccw_fraction','cw_ccw_bias','circulation_entropy']
        signed_cols = [c for c in signed_cols if c in df.columns]
        df[signed_cols].to_csv(args.out_signed, index=False)
        print(f'Saved → {args.out_signed}')

    if 'spectral' in layers:
        spec_cols = ['sample_id','patient_id','response_group','timepoint',
                     'n_interface'] + [c for c in df.columns if c.startswith('spec_')]
        spec_cols = [c for c in spec_cols if c in df.columns]
        df[spec_cols].to_csv(args.out_spectral, index=False)
        print(f'Saved → {args.out_spectral}')

    # ── Cohort summary figure ─────────────────────────────────────────────
    plot_cohort_summary(records,
                        fig_dir / 'fig_hcc_coexact_spatial_summary.png')
    print(f'Saved → {fig_dir}/fig_hcc_coexact_spatial_summary.png')

    # ── Statistics ────────────────────────────────────────────────────────
    ok = df[df['status'] == 'ok']
    if len(ok) == 0: print('No valid samples.'); return

    print('\n' + '='*65)
    print('SPATIAL HOTSPOT STATISTICS (quality-filtered, ok status)')
    print('='*65)
    print(f"{'sample':<22} {'R/NR':>4} {'n_iface':>8} "
          f"{'gini':>7} {'entropy':>8} {'lcc':>7} {'p95':>10}")
    print('─'*72)
    for resp_val in ['Responder','Non_Responder']:
        tag = 'R' if resp_val == 'Responder' else 'NR'
        for _, r in ok[ok['response_group']==resp_val].iterrows():
            print(f"{r['sample_id']:<22} {tag:>4} "
                  f"{int(r['n_interface']):>8} "
                  f"{r['hotspot_gini']:>7.4f} "
                  f"{r['hotspot_entropy']:>8.4f} "
                  f"{r['largest_component_fraction']:>7.4f} "
                  f"{r['hotspot_p95']:>10.2f}")
        print()

    print('GROUP MEDIANS')
    for resp_val, label in [('Responder','R'), ('Non_Responder','NR')]:
        g = ok[ok['response_group'] == resp_val]
        print(f"  {label}  n={len(g)}  "
              f"gini={g['hotspot_gini'].median():.3f}  "
              f"entropy={g['hotspot_entropy'].median():.3f}  "
              f"lcc={g['largest_component_fraction'].median():.3f}")

    if 'signed' in layers and 'orientation_coherence' in ok.columns:
        print('\nORIENTATION COHERENCE (mean of |s_i|/c_i at interface nodes)')
        for resp_val, label in [('Responder','R'),('Non_Responder','NR')]:
            g = ok[ok['response_group']==resp_val]['orientation_coherence']
            print(f"  {label}  median={g.median():.3f}  "
                  f"range=[{g.min():.3f}, {g.max():.3f}]")

    if 'spectral' in layers and 'spec_spectral_entropy' in ok.columns:
        print('\nSPECTRAL ENTROPY (higher=diffuse, lower=organized modes)')
        for resp_val, label in [('Responder','R'),('Non_Responder','NR')]:
            g = ok[ok['response_group']==resp_val]['spec_spectral_entropy']
            print(f"  {label}  median={g.median():.3f}  "
                  f"range=[{g.min():.3f}, {g.max():.3f}]")
        print('\nDOMINANT MODE FRACTION (max α_k / Σα_k; 1=one cycle dominates)')
        for resp_val, label in [('Responder','R'),('Non_Responder','NR')]:
            col = 'spec_dominant_mode_fraction'
            if col in ok.columns:
                g = ok[ok['response_group']==resp_val][col].dropna()
                if len(g):
                    print(f"  {label}  n={len(g)}  median={g.median():.3f}  "
                          f"range=[{g.min():.3f}, {g.max():.3f}]")


if __name__ == '__main__':
    main()
