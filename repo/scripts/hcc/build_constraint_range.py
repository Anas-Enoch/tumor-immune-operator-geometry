#!/usr/bin/env python3
"""
build_constraint_range.py — Finite-Range Constraint Analysis (Step 23)
=======================================================================
Tests whether the interface coexact density field is:
  (a) random (no spatial organisation),
  (b) globally periodic/spectral (long-range periodicity), or
  (c) locally constrained (short-range organisation, finite propagation range).

This analysis does NOT add a new biological claim. It characterises the
SPATIAL ORGANISATION TYPE of the coexact field and validates the claim
that responder-associated structures are finite-range constrained, not
random and not globally periodic.

Three components
─────────────────
Step 23a — Graph spectral power spectrum
    Project coexact density onto ordered Laplacian eigenmodes (low→high freq).
    Ask: is spectral power concentrated in low-frequency modes (globally periodic)
    or spread across many modes (locally constrained/random)?
    → Negative control: confirms interface is NOT globally periodic.

Step 23b — Local vs global prediction gap
    local_R²:   predict node i coexact density from k-NN mean of neighbours.
    global_R²:  predict from first k Laplacian eigenmodes (global spectral fit).
    Key metric: Δ = local_R² − global_R²
    If Δ > 0: local geometry predicts better than global modes → finite-range constraint.
    If Δ ≈ 0: both equally predictive (globally organised or random).

Step 23c — Spatial autocorrelation (empirical ACF)
    Compute Moran-style spatial autocorrelation at multiple distance lags.
    Fit exponential decay: r(d) ≈ exp(−d/λ) to extract correlation length λ.
    Short λ = finite-range; long λ = globally correlated (periodic/clustered).

Key metric for KTS comparison
──────────────────────────────
    local_R² − global_R²  (local-global prediction gap)

Expected:
  S3/S4 (responder front instability): high gap — locally organised, not spectral
  S1    (NR fragmented):               lower gap — either random or near-zero both

Inputs
──────
  data/hepatocellular_carcinoma/hcc_*_coexact.h5ad  (from build_coexact_spatial.py)
  results_hcc_kts_states.csv

Outputs
────────
  results_hcc_constraint_range.csv
  fig_hcc_constraint_range.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PATIENT_MAP = {
    'cytassist_70': (85557,'Responder'),    'cytassist_71': (85558,'Responder'),
    'cytassist_72': (85559,'Non_Responder'),'cytassist_73': (85560,'Non_Responder'),
    'cytassist_74': (85561,'Non_Responder'),'cytassist_76': (85563,'Responder'),
    'cytassist_79': (85565,'Non_Responder'),'cytassist_83': (85566,'Responder'),
    'cytassist_84': (85567,'Responder'),    'cytassist_85': (85568,'Non_Responder'),
    'cytassist_86': (85569,'Non_Responder'),
}
COLOURS = {'Responder': '#d62728', 'Non_Responder': '#1f77b4'}
EXCLUDE = {'S0', 'S?'}


# ── Step 23a: graph spectral power ─────────────────────────────────────────

def spectral_power(coords: np.ndarray, field: np.ndarray,
                   k_nn: int = 6, n_modes: int = 20
                   ) -> tuple[np.ndarray, np.ndarray]:
    """
    Project field onto ordered Laplacian eigenmodes (low → high frequency).
    Returns (eigenvalues, squared_projections / total_variance).
    """
    n = len(field)
    if n < n_modes + 3:
        return np.array([]), np.array([])

    nbrs = NearestNeighbors(n_neighbors=min(k_nn+1,n)).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    rows, cols, data = [], [], []
    for i, row_idx in enumerate(idx):
        for j in row_idx[1:]:
            rows.extend([i,j]); cols.extend([j,i]); data.extend([1.,1.])
    A   = sp.coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
    deg = np.array(A.sum(axis=1)).ravel()
    L   = sp.diags(deg) - A

    k_use = min(n_modes, n-2)
    try:
        vals, vecs = spla.eigsh(L.tocsr(), k=k_use, which='SM', tol=1e-6)
    except Exception:
        return np.array([]), np.array([])

    order = np.argsort(vals)
    vals  = vals[order]; vecs = vecs[:, order]

    f_centered = field - field.mean()
    tot_var    = max(np.dot(f_centered, f_centered), 1e-12)
    power      = np.array([np.dot(f_centered, vecs[:,k])**2 for k in range(k_use)])
    return vals, power / tot_var


# ── Step 23b: local vs global prediction ──────────────────────────────────

def local_r2(coords: np.ndarray, field: np.ndarray,
             k_nn: int = 6) -> float:
    """
    Leave-one-out k-NN mean prediction R².
    Each node is predicted from the mean of its k nearest neighbours.
    """
    n = len(field)
    nbrs = NearestNeighbors(n_neighbors=min(k_nn+1,n)).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    pred = np.array([field[row[1:]].mean() for row in idx])
    ss_res = np.sum((field - pred)**2)
    ss_tot = np.sum((field - field.mean())**2)
    return float(1. - ss_res / ss_tot) if ss_tot > 1e-12 else 0.


def global_r2(coords: np.ndarray, field: np.ndarray,
              k_nn: int = 6, n_modes: int = 5) -> float:
    """
    Global graph Laplacian eigenmode prediction R².
    Project field onto first n_modes eigenmodes, then reconstruct.
    """
    n = len(field)
    if n < n_modes + 3:
        return 0.

    nbrs = NearestNeighbors(n_neighbors=min(k_nn+1,n)).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    rows, cols, data = [], [], []
    for i, row_idx in enumerate(idx):
        for j in row_idx[1:]:
            rows.extend([i,j]); cols.extend([j,i]); data.extend([1.,1.])
    A   = sp.coo_matrix((data,(rows,cols)),shape=(n,n)).tocsr()
    deg = np.array(A.sum(axis=1)).ravel()
    L   = (sp.diags(deg) - A).tocsr()

    k_use = min(n_modes, n-2)
    try:
        _, vecs = spla.eigsh(L, k=k_use, which='SM', tol=1e-6)
    except Exception:
        return 0.

    # Reconstruct field as projection onto first k_use eigenmodes
    coeffs = vecs.T @ field
    pred   = vecs @ coeffs

    ss_res = np.sum((field - pred)**2)
    ss_tot = np.sum((field - field.mean())**2)
    return float(1. - ss_res / ss_tot) if ss_tot > 1e-12 else 0.


# ── Step 23c: spatial autocorrelation ─────────────────────────────────────

def spatial_acf(coords: np.ndarray, field: np.ndarray,
                n_bins: int = 10) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Empirical spatial autocorrelation at binned distance lags.
    Returns (lag_distances, correlations, correlation_length_lambda).

    Correlation length estimated by fitting r(d) = exp(−d/λ).
    """
    n = len(field)
    if n < 4:
        return np.array([]), np.array([]), np.nan

    # Pairwise distances (subsample if n > 400 for speed)
    rng = np.random.RandomState(42)
    if n > 400:
        idx = rng.choice(n, 400, replace=False)
        c   = coords[idx]; f = field[idx]
    else:
        c = coords; f = field

    diff = c[:, None, :] - c[None, :, :]   # (n,n,2)
    dist = np.sqrt((diff**2).sum(axis=2))   # (n,n)

    f_z  = (f - f.mean()) / max(f.std(), 1e-12)
    corr_mat = np.outer(f_z, f_z)           # (n,n)

    # Bin by distance
    d_flat = dist[np.triu_indices_from(dist, k=1)]
    c_flat = corr_mat[np.triu_indices_from(dist, k=1)]

    bins  = np.percentile(d_flat, np.linspace(5, 95, n_bins+1))
    bins  = np.unique(bins)
    if len(bins) < 3:
        return np.array([]), np.array([]), np.nan

    lag_d = []; lag_c = []
    for i in range(len(bins)-1):
        mask = (d_flat >= bins[i]) & (d_flat < bins[i+1])
        if mask.sum() > 5:
            lag_d.append(float(np.median(d_flat[mask])))
            lag_c.append(float(np.mean(c_flat[mask])))

    lag_d = np.array(lag_d); lag_c = np.array(lag_c)
    if len(lag_d) < 3:
        return lag_d, lag_c, np.nan

    # Fit exponential decay
    try:
        def exp_decay(d, lam): return np.exp(-d / lam)
        popt, _ = curve_fit(exp_decay, lag_d,
                            np.clip(lag_c, 0, 1),
                            p0=[lag_d.mean()],
                            bounds=(1e-3, np.inf),
                            maxfev=1000)
        lam = float(popt[0])
    except Exception:
        lam = np.nan

    return lag_d, lag_c, lam


# ── Per-sample processing ──────────────────────────────────────────────────

def process_sample(sid: str, h5ad_path: Path,
                   kts_state: str, k_nn: int,
                   n_global_modes: int) -> dict | None:
    if not h5ad_path.exists():
        return None

    adata = sc.read_h5ad(h5ad_path)
    cid   = '_'.join(sid.split('_')[:2])
    tp    = sid.split('_')[-1]
    pid, resp = PATIENT_MAP.get(cid, (None, None))

    c     = np.asarray(adata.obs['coexact_density'].values, dtype=np.float64)
    iface = np.asarray(adata.obs['interface_mask'].values,  dtype=bool)

    if iface.sum() < 10:
        return None

    coords_if = np.ascontiguousarray(adata.obsm['spatial'][iface], dtype=np.float64)
    c_if      = c[iface]

    if c_if.std() < 1e-10:
        return None

    # 23a: spectral power (low-frequency concentration)
    evals, power = spectral_power(coords_if, c_if, k_nn, n_modes=20)
    if len(power) >= 5:
        # Fraction of variance in lowest 5 modes
        lf_frac = float(power[:5].sum())
        # Spectral entropy
        p_n = power[power > 0] / power.sum() if power.sum() > 0 else power
        spec_entropy = float(-np.dot(p_n, np.log(p_n+1e-12)))
    else:
        lf_frac = np.nan; spec_entropy = np.nan

    # 23b: local vs global prediction gap
    r2_local  = local_r2(coords_if,  c_if, k_nn)
    r2_global = global_r2(coords_if, c_if, k_nn, n_global_modes)
    local_global_gap = r2_local - r2_global

    # 23c: spatial ACF and correlation length
    _, _, corr_len = spatial_acf(coords_if, c_if, n_bins=10)

    rec = dict(
        sample_id    = sid,
        cytassist_id = cid,
        timepoint    = tp,
        patient_id   = pid,
        Response     = resp,
        kts_state    = kts_state,
        n_interface  = int(iface.sum()),
        r2_local     = r2_local,
        r2_global    = r2_global,
        local_global_gap = local_global_gap,
        lf_spectral_frac = lf_frac,
        spectral_entropy  = spec_entropy,
        correlation_length = corr_len,
    )
    print(f"  [{sid}] state={kts_state}  "
          f"R²_local={r2_local:.3f}  R²_global={r2_global:.3f}  "
          f"gap={local_global_gap:+.3f}  λ={corr_len:.1f}")
    return rec


# ── Figure ─────────────────────────────────────────────────────────────────

def plot_constraint(df: pd.DataFrame, outpath: Path) -> None:
    ok  = df[~df['kts_state'].isin(EXCLUDE)].copy()
    metrics = [
        ('local_global_gap', 'Local–global prediction gap\n(local R² − global R²)',
         'High: locally constrained\nLow: random or globally periodic'),
        ('r2_local',         'Local prediction R²\n(k-NN neighbourhood)',
         'How well k-NN predicts coexact density'),
        ('r2_global',        'Global prediction R²\n(Laplacian eigenmodes)',
         'How well low-frequency modes predict coexact density'),
        ('correlation_length','Spatial correlation length λ\n(exponential ACF decay)',
         'Short λ: finite-range\nLong λ: globally correlated'),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics)+1, 5.5))
    fig.suptitle('Step 23 — Finite-Range Constraint Analysis\n'
                 'Is the interface coexact field locally constrained, '
                 'globally periodic, or random?', fontsize=10)

    rng = np.random.RandomState(42)
    for ax, (col, ylabel, note) in zip(axes, metrics):
        if col not in ok.columns:
            ax.set_visible(False); continue
        for xi, (resp, col_c) in enumerate([
            ('Responder','#d62728'), ('Non_Responder','#1f77b4')
        ]):
            grp  = ok[ok['Response']==resp]
            vals = grp[col].dropna().values
            if len(vals) == 0: continue
            jx   = xi + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(jx, vals, color=col_c, s=65, zorder=3, alpha=0.9)
            ax.plot([xi-.28,xi+.28],[np.median(vals)]*2,color=col_c,lw=2.5,zorder=4)
            for i,(_, row) in enumerate(grp.dropna(subset=[col]).iterrows()):
                ax.text(xi+.22, vals[i],
                        row['sample_id'].replace('cytassist_',''),
                        fontsize=6.5, va='center', color=col_c)

        if col == 'local_global_gap':
            ax.axhline(0, color='grey', lw=1.0, ls='--', alpha=0.7)
        ax.set_xticks([0,1])
        ax.set_xticklabels(['R','NR'], fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(note, fontsize=7.5, style='italic')

        r_v  = ok[ok['Response']=='Responder'][col].dropna()
        nr_v = ok[ok['Response']=='Non_Responder'][col].dropna()
        if len(r_v)>1 and len(nr_v)>1:
            _, p = mannwhitneyu(r_v, nr_v, alternative='two-sided')
            ax.set_xlabel(f'p = {p:.3f}', fontsize=8)

    fig.legend(handles=[
        mpatches.Patch(color='#d62728', label='Responder'),
        mpatches.Patch(color='#1f77b4', label='Non-Responder'),
    ], fontsize=9, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.04))
    fig.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches='tight')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--kts-csv',        default='results_hcc_kts_states.csv')
    ap.add_argument('--h5ad-dir',       default='data/hepatocellular_carcinoma')
    ap.add_argument('--k-nn',           type=int, default=6)
    ap.add_argument('--n-global-modes', type=int, default=5,
                    help='Number of Laplacian eigenmodes for global prediction')
    ap.add_argument('--out', default='results_hcc_constraint_range.csv')
    ap.add_argument('--fig', default='fig_hcc_constraint_range.png')
    args = ap.parse_args()

    kts      = pd.read_csv(args.kts_csv)
    h5ad_dir = Path(args.h5ad_dir)

    print(f"\nProcessing samples …\n")
    records = []
    for _, row in kts.iterrows():
        sid   = row['sample_id']
        state = row['state']
        if state in EXCLUDE: continue
        h5  = h5ad_dir / f'hcc_{sid}_coexact.h5ad'
        rec = process_sample(sid, h5, state,
                             args.k_nn, args.n_global_modes)
        if rec: records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}')

    ok = df[~df['kts_state'].isin(EXCLUDE)]

    print('\n' + '='*65)
    print('CONSTRAINT RANGE ANALYSIS — R vs NR')
    print('='*65)
    for col, label in [
        ('local_global_gap',   'Local–global gap (Δ = R²_local − R²_global)'),
        ('r2_local',           'Local R² (k-NN prediction)'),
        ('r2_global',          'Global R² (eigenmode prediction)'),
        ('correlation_length', 'Spatial correlation length λ'),
        ('lf_spectral_frac',   'Low-frequency spectral fraction (23a)'),
    ]:
        if col not in ok.columns: continue
        r  = ok[ok['Response']=='Responder'][col].dropna()
        nr = ok[ok['Response']=='Non_Responder'][col].dropna()
        if len(r)<2 or len(nr)<2: continue
        _, p2 = mannwhitneyu(r, nr, alternative='two-sided')
        _, p1 = mannwhitneyu(r, nr, alternative='greater')
        sig   = '**' if p1<0.05 else ('~' if p1<0.15 else '')
        print(f"\n  {label}")
        print(f"    R  median={r.median():.4f}  range=[{r.min():.3f},{r.max():.3f}]")
        print(f"    NR median={nr.median():.4f}  range=[{nr.min():.3f},{nr.max():.3f}]")
        print(f"    p(two-sided)={p2:.4f}  p(R>NR)={p1:.4f} {sig}")

    print('\n' + '='*65)
    print('LOCAL-GLOBAL GAP BY KTS STATE')
    print('='*65)
    for state in ['S2','S3','S4','S1']:
        sub = ok[ok['kts_state']==state]
        if sub.empty: continue
        gap = sub['local_global_gap'].dropna()
        print(f"  {state}  n={len(sub)}  "
              f"gap median={gap.median():.4f}  "
              f"range=[{gap.min():.3f},{gap.max():.3f}]  "
              f"resp={sub['Response'].value_counts().to_dict()}")

    print('\n  Interpretation:')
    print('  gap > 0: local neighbourhood better than global modes → finite-range constraint')
    print('  gap < 0: global modes better → long-range structure (or overfitting)')
    print('  gap ≈ 0: random field OR both methods equally poor')

    plot_constraint(df, Path(args.fig))
    print(f'\nFigure → {args.fig}')


if __name__ == '__main__':
    main()
