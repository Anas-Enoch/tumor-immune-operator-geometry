#!/usr/bin/env python3
"""
build_ks_instability.py — KS-Inspired Interface Instability Proxies
====================================================================
Tests whether the coexact density field at tumor–immune interfaces
shows characteristics consistent with KS-like boundary instabilities:
preferential amplification, roughening, anti-diffusion, and nonlinear
self-amplification at the interface relative to bulk tissue.

Conceptual connection
──────────────────────
KS equation:  u_t + u·u_x + ν·u_xx + κ·u_xxxx = 0

  u·u_x       → self-advection / nonlinear amplification
  ν·u_xx      → negative diffusion (destabilising when ν < 0)
  κ·u_xxxx    → biharmonic smoothing (stabilising)
  Result       → bounded chaotic roughening patterns at boundaries

The coexact density field c_i from the Hodge decomposition is the
spatial analog of u. We ask: does c_i at the interface exhibit
signatures of this instability class?

KS proxy metrics
─────────────────
CDIS (Coexact Density Interface Score)
    = mean c_i(interface) / mean c_i(bulk)
    CDIS >> 1: interface is a boundary-instability amplification hotspot.
    Proxy for: preferential amplification of non-gradient modes at boundary.

Roughness index (CV)
    = std c_i(interface) / mean c_i(interface)  [coefficient of variation]
    High CV: the interface field has sharp local peaks and troughs —
    spatially rough in the KS sense.
    Proxy for: the roughening term producing high-frequency spatial variation.

Anti-diffusion fraction (ADF)
    = fraction of interface nodes where Δ²c_i < 0
    (Δ² = discrete graph Laplacian of c)
    Negative Laplacian means the node is locally convex — higher than
    its neighbours. This resists local smoothing.
    Proxy for: the sign of the ν·u_xx term (destabilising negative diffusion).

Nonlinear amplification index (NAI)
    = Spearman ρ(c_i, |Δ²c_i|) at interface nodes
    Positive ρ: dense nodes also have steep local gradients, consistent
    with self-amplification geometry (u·u_x term).
    Proxy for: correlation between amplitude and local gradient — the
    nonlinear advection term.

Critical caution
─────────────────
These are STATIC spatial proxies for instability characteristics that
in KS arise from temporal dynamics. We observe the spatial PATTERN
that instability dynamics would produce, not the dynamics themselves.
No claim is made about actual PDE dynamics in tissue. The KS framework
provides a formal vocabulary for classifying the spatial organisation
of the non-gradient interface field.

Inputs
──────
data/hepatocellular_carcinoma/hcc_*_coexact.h5ad  (from build_coexact_spatial.py)
results_hcc_kts_states.csv                         (KTS state assignments)

Outputs
────────
results_hcc_ks_instability.csv
fig_hcc_ks_instability.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PATIENT_MAP = {
    'cytassist_70': (85557, 'Responder'),    'cytassist_71': (85558, 'Responder'),
    'cytassist_72': (85559, 'Non_Responder'),'cytassist_73': (85560, 'Non_Responder'),
    'cytassist_74': (85561, 'Non_Responder'),'cytassist_76': (85563, 'Responder'),
    'cytassist_79': (85565, 'Non_Responder'),'cytassist_83': (85566, 'Responder'),
    'cytassist_84': (85567, 'Responder'),    'cytassist_85': (85568, 'Non_Responder'),
    'cytassist_86': (85569, 'Non_Responder'),
}
COLOURS  = {'Responder': '#d62728', 'Non_Responder': '#1f77b4'}
EXCLUDE  = {'S0', 'S?'}


# ── Discrete graph Laplacian ───────────────────────────────────────────────

def graph_laplacian_values(coords: np.ndarray, field: np.ndarray,
                           k: int = 6) -> np.ndarray:
    """
    Discrete graph Laplacian of scalar field at each node:
        Δ²f_i = (1/deg_i) Σ_{j ∈ N(i)} (f_j - f_i)

    Returns array of length n (one value per node).
    Positive: node is locally concave (lower than neighbours).
    Negative: node is locally convex (higher than neighbours).
    """
    n = len(field)
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n)).fit(coords)
    _, idx = nbrs.kneighbors(coords)

    lap = np.zeros(n)
    for i, nbr_row in enumerate(idx):
        neighbours = nbr_row[1:]   # exclude self
        if len(neighbours) == 0:
            continue
        lap[i] = (field[neighbours] - field[i]).mean()
    return lap


# ── Per-sample processing ──────────────────────────────────────────────────

def process_sample(sid: str, h5ad_path: Path,
                   kts_state: str, k: int) -> dict | None:
    if not h5ad_path.exists():
        return None

    adata = sc.read_h5ad(h5ad_path)
    cid   = '_'.join(sid.split('_')[:2])
    tp    = sid.split('_')[-1]
    pid, resp = PATIENT_MAP.get(cid, (None, None))

    c     = np.asarray(adata.obs['coexact_density'].values, dtype=np.float64)
    iface = np.asarray(adata.obs['interface_mask'].values,  dtype=bool)

    if iface.sum() < 5 or (~iface).sum() < 5:
        return None

    c_iface = c[iface]
    c_bulk  = c[~iface]

    # ── CDIS (interface amplification) ────────────────────────────────────
    mean_iface = float(c_iface.mean())
    mean_bulk  = float(c_bulk.mean())
    cdis = mean_iface / max(mean_bulk, 1e-12)

    # ── Roughness index (CV) ───────────────────────────────────────────────
    std_iface = float(c_iface.std())
    roughness = std_iface / max(mean_iface, 1e-12)   # coefficient of variation

    # ── Discrete graph Laplacian ───────────────────────────────────────────
    coords = np.ascontiguousarray(adata.obsm['spatial'], dtype=np.float64)
    lap    = graph_laplacian_values(coords, c, k=k)
    lap_if = lap[iface]

    # Anti-diffusion fraction: fraction of interface nodes with Δ²c < 0
    adf    = float((lap_if < 0).mean())
    mean_lap_if = float(lap_if.mean())   # signed; negative = overall anti-diffusive

    # ── Nonlinear amplification index ─────────────────────────────────────
    abs_lap = np.abs(lap_if)
    if len(c_iface) >= 5 and abs_lap.std() > 1e-12:
        nai, _ = spearmanr(c_iface, abs_lap)
        nai = float(nai) if not np.isnan(nai) else 0.
    else:
        nai = 0.

    # ── KS instability score (composite) ──────────────────────────────────
    # Normalise each metric to [0,1] range and average.
    # This is a convenience composite for ranking — not a rigorous index.
    # Individual metrics are primary; composite is illustrative.
    # We store all raw metrics so the composite can be recomputed.

    rec = dict(
        sample_id    = sid,
        cytassist_id = cid,
        timepoint    = tp,
        patient_id   = pid,
        Response     = resp,
        kts_state    = kts_state,
        n_interface  = int(iface.sum()),
        n_bulk       = int((~iface).sum()),
        mean_iface   = mean_iface,
        mean_bulk    = mean_bulk,
        cdis         = cdis,
        roughness_cv = roughness,
        adf          = adf,
        mean_laplacian_iface = mean_lap_if,
        nai          = nai,
    )

    print(f"  [{sid}] state={kts_state}  "
          f"CDIS={cdis:.3f}  CV={roughness:.3f}  "
          f"ADF={adf:.3f}  NAI={nai:.3f}")
    return rec


# ── Figure ─────────────────────────────────────────────────────────────────

def plot_instability(df: pd.DataFrame, outpath: Path) -> None:
    metrics = [
        ('cdis',         'CDIS\n(interface amplification ratio)',
         'Boundary hotspot: coexact density\nat interface vs bulk'),
        ('roughness_cv', 'Roughness CV\n(spatial heterogeneity)',
         'KS roughening: high CV = sharp peaks\nand troughs at boundary'),
        ('adf',          'Anti-diffusion fraction\n(fraction of convex interface nodes)',
         'Negative-diffusion signature:\nfraction resisting local smoothing'),
        ('nai',          'Nonlinear amplification index\n(Spearman ρ: density vs |Δ²c|)',
         'Self-amplification: dense nodes\nalso have steepest gradients'),
    ]
    n_met = len(metrics)
    fig, axes = plt.subplots(1, n_met, figsize=(4*n_met+1, 5.5))
    fig.suptitle('KS-Inspired Interface Instability Proxies\n'
                 '(static spatial signatures of boundary instability dynamics)',
                 fontsize=10)

    ok  = df[~df['kts_state'].isin(EXCLUDE)]
    rng = np.random.RandomState(42)

    for ax, (col, ylabel, note) in zip(axes, metrics):
        if col not in ok.columns:
            ax.set_visible(False); continue
        for xi, (resp, col_c) in enumerate([
            ('Responder','#d62728'), ('Non_Responder','#1f77b4')
        ]):
            grp  = ok[ok['Response'] == resp]
            vals = grp[col].dropna().values
            if len(vals) == 0: continue
            jx   = xi + rng.uniform(-0.18, 0.18, len(vals))
            ax.scatter(jx, vals, color=col_c, s=65, zorder=3, alpha=0.9)
            ax.plot([xi-.28, xi+.28], [np.median(vals)]*2,
                    color=col_c, lw=2.5, zorder=4)
            for i, (_, row) in enumerate(grp.dropna(subset=[col]).iterrows()):
                ax.text(xi+.22, vals[i],
                        row['sample_id'].replace('cytassist_',''),
                        fontsize=6.5, va='center', color=col_c)

        ax.axhline(0, color='grey', lw=0.5, ls='--')
        if col == 'cdis':
            ax.axhline(1, color='orange', lw=1.0, ls=':', alpha=0.7,
                       label='CDIS=1 (no amplification)')
            ax.legend(fontsize=7)
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
    ap.add_argument('--kts-csv',  default='results_hcc_kts_states.csv')
    ap.add_argument('--h5ad-dir', default='data/hepatocellular_carcinoma')
    ap.add_argument('--k',        type=int, default=6)
    ap.add_argument('--out',      default='results_hcc_ks_instability.csv')
    ap.add_argument('--fig',      default='fig_hcc_ks_instability.png')
    args = ap.parse_args()

    kts      = pd.read_csv(args.kts_csv)
    h5ad_dir = Path(args.h5ad_dir)

    print(f"\nProcessing samples …\n")
    records = []
    for _, row in kts.iterrows():
        sid   = row['sample_id']
        state = row['state']
        if state in EXCLUDE:
            continue
        h5 = h5ad_dir / f'hcc_{sid}_coexact.h5ad'
        rec = process_sample(sid, h5, state, args.k)
        if rec:
            records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(args.out, index=False)
    print(f'\nSaved → {args.out}')

    ok = df[~df['kts_state'].isin(EXCLUDE)]

    # ── Statistics ────────────────────────────────────────────────────────
    print('\n' + '='*65)
    print('KS INSTABILITY PROXIES — R vs NR')
    print('='*65)
    metrics = [
        ('cdis',         'CDIS (interface amplification)'),
        ('roughness_cv', 'Roughness CV'),
        ('adf',          'Anti-diffusion fraction'),
        ('nai',          'Nonlinear amplification index'),
    ]
    for col, label in metrics:
        if col not in ok.columns: continue
        r  = ok[ok['Response']=='Responder'][col].dropna()
        nr = ok[ok['Response']=='Non_Responder'][col].dropna()
        if len(r)<2 or len(nr)<2: continue
        _, p2 = mannwhitneyu(r, nr, alternative='two-sided')
        _, p1 = mannwhitneyu(r, nr, alternative='greater')
        sig = '**' if p1 < 0.05 else ('~' if p1 < 0.15 else '')
        print(f"\n  {label}")
        print(f"    R  n={len(r)}  values={r.round(3).values.tolist()}  "
              f"median={r.median():.4f}")
        print(f"    NR n={len(nr)}  values={nr.round(3).values.tolist()}  "
              f"median={nr.median():.4f}")
        print(f"    p(two-sided)={p2:.4f}  p(R>NR)={p1:.4f} {sig}")

    print('\n' + '='*65)
    print('KS PROXIES BY KTS STATE')
    print('='*65)
    for state in ['S2','S3','S4','S1']:
        sub = ok[ok['kts_state']==state]
        if sub.empty: continue
        print(f"\n  {state}  n={len(sub)}  "
              f"resp={sub['Response'].value_counts().to_dict()}")
        for col, label in metrics:
            if col not in sub.columns: continue
            print(f"    {label:<35} "
                  f"median={sub[col].median():.4f}  "
                  f"range=[{sub[col].min():.3f}, {sub[col].max():.3f}]")

    print('\n' + '='*65)
    print('CDIS > 1 CHECK (interface is amplification hotspot)')
    print('='*65)
    for resp_val in ['Responder','Non_Responder']:
        sub = ok[ok['Response']==resp_val]
        n_above = (sub['cdis'] > 1).sum()
        print(f"  {resp_val}: CDIS > 1 in {n_above}/{len(sub)} sections")
        for _, row in sub.iterrows():
            print(f"    {row['sample_id']:<25} CDIS={row['cdis']:.3f}  "
                  f"state={row['kts_state']}")

    plot_instability(df, Path(args.fig))
    print(f'\nFigure → {args.fig}')


if __name__ == '__main__':
    main()
