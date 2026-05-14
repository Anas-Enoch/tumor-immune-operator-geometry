#!/usr/bin/env python3

"""
Harmonic biological manifold / group-FFT-inspired analysis.

This is NOT literal group FFT on a biological symmetry group.
It is a cautious harmonic decomposition of interface operator fields.

We assign each interface node an angular coordinate around the interface
centroid and decompose coexact density into circular Fourier harmonics.

Output:
    results_hcc_harmonic_biological_manifold.csv

Interpretation:
    low harmonics  -> globally coherent/exclusion-ring-like field
    mid harmonics  -> finite-range front organization
    high harmonics -> punctate/noisy fragmentation
"""

from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import NearestNeighbors

INFILE = Path("data/hepatocellular_carcinoma/hcc_scored.h5ad")
KTS = Path("results_hcc_kts_states.csv")
OUT = Path("results_hcc_harmonic_biological_manifold.csv")

K = 6
MIN_INTERFACE = 20
RIDGE = 1e-8
N_BINS = 64
MAX_MODE = 16
EPS = 1e-12

RESPONSE = {
    "cytassist_70_pre": "Responder", "cytassist_70_post": "Responder",
    "cytassist_71_pre": "Responder", "cytassist_71_post": "Responder",
    "cytassist_76_pre": "Responder", "cytassist_76_post": "Responder",
    "cytassist_83_pre": "Responder", "cytassist_83_post": "Responder",
    "cytassist_84_pre": "Responder", "cytassist_84_post": "Responder",
    "cytassist_72_pre": "Non_Responder", "cytassist_72_post": "Non_Responder",
    "cytassist_73_pre": "Non_Responder", "cytassist_73_post": "Non_Responder",
    "cytassist_74_pre": "Non_Responder", "cytassist_74_post": "Non_Responder",
    "cytassist_79_pre": "Non_Responder", "cytassist_79_post": "Non_Responder",
    "cytassist_85_pre": "Non_Responder", "cytassist_85_post": "Non_Responder",
    "cytassist_86_pre": "Non_Responder", "cytassist_86_post": "Non_Responder",
}

adata = sc.read_h5ad(INFILE)
kts = pd.read_csv(KTS) if KTS.exists() else None

if "response" not in adata.obs.columns:
    adata.obs["response"] = adata.obs["sample_id"].map(RESPONSE)

def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.sum(x) <= 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((2*np.sum(np.arange(1,n+1)*x)/(n*np.sum(x))) - (n+1)/n)

def entropy(x):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    s = x.sum()
    if s <= 0:
        return np.nan
    p = x / s
    return float(-np.sum(p * np.log(p + EPS)))

def build_edges(coords):
    n = len(coords)
    nbrs = NearestNeighbors(n_neighbors=min(K+1, n)).fit(coords)
    _, knn = nbrs.kneighbors(coords)
    edges = []
    for i in range(n):
        for j in knn[i, 1:]:
            if i != int(j):
                edges.append((i, int(j)))
    return edges

def coexact_density(A, B, edges, n):
    m = len(edges)
    omega = np.zeros(m)
    for e, (i, j) in enumerate(edges):
        omega[e] = A[i]*B[j] - A[j]*B[i]

    rows, cols, vals = [], [], []
    for e, (i, j) in enumerate(edges):
        rows += [e, e]
        cols += [i, j]
        vals += [-1.0, 1.0]

    B1 = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))
    L0 = B1.T @ B1 + RIDGE * sparse.eye(n, format="csr")
    rhs = B1.T @ omega

    alpha = lsqr(L0, rhs, atol=1e-8, btol=1e-8, iter_lim=1000)[0]
    omega_exact = B1 @ alpha
    omega_coexact = omega - omega_exact

    cf = float(np.sum(omega_coexact**2) / (np.sum(omega**2) + EPS))

    dens_sum = np.zeros(n)
    deg = np.zeros(n)
    for val, (i, j) in zip(np.abs(omega_coexact), edges):
        dens_sum[i] += val
        dens_sum[j] += val
        deg[i] += 1
        deg[j] += 1

    return dens_sum / np.maximum(deg, 1), cf

def angular_profile(coords, signal):
    center = coords.mean(axis=0)
    xy = coords - center
    theta = np.arctan2(xy[:,1], xy[:,0])
    theta = (theta + 2*np.pi) % (2*np.pi)

    bins = np.linspace(0, 2*np.pi, N_BINS+1)
    prof = np.zeros(N_BINS)
    counts = np.zeros(N_BINS)

    idx = np.digitize(theta, bins) - 1
    idx = np.clip(idx, 0, N_BINS-1)

    for i, b in enumerate(idx):
        prof[b] += signal[i]
        counts[b] += 1

    prof = prof / np.maximum(counts, 1)
    prof = prof - prof.mean()

    return prof

def harmonic_features(profile):
    coeff = np.fft.rfft(profile)
    power = np.abs(coeff)**2

    # remove DC mode
    power = power[1:MAX_MODE+1]
    total = power.sum() + EPS

    frac = power / total

    low = frac[0:2].sum()       # modes 1-2
    mid = frac[2:6].sum()       # modes 3-6
    high = frac[6:].sum()       # modes 7+

    dom_mode = int(np.argmax(frac) + 1)
    dom_frac = float(np.max(frac))

    return {
        "harmonic_low_fraction": float(low),
        "harmonic_mid_fraction": float(mid),
        "harmonic_high_fraction": float(high),
        "harmonic_entropy": entropy(frac),
        "dominant_harmonic_mode": dom_mode,
        "dominant_harmonic_fraction": dom_frac,
    }

records = []

print("\nProcessing harmonic biological manifold analysis\n")

for sid in sorted(adata.obs["sample_id"].unique()):
    idx = np.where(adata.obs["sample_id"].values == sid)[0]
    obs = adata.obs.iloc[idx]

    if "is_interface" in obs.columns:
        iface = obs["is_interface"].values.astype(bool)
    else:
        tq = obs["tumor_score"].quantile(0.75)
        iq = obs["tcell_score"].quantile(0.75)
        iface = (obs["tumor_score"].values > tq) & (obs["tcell_score"].values > iq)

    iface_idx = idx[iface]
    if len(iface_idx) < MIN_INTERFACE:
        print(f"[{sid}] skipped small interface n={len(iface_idx)}")
        continue

    sub = adata.obs.iloc[iface_idx]
    coords = np.asarray(adata.obsm["spatial"][iface_idx], dtype=float)

    tumor = sub["tumor_score"].values.astype(float)
    exhaustion = sub["exhaustion_score"].values.astype(float)

    edges = build_edges(coords)
    density, cf = coexact_density(tumor, exhaustion, edges, len(coords))

    prof = angular_profile(coords, density)
    hf = harmonic_features(prof)

    rec = {
        "sample_id": sid,
        "response": sub["response"].iloc[0],
        "n_interface": len(coords),
        "coexact_fraction": cf,
        "density_gini": gini(density),
        "density_p95": float(np.quantile(density, 0.95)),
        **hf,
    }

    records.append(rec)

    print(
        f"[{sid:18s}] {rec['response']:14s} "
        f"cf={cf:.4f} low={hf['harmonic_low_fraction']:.3f} "
        f"mid={hf['harmonic_mid_fraction']:.3f} high={hf['harmonic_high_fraction']:.3f}"
    )

out = pd.DataFrame(records)

if kts is not None and "sample_id" in kts.columns:
    keep = [c for c in ["sample_id", "kts_state", "spatial_score", "lcc_fraction"] if c in kts.columns]
    if len(keep) > 1:
        out = out.merge(kts[keep], on="sample_id", how="left")

out.to_csv(OUT, index=False)

metrics = [
    "harmonic_low_fraction",
    "harmonic_mid_fraction",
    "harmonic_high_fraction",
    "harmonic_entropy",
    "dominant_harmonic_mode",
    "dominant_harmonic_fraction",
    "coexact_fraction",
]

print(f"\nSaved → {OUT}")

print("\nGroup medians:")
print(out.groupby("response")[metrics].median(numeric_only=True))

R = out[out["response"] == "Responder"]
NR = out[out["response"] == "Non_Responder"]

print("\nResponder vs NonResponder tests:")
for m in metrics:
    x = R[m].dropna()
    y = NR[m].dropna()
    if len(x) and len(y):
        _, pg = mannwhitneyu(x, y, alternative="greater")
        _, pl = mannwhitneyu(x, y, alternative="less")
        print(f"{m:30s} R={x.median():.4f} NR={y.median():.4f} p(R>NR)={pg:.4f} p(R<NR)={pl:.4f}")

for m in ["harmonic_low_fraction", "harmonic_mid_fraction", "harmonic_high_fraction", "harmonic_entropy"]:
    rho, p = spearmanr(out[m], out["coexact_fraction"], nan_policy="omit")
    print(f"\nSpearman {m} vs coexact_fraction: rho={rho:.3f}, p={p:.4f}")

print("\nInterpretation:")
print("  low harmonics  = global/ring-like organization")
print("  mid harmonics  = finite-range front-like organization")
print("  high harmonics = punctate/noisy fragmentation")
print("  This is group-FFT-inspired harmonic decomposition, not literal group FFT.")
