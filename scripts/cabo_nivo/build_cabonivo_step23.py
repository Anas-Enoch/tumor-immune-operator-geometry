from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import lsqr, eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score

INFILE = Path("data/GSE238264/cabonivo_interface.h5ad")
KTS = Path("results/cabonivo/cabonivo_kts_states.csv")
OUT = Path("results/cabonivo/cabonivo_step23_local_global.csv")

K = 6
MIN_INTERFACE = 20
RIDGE = 1e-8
N_GLOBAL_MODES = 5

adata = sc.read_h5ad(INFILE)
kts = pd.read_csv(KTS)

records = []

def compute_hodge_density(coords, A, B, k=6):
    n = len(A)

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="kd_tree"
    ).fit(coords)

    _, knn = nbrs.kneighbors(coords)

    edges = []
    omega = []

    for i in range(n):
        for j in knn[i, 1:]:
            j = int(j)
            if i == j:
                continue
            edges.append((i, j))
            omega.append(A[i] * B[j] - A[j] * B[i])

    m = len(edges)
    omega = np.asarray(omega, dtype=float)

    rows, cols, vals = [], [], []

    for e, (src, tgt) in enumerate(edges):
        rows.extend([e, e])
        cols.extend([src, tgt])
        vals.extend([-1.0, 1.0])

    B1 = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))
    L0 = B1.T @ B1
    L0 = L0 + RIDGE * sparse.eye(n, format="csr")

    rhs = B1.T @ omega

    alpha = lsqr(
        L0,
        rhs,
        atol=1e-8,
        btol=1e-8,
        iter_lim=1000
    )[0]

    omega_exact = B1 @ alpha
    omega_coexact = omega - omega_exact

    density_sum = np.zeros(n)
    degree = np.zeros(n)

    abs_c = np.abs(omega_coexact)

    srcs = np.array([a for a, b in edges], dtype=int)
    tgts = np.array([b for a, b in edges], dtype=int)

    np.add.at(density_sum, srcs, abs_c)
    np.add.at(density_sum, tgts, abs_c)
    np.add.at(degree, srcs, 1)
    np.add.at(degree, tgts, 1)

    density = density_sum / np.maximum(degree, 1)

    return density, edges

def local_prediction_r2(density, edges, n):
    adj = [[] for _ in range(n)]

    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)

    y_true = []
    y_pred = []

    for i in range(n):
        nb = adj[i]
        if len(nb) == 0:
            continue

        y_true.append(density[i])
        y_pred.append(np.mean(density[nb]))

    if len(y_true) < 5:
        return np.nan

    return float(r2_score(y_true, y_pred))

def global_spectral_r2(density, edges, n, n_modes=5):
    rows = []
    cols = []
    vals = []

    for a, b in edges:
        rows.extend([a, b])
        cols.extend([b, a])
        vals.extend([1.0, 1.0])

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A.data[:] = 1.0
    A.eliminate_zeros()

    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sparse.diags(deg) - A

    k = min(n_modes + 1, n - 2)

    if k < 2:
        return np.nan, np.nan

    try:
        eigvals, eigvecs = eigsh(L, k=k, which="SM")
    except Exception:
        return np.nan, np.nan

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # drop constant mode
    Phi = eigvecs[:, 1:min(n_modes + 1, eigvecs.shape[1])]

    if Phi.shape[1] == 0:
        return np.nan, np.nan

    y = density.astype(float)
    y_centered = y - np.mean(y)

    coef = Phi.T @ y_centered
    recon = Phi @ coef + np.mean(y)

    r2 = float(r2_score(y, recon))

    total_power = float(np.sum(y_centered ** 2))
    low_power = float(np.sum(coef ** 2))
    low_freq_frac = low_power / (total_power + 1e-12)

    return r2, low_freq_frac

print("Processing CABO/NIVO Step23 local/global predictability")

for sid in sorted(adata.obs["sample_id"].unique()):
    idx_sample = np.where(adata.obs["sample_id"].values == sid)[0]
    obs = adata.obs.iloc[idx_sample].copy()

    iface_local = obs["is_interface"].values.astype(bool)
    iface_idx = idx_sample[iface_local]

    n = len(iface_idx)

    if n < MIN_INTERFACE:
        print(f"[{sid}] SMALL INTERFACE n={n}")
        continue

    coords = adata.obsm["spatial"][iface_idx]

    A = adata.obs.iloc[iface_idx]["tumor_score"].values.astype(float)
    B = adata.obs.iloc[iface_idx]["exhaustion_score"].values.astype(float)

    density, edges = compute_hodge_density(coords, A, B, k=K)

    local_r2 = local_prediction_r2(density, edges, n)
    global_r2, low_freq_frac = global_spectral_r2(
        density,
        edges,
        n,
        n_modes=N_GLOBAL_MODES
    )

    rec = {
        "sample_id": sid,
        "response": obs["response"].iloc[0],
        "n_interface": n,
        "local_r2": local_r2,
        "global_r2": global_r2,
        "low_freq_frac": low_freq_frac,
        "local_minus_global": local_r2 - global_r2,
        "density_mean": float(np.mean(density)),
        "density_p95": float(np.quantile(density, 0.95)),
    }

    records.append(rec)

    print(
        f"[{sid:6s}] {rec['response']:14s} "
        f"n={n:4d} "
        f"localR2={local_r2:.3f} "
        f"globalR2={global_r2:.3f} "
        f"lowfreq={low_freq_frac:.3f}"
    )

out = pd.DataFrame(records)

out = out.merge(
    kts[["sample_id", "kts_state", "coexact_fraction", "spatial_score"]],
    on="sample_id",
    how="left",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)

print("\nStep23 table:")
print(
    out[
        [
            "sample_id",
            "response",
            "kts_state",
            "local_r2",
            "global_r2",
            "low_freq_frac",
            "local_minus_global",
        ]
    ].to_string(index=False)
)

print("\nGroup medians:")
print(
    out.groupby("response")[
        ["local_r2", "global_r2", "low_freq_frac", "local_minus_global"]
    ].median()
)

print("\nKTS-state medians:")
print(
    out.groupby("kts_state")[
        ["local_r2", "global_r2", "low_freq_frac", "local_minus_global"]
    ].median()
)

print("\nExpected interpretation:")
print("  local_r2 high  -> finite-range local constraint")
print("  global_r2 poor -> not globally periodic/compressible")
print("  S3>S1 local_r2 -> KTS states encode local organization")
