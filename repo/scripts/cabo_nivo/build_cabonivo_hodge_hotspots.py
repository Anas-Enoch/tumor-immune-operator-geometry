from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors

INFILE = Path("data/GSE238264/cabonivo_interface.h5ad")
OUTFILE = Path("results/cabonivo/cabonivo_hodge_hotspots.csv")

K = 6
RIDGE = 1e-8
MIN_INTERFACE = 20

adata = sc.read_h5ad(INFILE)

records = []

def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.sum(x) <= 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n

def largest_component_fraction(edges, hotspot_mask):
    nodes = set(np.where(hotspot_mask)[0])
    if not nodes:
        return 0.0

    adj = {i: [] for i in nodes}
    for a, b in edges:
        if a in nodes and b in nodes:
            adj[a].append(b)
            adj[b].append(a)

    seen = set()
    largest = 0

    for start in nodes:
        if start in seen:
            continue

        q = deque([start])
        seen.add(start)
        size = 0

        while q:
            u = q.popleft()
            size += 1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        largest = max(largest, size)

    return largest / max(len(nodes), 1)

print("Processing CABO/NIVO samples")

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

    nbrs = NearestNeighbors(
        n_neighbors=min(K + 1, n),
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
        iter_lim=1000,
    )[0]

    omega_exact = B1 @ alpha
    omega_coexact = omega - omega_exact

    total_E = float(np.sum(omega ** 2))
    exact_E = float(np.sum(omega_exact ** 2))
    coexact_E = float(np.sum(omega_coexact ** 2))
    coexact_fraction = coexact_E / (total_E + 1e-12)

    # node-level coexact density
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

    density_gini = float(gini(density))
    density_p95 = float(np.quantile(density, 0.95))
    density_mean = float(np.mean(density))

    hotspot_thr = np.quantile(density, 0.75)
    hotspot_mask = density > hotspot_thr
    lcc = float(largest_component_fraction(edges, hotspot_mask))
    spatial_score = lcc * density_gini

    rec = {
        "sample_id": sid,
        "response": obs["response"].iloc[0],
        "n_interface": n,
        "n_edges": m,
        "total_energy": total_E,
        "exact_energy": exact_E,
        "coexact_energy": coexact_E,
        "coexact_fraction": coexact_fraction,
        "density_mean": density_mean,
        "density_p95": density_p95,
        "density_gini": density_gini,
        "lcc_fraction": lcc,
        "spatial_score": spatial_score,
    }

    records.append(rec)

    print(
        f"[{sid:6s}] {rec['response']:14s} "
        f"iface={n:4d} "
        f"cf={coexact_fraction:.3f} "
        f"gini={density_gini:.3f} "
        f"lcc={lcc:.3f} "
        f"score={spatial_score:.3f} "
        f"p95={density_p95:.4f}"
    )

out = pd.DataFrame(records)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTFILE, index=False)

print("\nSaved:")
print(OUTFILE)

print("\nGroup medians:")
print(
    out.groupby("response")[
        ["coexact_fraction", "density_gini", "lcc_fraction", "spatial_score", "density_p95"]
    ].median()
)
