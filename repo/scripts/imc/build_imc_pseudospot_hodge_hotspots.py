from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr
from sklearn.neighbors import NearestNeighbors
from collections import deque

INFILE = Path("results/imc/ici_pseudospots.csv")
OUT_SUMMARY = Path("results/imc/ici_pseudospot_hodge_hotspots.csv")

K = 6
RIDGE = 1e-8
MIN_INTERFACE = 20

df = pd.read_csv(INFILE)
samples = sorted(df["sample_id"].unique())

records = []

def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.sum(x) <= 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1)) * x) / (n * np.sum(x))) - (n + 1) / n

def largest_component_fraction(n, edges, hotspot_mask):
    hotspot_nodes = set(np.where(hotspot_mask)[0])
    if not hotspot_nodes:
        return 0.0

    adj = {i: [] for i in hotspot_nodes}
    for a, b in edges:
        if a in hotspot_nodes and b in hotspot_nodes:
            adj[a].append(b)
            adj[b].append(a)

    seen = set()
    largest = 0

    for start in hotspot_nodes:
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

    return largest / max(len(hotspot_nodes), 1)

print(f"Processing {len(samples)} ROIs")

for sid in samples:
    sub = df[df["sample_id"] == sid].copy()
    iface = sub[sub["is_interface"]].copy()

    n = len(iface)

    if n < MIN_INTERFACE:
        print(f"[{sid}] SMALL INTERFACE n={n}")
        continue

    xy = iface[["x", "y"]].values
    A = iface["tumor_score"].values
    B = iface["exhaustion_score"].values

    nbrs = NearestNeighbors(
        n_neighbors=min(K + 1, n),
        algorithm="kd_tree"
    ).fit(xy)

    _, idx = nbrs.kneighbors(xy)

    edges = []
    omega = []

    for i in range(n):
        for j in idx[i, 1:]:
            if i == j:
                continue
            edges.append((i, int(j)))
            omega.append(A[i] * B[j] - A[j] * B[i])

    m = len(edges)
    omega = np.asarray(omega, dtype=float)

    rows = []
    cols = []
    vals = []

    for e, (src, tgt) in enumerate(edges):
        rows.extend([e, e])
        cols.extend([src, tgt])
        vals.extend([-1.0, 1.0])

    B1 = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))
    L0 = B1.T @ B1
    L0 = L0 + RIDGE * sparse.eye(n, format="csr")

    rhs = B1.T @ omega
    alpha = lsqr(L0, rhs, atol=1e-8, btol=1e-8, iter_lim=500)[0]

    omega_exact = B1 @ alpha
    omega_coexact = omega - omega_exact

    total_E = float(np.sum(omega ** 2))
    exact_E = float(np.sum(omega_exact ** 2))
    coexact_E = float(np.sum(omega_coexact ** 2))
    coexact_fraction = coexact_E / (total_E + 1e-12)

    # Node coexact density
    density_sum = np.zeros(n)
    degree = np.zeros(n)

    abs_c = np.abs(omega_coexact)

    srcs = np.array([e[0] for e in edges], dtype=int)
    tgts = np.array([e[1] for e in edges], dtype=int)

    np.add.at(density_sum, srcs, abs_c)
    np.add.at(density_sum, tgts, abs_c)
    np.add.at(degree, srcs, 1)
    np.add.at(degree, tgts, 1)

    density = density_sum / np.maximum(degree, 1)

    p95 = float(np.quantile(density, 0.95))
    g = float(gini(density))

    hotspot_thr = np.quantile(density, 0.75)
    hotspot_mask = density > hotspot_thr
    lcc = float(largest_component_fraction(n, edges, hotspot_mask))

    spatial_score = lcc * g

    rec = {
        "sample_id": sid,
        "patient_id": iface["patient_id"].iloc[0],
        "roi_id": iface["roi_id"].iloc[0],
        "n_interface": n,
        "n_edges": m,
        "total_energy": total_E,
        "exact_energy": exact_E,
        "coexact_energy": coexact_E,
        "coexact_fraction": coexact_fraction,
        "density_mean": float(np.mean(density)),
        "density_p95": p95,
        "density_gini": g,
        "lcc_fraction": lcc,
        "spatial_score": spatial_score,
    }

    records.append(rec)

    print(
        f"[{sid}] iface={n:5d} "
        f"cf={coexact_fraction:.3f} "
        f"gini={g:.3f} "
        f"lcc={lcc:.3f} "
        f"score={spatial_score:.3f}"
    )

out = pd.DataFrame(records)
OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT_SUMMARY, index=False)

print("\nSaved:", OUT_SUMMARY)
print(out.head())
