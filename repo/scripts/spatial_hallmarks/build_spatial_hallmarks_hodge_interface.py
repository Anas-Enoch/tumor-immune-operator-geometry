#!/usr/bin/env python3

"""
Run the original core Hodge interface layer on the Spatial Hallmarks pan-cancer Visium cohort.

Input:
    results_spatial_hallmarks/spatial_hallmarks_interface.h5ad

Output:
    results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv
    results_spatial_hallmarks/spatial_hallmarks_hodge_node_scores.csv.gz

Goal:
    Test whether tumor–immune interfaces in a large external pan-cancer Visium cohort
    show enrichment of coexact non-gradient interaction intensity relative to tumor core.

This reproduces the original core HCC logic:
    tumor / immune programs
        ↓
    antisymmetric wedge field
        ↓
    graph Hodge decomposition
        ↓
    node-level coexact density
        ↓
    interface vs tumor-core enrichment
"""

from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import scanpy as sc

from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import mannwhitneyu, binomtest
from sklearn.neighbors import NearestNeighbors


INFILE = Path("results_spatial_hallmarks/spatial_hallmarks_interface.h5ad")

OUT_SUMMARY = Path("results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv")
OUT_NODES = Path("results_spatial_hallmarks/spatial_hallmarks_hodge_node_scores.csv.gz")

K = 6
RIDGE = 1e-8
EPS = 1e-12

MIN_INTERFACE = 20
MIN_TUMOR_CORE = 20


def build_knn_edges(coords, k=6):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="kd_tree",
    ).fit(coords)

    _, knn = nbrs.kneighbors(coords)

    directed = []
    undirected = set()

    for i in range(n):
        for j in knn[i, 1:]:
            j = int(j)
            if i == j:
                continue

            directed.append((i, j))
            a, b = sorted((i, j))
            undirected.add((a, b))

    return directed, sorted(undirected)


def largest_component_fraction(mask, undirected_edges, n):
    nodes = set(np.where(mask)[0])

    if not nodes:
        return 0.0

    adj = {i: [] for i in nodes}

    for a, b in undirected_edges:
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


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0 or np.sum(x) <= 0:
        return np.nan

    x = np.sort(x)
    n = len(x)

    return float(
        (2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x)))
        - (n + 1) / n
    )


def hodge_decompose_node_density(A, B, directed_edges, n):
    """
    Compute antisymmetric wedge field omega_ij = A_i B_j - A_j B_i.
    Project onto exact gradients, retain residual coexact-like component.

    This is the graph-level lower-Hodge construction used for core validation.
    """
    m = len(directed_edges)

    omega = np.zeros(m, dtype=float)

    for e, (src, tgt) in enumerate(directed_edges):
        omega[e] = A[src] * B[tgt] - A[tgt] * B[src]

    rows, cols, vals = [], [], []

    for e, (src, tgt) in enumerate(directed_edges):
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

    total_energy = float(np.sum(omega ** 2))
    exact_energy = float(np.sum(omega_exact ** 2))
    coexact_energy = float(np.sum(omega_coexact ** 2))

    coexact_fraction = coexact_energy / (total_energy + EPS)
    exact_fraction = exact_energy / (total_energy + EPS)

    density_sum = np.zeros(n)
    degree = np.zeros(n)

    abs_c = np.abs(omega_coexact)

    for val, (src, tgt) in zip(abs_c, directed_edges):
        density_sum[src] += val
        density_sum[tgt] += val
        degree[src] += 1
        degree[tgt] += 1

    node_density = density_sum / np.maximum(degree, 1)

    return {
        "omega": omega,
        "omega_exact": omega_exact,
        "omega_coexact": omega_coexact,
        "node_coexact_density": node_density,
        "total_energy": total_energy,
        "exact_energy": exact_energy,
        "coexact_energy": coexact_energy,
        "coexact_fraction": coexact_fraction,
        "exact_fraction": exact_fraction,
    }


def safe_mwu_greater(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    try:
        return float(mannwhitneyu(x, y, alternative="greater").pvalue)
    except Exception:
        return np.nan


def main():
    ad = sc.read_h5ad(INFILE)

    required = [
        "sample_id",
        "cancer_type",
        "tumor_score",
        "immune_score",
        "tcell_score",
        "exhaustion_score",
        "myeloid_score",
        "is_interface",
        "is_tumor_core",
        "interface_region",
    ]

    for c in required:
        if c not in ad.obs.columns:
            raise ValueError(f"Missing required obs column: {c}")

    if "spatial" not in ad.obsm:
        raise ValueError("Missing ad.obsm['spatial']")

    summary_records = []
    node_records = []

    print("\nRunning Spatial Hallmarks Hodge interface validation\n")

    for sid in sorted(ad.obs["sample_id"].unique()):
        idx = np.where(ad.obs["sample_id"].values == sid)[0]
        obs = ad.obs.iloc[idx].copy()
        coords = np.asarray(ad.obsm["spatial"][idx], dtype=float)

        interface_mask = obs["is_interface"].astype(bool).values
        tumor_core_mask = obs["is_tumor_core"].astype(bool).values

        n_interface = int(interface_mask.sum())
        n_tumor_core = int(tumor_core_mask.sum())

        if n_interface < MIN_INTERFACE or n_tumor_core < MIN_TUMOR_CORE:
            print(f"[{sid}] skipped: interface={n_interface}, tumor_core={n_tumor_core}")
            continue

        A = obs["tumor_score"].values.astype(float)
        B = obs["immune_score"].values.astype(float)

        directed_edges, undirected_edges = build_knn_edges(coords, k=K)

        res = hodge_decompose_node_density(
            A=A,
            B=B,
            directed_edges=directed_edges,
            n=len(idx),
        )

        density = res["node_coexact_density"]

        interface_density = density[interface_mask]
        tumor_density = density[tumor_core_mask]

        interface_mean = float(np.nanmean(interface_density))
        tumor_mean = float(np.nanmean(tumor_density))

        enrichment_ratio = interface_mean / (tumor_mean + EPS)
        log2_enrichment = float(np.log2(enrichment_ratio + EPS))

        p_mwu = safe_mwu_greater(interface_density, tumor_density)

        interface_lcc = largest_component_fraction(interface_mask, undirected_edges, len(idx))

        rec = {
            "sample_id": sid,
            "cancer_type": obs["cancer_type"].iloc[0],
            "n_spots": len(idx),
            "n_edges": len(directed_edges),
            "n_interface": n_interface,
            "n_tumor_core": n_tumor_core,

            "total_energy": res["total_energy"],
            "exact_energy": res["exact_energy"],
            "coexact_energy": res["coexact_energy"],
            "exact_fraction": res["exact_fraction"],
            "coexact_fraction": res["coexact_fraction"],

            "interface_coexact_density_mean": interface_mean,
            "tumor_core_coexact_density_mean": tumor_mean,
            "interface_vs_tumor_enrichment": enrichment_ratio,
            "log2_interface_vs_tumor_enrichment": log2_enrichment,
            "mwu_p_interface_gt_tumor": p_mwu,

            "interface_density_p95": float(np.nanquantile(interface_density, 0.95)),
            "tumor_core_density_p95": float(np.nanquantile(tumor_density, 0.95)),
            "density_gini_all": gini(density),
            "density_gini_interface": gini(interface_density),

            "interface_lcc_fraction": interface_lcc,
            "spatial_score": float(interface_lcc * gini(interface_density)),
            "valid": True,
        }

        summary_records.append(rec)

        for local_i, global_i in enumerate(idx):
            node_records.append({
                "sample_id": sid,
                "cancer_type": obs["cancer_type"].iloc[0],
                "barcode": ad.obs_names[global_i],
                "x": coords[local_i, 0],
                "y": coords[local_i, 1],
                "interface_region": obs["interface_region"].iloc[local_i],
                "is_interface": bool(interface_mask[local_i]),
                "is_tumor_core": bool(tumor_core_mask[local_i]),
                "tumor_score": A[local_i],
                "immune_score": B[local_i],
                "node_coexact_density": density[local_i],
            })

        verdict = "R>1" if enrichment_ratio > 1 else "R<=1"

        print(
            f"[{sid:14s}] {rec['cancer_type']:12s} "
            f"R={enrichment_ratio:.3f} "
            f"cf={res['coexact_fraction']:.3f} "
            f"p={p_mwu:.2e} "
            f"{verdict}"
        )

    summary = pd.DataFrame(summary_records)
    nodes = pd.DataFrame(node_records)

    summary.to_csv(OUT_SUMMARY, index=False)
    nodes.to_csv(OUT_NODES, index=False, compression="gzip")

    print("\nSaved:")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_NODES}")

    if len(summary):
        n_gt = int((summary["interface_vs_tumor_enrichment"] > 1).sum())
        n = len(summary)

        try:
            p_sign = binomtest(n_gt, n, p=0.5, alternative="greater").pvalue
        except Exception:
            p_sign = np.nan

        print("\nCohort summary:")
        print(f"  R>1: {n_gt}/{n}")
        print(f"  sign test p: {p_sign:.3e}")
        print(f"  median R: {summary['interface_vs_tumor_enrichment'].median():.3f}")
        print(f"  median coexact fraction: {summary['coexact_fraction'].median():.3f}")

        print("\nBy cancer type:")
        by = summary.groupby("cancer_type").agg(
            n=("sample_id", "count"),
            n_R_gt_1=("interface_vs_tumor_enrichment", lambda x: int((x > 1).sum())),
            median_R=("interface_vs_tumor_enrichment", "median"),
            median_cf=("coexact_fraction", "median"),
            median_spatial_score=("spatial_score", "median"),
        )
        print(by)

        print("\nSignificant interface enrichment p<0.05:")
        print(int((summary["mwu_p_interface_gt_tumor"] < 0.05).sum()), "/", len(summary))


if __name__ == "__main__":
    main()
