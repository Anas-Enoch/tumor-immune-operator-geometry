#!/usr/bin/env python3

"""
Graph-KS instability proxy for Spatial Hallmarks pan-cancer Visium cohort.

Input:
    results_spatial_hallmarks/spatial_hallmarks_hodge_node_scores.csv.gz
    results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv

Output:
    results_spatial_hallmarks/spatial_hallmarks_ks_instability.csv
    results_spatial_hallmarks/spatial_hallmarks_ks_instability_summary.csv

Purpose:
    Apply the original core graph-KS-style instability proxy to the large
    external pan-cancer spatial cohort.

Important:
    This does NOT claim that tumor biology solves the Kuramoto–Sivashinsky PDE.
    It computes a graph operator proxy on the coexact-density field:

        KS_proxy(u) = -L u - alpha L^2 u

    where:
        u = node coexact-density field
        L = graph Laplacian on spatial kNN graph

Interpretation:
    High interface/tumor-core KS ratio means the interface coexact field is
    more instability-like / nonlinear operator-active than tumor core.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu, binomtest, kruskal
from sklearn.neighbors import NearestNeighbors


NODES = Path("results_spatial_hallmarks/spatial_hallmarks_hodge_node_scores.csv.gz")
HODGE = Path("results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv")

OUT = Path("results_spatial_hallmarks/spatial_hallmarks_ks_instability.csv")
SUMMARY = Path("results_spatial_hallmarks/spatial_hallmarks_ks_instability_summary.csv")

K = 6
ALPHA = 0.25
EPS = 1e-12
MIN_INTERFACE = 20
MIN_TUMOR_CORE = 20


def build_weighted_graph(coords, k=6):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="kd_tree",
    ).fit(coords)

    dists, knn = nbrs.kneighbors(coords)

    sigma = np.nanmedian(dists[:, 1:]) + EPS

    rows, cols, vals = [], [], []

    for i in range(n):
        for j, d in zip(knn[i, 1:], dists[i, 1:]):
            j = int(j)
            if i == j:
                continue

            w = np.exp(-(d ** 2) / (2 * sigma ** 2))

            rows.extend([i, j])
            cols.extend([j, i])
            vals.extend([w, w])

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W.sum_duplicates()

    deg = np.asarray(W.sum(axis=1)).ravel()
    L = sparse.diags(deg) - W

    return L, W


def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < EPS:
        return np.zeros_like(x)
    return (x - mu) / sd


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


def ks_proxy(u, L, alpha=0.25):
    """
    Graph KS-inspired operator:
        q = -L u - alpha L^2 u

    We use |q| as local instability-like activity.
    """
    u = np.asarray(u, dtype=float)
    Lu = L @ u
    L2u = L @ Lu
    q = -Lu - alpha * L2u
    return np.asarray(q).ravel()


def main():
    if not NODES.exists():
        raise FileNotFoundError(NODES)

    nodes = pd.read_csv(NODES)
    hodge = pd.read_csv(HODGE) if HODGE.exists() else None

    required = [
        "sample_id",
        "cancer_type",
        "x",
        "y",
        "is_interface",
        "is_tumor_core",
        "node_coexact_density",
    ]

    for c in required:
        if c not in nodes.columns:
            raise ValueError(f"Missing required node column: {c}")

    records = []

    print("\nRunning Spatial Hallmarks graph-KS instability proxy\n")

    for sid, g in nodes.groupby("sample_id", sort=True):
        g = g.copy()

        interface = g["is_interface"].astype(bool).values
        tumor_core = g["is_tumor_core"].astype(bool).values

        n_interface = int(interface.sum())
        n_tumor = int(tumor_core.sum())

        if n_interface < MIN_INTERFACE or n_tumor < MIN_TUMOR_CORE:
            print(f"[{sid}] skipped low n interface={n_interface} tumor={n_tumor}")
            continue

        coords = g[["x", "y"]].values.astype(float)
        u = g["node_coexact_density"].values.astype(float)

        # Standardize density within section so KS is not merely scale.
        u_z = zscore(u)

        L, W = build_weighted_graph(coords, k=K)

        q = ks_proxy(u_z, L, alpha=ALPHA)
        activity = np.abs(q)

        interface_mean = float(np.nanmean(activity[interface]))
        tumor_mean = float(np.nanmean(activity[tumor_core]))

        ratio = interface_mean / (tumor_mean + EPS)
        log2_ratio = float(np.log2(ratio + EPS))

        p = safe_mwu_greater(activity[interface], activity[tumor_core])

        rec = {
            "sample_id": sid,
            "cancer_type": g["cancer_type"].iloc[0],
            "n_nodes": len(g),
            "n_interface": n_interface,
            "n_tumor_core": n_tumor,
            "ks_alpha": ALPHA,

            "ks_interface_mean": interface_mean,
            "ks_tumor_core_mean": tumor_mean,
            "ks_interface_vs_tumor_ratio": ratio,
            "ks_log2_interface_vs_tumor_ratio": log2_ratio,
            "ks_mwu_p_interface_gt_tumor": p,

            "ks_interface_p95": float(np.nanquantile(activity[interface], 0.95)),
            "ks_tumor_core_p95": float(np.nanquantile(activity[tumor_core], 0.95)),
            "ks_global_mean": float(np.nanmean(activity)),
            "ks_global_p95": float(np.nanquantile(activity, 0.95)),
        }

        records.append(rec)

        print(
            f"[{sid:14s}] {rec['cancer_type']:12s} "
            f"KS_R={ratio:.3f} p={p:.2e}"
        )

    out = pd.DataFrame(records)

    if hodge is not None:
        keep = [
            c for c in [
                "sample_id",
                "interface_vs_tumor_enrichment",
                "coexact_fraction",
                "spatial_score",
            ]
            if c in hodge.columns
        ]
        if len(keep) > 1:
            out = out.merge(hodge[keep].drop_duplicates("sample_id"), on="sample_id", how="left")

    out.to_csv(OUT, index=False)

    rows = []

    if len(out):
        n = len(out)
        n_gt = int((out["ks_interface_vs_tumor_ratio"] > 1).sum())
        n_sig = int((out["ks_mwu_p_interface_gt_tumor"] < 0.05).sum())

        try:
            p_sign = binomtest(n_gt, n, p=0.5, alternative="greater").pvalue
        except Exception:
            p_sign = np.nan

        rows.append({
            "level": "cohort",
            "group": "all",
            "n": n,
            "n_ratio_gt_1": n_gt,
            "n_sig_p_lt_0_05": n_sig,
            "median_ratio": float(out["ks_interface_vs_tumor_ratio"].median()),
            "median_log2_ratio": float(out["ks_log2_interface_vs_tumor_ratio"].median()),
            "sign_test_p_ratio_gt_1": float(p_sign),
        })

        for cancer, g in out.groupby("cancer_type"):
            n = len(g)
            n_gt = int((g["ks_interface_vs_tumor_ratio"] > 1).sum())
            n_sig = int((g["ks_mwu_p_interface_gt_tumor"] < 0.05).sum())

            try:
                p_sign = binomtest(n_gt, n, p=0.5, alternative="greater").pvalue
            except Exception:
                p_sign = np.nan

            rows.append({
                "level": "cancer_type",
                "group": cancer,
                "n": n,
                "n_ratio_gt_1": n_gt,
                "n_sig_p_lt_0_05": n_sig,
                "median_ratio": float(g["ks_interface_vs_tumor_ratio"].median()),
                "median_log2_ratio": float(g["ks_log2_interface_vs_tumor_ratio"].median()),
                "sign_test_p_ratio_gt_1": float(p_sign),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY, index=False)

    print(f"\nSaved → {OUT}")
    print(f"Saved → {SUMMARY}")

    if len(out):
        print("\nCohort KS summary:")
        print(summary.to_string(index=False))

        print("\nBy cancer type median KS ratios:")
        print(out.groupby("cancer_type")["ks_interface_vs_tumor_ratio"].median())

        print("\nKruskal across cancer type:")
        groups = [g["ks_interface_vs_tumor_ratio"].dropna().values for _, g in out.groupby("cancer_type")]
        groups = [x for x in groups if len(x) > 0]
        if len(groups) >= 2:
            stat, p = kruskal(*groups)
            print(f"KS ratio: H={stat:.3f}, p={p:.4f}")

    print("\nInterpretation:")
    print("  KS_R > 1 means interface coexact-density field has higher graph-KS activity than tumor core.")
    print("  This is an instability-like operator proxy, not a claim that the tissue solves the KS PDE.")
    print("  Use as external pan-cancer validation only if direction is consistent and not redundant.")


if __name__ == "__main__":
    main()
