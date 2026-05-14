#!/usr/bin/env python3

"""
Build tumor–immune interface masks for the Spatial Hallmarks pan-cancer Visium cohort.

Input:
    results_spatial_hallmarks/spatial_hallmarks_scored.h5ad

Output:
    results_spatial_hallmarks/spatial_hallmarks_interface.h5ad
    results_spatial_hallmarks/spatial_hallmarks_interface_summary.csv

Definition:
    tumor_high  = tumor_score >= sample-wise tumor quantile
    immune_high = immune_score >= sample-wise immune quantile

Interface spots are tumor-high spots with immune-high neighbors
or immune-high spots with tumor-high neighbors within a spatial kNN graph.

This is deliberately close to the original HCC interface logic while being
cross-cancer and response-label independent.
"""

from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


INFILE = Path("results_spatial_hallmarks/spatial_hallmarks_scored.h5ad")
OUT_H5AD = Path("results_spatial_hallmarks/spatial_hallmarks_interface.h5ad")
OUT_SUMMARY = Path("results_spatial_hallmarks/spatial_hallmarks_interface_summary.csv")

K = 6
TUMOR_Q = 0.75
IMMUNE_Q = 0.75
MIN_INTERFACE = 20
MIN_TUMOR = 20
MIN_IMMUNE = 20


def build_knn(coords, k=6):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="kd_tree",
    ).fit(coords)

    _, knn = nbrs.kneighbors(coords)

    neigh = [[] for _ in range(n)]

    for i in range(n):
        for j in knn[i, 1:]:
            j = int(j)
            if i == j:
                continue
            neigh[i].append(j)
            neigh[j].append(i)

    # unique neighbors
    neigh = [sorted(set(x)) for x in neigh]
    return neigh


def connected_components(mask, neigh):
    nodes = set(np.where(mask)[0])
    if not nodes:
        return 0, 0, 0.0

    seen = set()
    sizes = []

    for start in nodes:
        if start in seen:
            continue

        q = deque([start])
        seen.add(start)
        size = 0

        while q:
            u = q.popleft()
            size += 1

            for v in neigh[u]:
                if v in nodes and v not in seen:
                    seen.add(v)
                    q.append(v)

        sizes.append(size)

    largest = max(sizes) if sizes else 0
    return len(sizes), largest, largest / max(len(nodes), 1)


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
    ]

    for c in required:
        if c not in ad.obs.columns:
            raise ValueError(f"Missing required obs column: {c}")

    if "spatial" not in ad.obsm:
        raise ValueError("Missing ad.obsm['spatial']")

    ad.obs["tumor_high"] = False
    ad.obs["immune_high"] = False
    ad.obs["is_interface"] = False
    ad.obs["is_tumor_core"] = False
    ad.obs["is_immune_region"] = False
    ad.obs["interface_region"] = "other"

    summary = []

    print("\nBuilding Spatial Hallmarks tumor–immune interfaces\n")

    for sid in sorted(ad.obs["sample_id"].unique()):
        idx = np.where(ad.obs["sample_id"].values == sid)[0]
        obs = ad.obs.iloc[idx]
        coords = np.asarray(ad.obsm["spatial"][idx], dtype=float)

        tumor = obs["tumor_score"].values.astype(float)
        immune = obs["immune_score"].values.astype(float)

        tq = np.quantile(tumor, TUMOR_Q)
        iq = np.quantile(immune, IMMUNE_Q)

        tumor_high = tumor >= tq
        immune_high = immune >= iq

        neigh = build_knn(coords, k=K)

        interface = np.zeros(len(idx), dtype=bool)

        for i in range(len(idx)):
            nb = neigh[i]
            if not nb:
                continue

            # tumor spot adjacent to immune-high neighborhood
            if tumor_high[i] and any(immune_high[j] for j in nb):
                interface[i] = True

            # immune spot adjacent to tumor-high neighborhood
            if immune_high[i] and any(tumor_high[j] for j in nb):
                interface[i] = True

        # Tumor core = tumor-high but not interface.
        tumor_core = tumor_high & (~interface)

        # Immune region = immune-high but not interface.
        immune_region = immune_high & (~interface)

        region = np.array(["other"] * len(idx), dtype=object)
        region[tumor_core] = "tumor_core"
        region[immune_region] = "immune"
        region[interface] = "interface"

        n_components, largest, lcc = connected_components(interface, neigh)

        valid = (
            interface.sum() >= MIN_INTERFACE
            and tumor_core.sum() >= MIN_TUMOR
            and immune_region.sum() >= MIN_IMMUNE
        )

        ad.obs.iloc[idx, ad.obs.columns.get_loc("tumor_high")] = tumor_high
        ad.obs.iloc[idx, ad.obs.columns.get_loc("immune_high")] = immune_high
        ad.obs.iloc[idx, ad.obs.columns.get_loc("is_interface")] = interface
        ad.obs.iloc[idx, ad.obs.columns.get_loc("is_tumor_core")] = tumor_core
        ad.obs.iloc[idx, ad.obs.columns.get_loc("is_immune_region")] = immune_region
        ad.obs.iloc[idx, ad.obs.columns.get_loc("interface_region")] = region

        rec = {
            "sample_id": sid,
            "cancer_type": obs["cancer_type"].iloc[0],
            "n_spots": len(idx),
            "tumor_q": TUMOR_Q,
            "immune_q": IMMUNE_Q,
            "tumor_threshold": tq,
            "immune_threshold": iq,
            "n_tumor_high": int(tumor_high.sum()),
            "n_immune_high": int(immune_high.sum()),
            "n_interface": int(interface.sum()),
            "n_tumor_core": int(tumor_core.sum()),
            "n_immune_region": int(immune_region.sum()),
            "interface_fraction": float(interface.mean()),
            "interface_components": int(n_components),
            "interface_largest_component": int(largest),
            "interface_lcc_fraction": float(lcc),
            "valid_for_hodge": bool(valid),
        }

        summary.append(rec)

        status = "VALID" if valid else "LOW_N"
        print(
            f"[{sid:14s}] {rec['cancer_type']:12s} "
            f"spots={len(idx):5d} "
            f"interface={rec['n_interface']:4d} "
            f"tumor_core={rec['n_tumor_core']:4d} "
            f"immune={rec['n_immune_region']:4d} "
            f"LCC={lcc:.3f} "
            f"{status}"
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT_SUMMARY, index=False)
    ad.write_h5ad(OUT_H5AD)

    print("\nSaved:")
    print(f"  {OUT_H5AD}")
    print(f"  {OUT_SUMMARY}")

    print("\nValidity by cancer type:")
    print(summary.groupby("cancer_type")["valid_for_hodge"].agg(["sum", "count"]))

    print("\nInterface counts by cancer type:")
    print(summary.groupby("cancer_type")["n_interface"].describe())


if __name__ == "__main__":
    main()
