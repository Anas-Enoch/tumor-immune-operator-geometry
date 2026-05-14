#!/usr/bin/env python3

"""
Build NCG-style commutator summaries for the Spatial Hallmarks pan-cancer Visium cohort.

Input:
    results_spatial_hallmarks/spatial_hallmarks_interface.h5ad

Output:
    results_spatial_hallmarks/spatial_hallmarks_ncg_commutators.csv

Goal:
    Test whether the original HCC NCG layer generalizes:
        - tumour–myeloid backbone
        - immune-sector algebra: IE + IM + EM
        - operator entropy / interaction diversity

Channels:
    TI = tumor × T-cell
    TE = tumor × exhaustion
    TM = tumor × myeloid
    IE = T-cell × exhaustion
    IM = T-cell × myeloid
    EM = exhaustion × myeloid
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import binomtest
from sklearn.neighbors import NearestNeighbors


INFILE = Path("results_spatial_hallmarks/spatial_hallmarks_interface.h5ad")
OUT = Path("results_spatial_hallmarks/spatial_hallmarks_ncg_commutators.csv")

K = 6
MIN_INTERFACE = 20
EPS = 1e-12

CHANNELS = {
    "TI": ("tumor_score", "tcell_score"),
    "TE": ("tumor_score", "exhaustion_score"),
    "TM": ("tumor_score", "myeloid_score"),
    "IE": ("tcell_score", "exhaustion_score"),
    "IM": ("tcell_score", "myeloid_score"),
    "EM": ("exhaustion_score", "myeloid_score"),
}


def build_knn_edges(coords, k=6):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="kd_tree",
    ).fit(coords)

    _, knn = nbrs.kneighbors(coords)

    edges = []

    for i in range(n):
        for j in knn[i, 1:]:
            j = int(j)
            if i != j:
                edges.append((i, j))

    return edges


def commutator_energy(A, B, edges):
    vals = np.zeros(len(edges), dtype=float)

    for e, (i, j) in enumerate(edges):
        vals[e] = A[i] * B[j] - A[j] * B[i]

    return float(np.sqrt(np.sum(vals ** 2)))


def entropy_from_positive(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, 0.0)

    s = x.sum()
    if s <= 0:
        return np.nan

    p = x / s
    return float(-np.sum(p * np.log(p + EPS)))


def main():
    ad = sc.read_h5ad(INFILE)

    required = [
        "sample_id",
        "cancer_type",
        "is_interface",
        "tumor_score",
        "tcell_score",
        "exhaustion_score",
        "myeloid_score",
    ]

    for c in required:
        if c not in ad.obs.columns:
            raise ValueError(f"Missing required obs column: {c}")

    if "spatial" not in ad.obsm:
        raise ValueError("Missing ad.obsm['spatial']")

    records = []

    print("\nRunning Spatial Hallmarks NCG commutator layer\n")

    for sid in sorted(ad.obs["sample_id"].unique()):
        idx_all = np.where(ad.obs["sample_id"].values == sid)[0]
        obs_all = ad.obs.iloc[idx_all]

        iface_mask = obs_all["is_interface"].astype(bool).values
        idx = idx_all[iface_mask]

        if len(idx) < MIN_INTERFACE:
            print(f"[{sid}] skipped: interface={len(idx)}")
            continue

        obs = ad.obs.iloc[idx].copy()
        coords = np.asarray(ad.obsm["spatial"][idx], dtype=float)

        edges = build_knn_edges(coords, k=K)

        programs = {
            "tumor_score": obs["tumor_score"].values.astype(float),
            "tcell_score": obs["tcell_score"].values.astype(float),
            "exhaustion_score": obs["exhaustion_score"].values.astype(float),
            "myeloid_score": obs["myeloid_score"].values.astype(float),
        }

        comm = {}

        for name, (a, b) in CHANNELS.items():
            comm[name] = commutator_energy(
                programs[a],
                programs[b],
                edges,
            )

        total = sum(comm.values()) + EPS

        frac = {k: v / total for k, v in comm.items()}

        immune_sector = comm["IE"] + comm["IM"] + comm["EM"]
        tumor_sector = comm["TI"] + comm["TE"] + comm["TM"]

        immune_sector_frac = immune_sector / total
        tumor_sector_frac = tumor_sector / total

        tm_dominance = comm["TM"] / total
        immune_to_tm_ratio = immune_sector / (comm["TM"] + EPS)

        op_entropy = entropy_from_positive([comm[k] for k in ["TI", "TE", "TM", "IE", "IM", "EM"]])

        dominant_channel = max(comm, key=comm.get)

        rec = {
            "sample_id": sid,
            "cancer_type": obs["cancer_type"].iloc[0],
            "n_interface": len(idx),
            "n_edges": len(edges),

            "comm_TI": comm["TI"],
            "comm_TE": comm["TE"],
            "comm_TM": comm["TM"],
            "comm_IE": comm["IE"],
            "comm_IM": comm["IM"],
            "comm_EM": comm["EM"],

            "frac_TI": frac["TI"],
            "frac_TE": frac["TE"],
            "frac_TM": frac["TM"],
            "frac_IE": frac["IE"],
            "frac_IM": frac["IM"],
            "frac_EM": frac["EM"],

            "tumor_sector_strength": tumor_sector,
            "immune_sector_strength": immune_sector,
            "tumor_sector_fraction": tumor_sector_frac,
            "immune_sector_fraction": immune_sector_frac,

            "tm_dominance": tm_dominance,
            "immune_to_TM_ratio": immune_to_tm_ratio,
            "operator_entropy": op_entropy,
            "dominant_channel": dominant_channel,

            "TM_is_dominant": dominant_channel == "TM",
            "immune_sector_gt_TM": immune_sector > comm["TM"],
        }

        records.append(rec)

        print(
            f"[{sid:14s}] {rec['cancer_type']:12s} "
            f"dom={dominant_channel:2s} "
            f"TM_frac={tm_dominance:.3f} "
            f"immune_frac={immune_sector_frac:.3f} "
            f"entropy={op_entropy:.3f}"
        )

    out = pd.DataFrame(records)
    out.to_csv(OUT, index=False)

    print(f"\nSaved → {OUT}")

    if len(out):
        print("\nCohort summary:")
        print(f"  Samples: {len(out)}")
        print(f"  TM dominant: {int(out['TM_is_dominant'].sum())}/{len(out)}")
        print(f"  Immune sector > TM: {int(out['immune_sector_gt_TM'].sum())}/{len(out)}")
        print(f"  Median TM dominance: {out['tm_dominance'].median():.3f}")
        print(f"  Median immune-sector fraction: {out['immune_sector_fraction'].median():.3f}")
        print(f"  Median operator entropy: {out['operator_entropy'].median():.3f}")

        try:
            p_tm = binomtest(int(out["TM_is_dominant"].sum()), len(out), p=1/6, alternative="greater").pvalue
        except Exception:
            p_tm = np.nan

        print(f"  Binomial p(TM dominant vs 1/6): {p_tm:.3e}")

        print("\nBy cancer type:")
        by = out.groupby("cancer_type").agg(
            n=("sample_id", "count"),
            n_TM_dominant=("TM_is_dominant", lambda x: int(x.sum())),
            median_TM_dominance=("tm_dominance", "median"),
            median_immune_sector_fraction=("immune_sector_fraction", "median"),
            median_immune_to_TM_ratio=("immune_to_TM_ratio", "median"),
            median_entropy=("operator_entropy", "median"),
        )
        print(by)

        print("\nDominant channel counts:")
        print(out["dominant_channel"].value_counts())


if __name__ == "__main__":
    main()
