#!/usr/bin/env python3

"""
Build KTS-like operator-regime states for the Spatial Hallmarks pan-cancer Visium cohort.

Inputs:
    results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv
    results_spatial_hallmarks/spatial_hallmarks_ncg_commutators.csv

Output:
    results_spatial_hallmarks/spatial_hallmarks_kts_states.csv

Purpose:
    Apply the original HCC-style operator-regime logic to a large external
    pan-cancer spatial cohort.

Important:
    This cohort has no therapy-response or pre/post labels.
    Therefore this script does NOT test response transitions.
    It assigns section-level operator regimes based on:
        coexact enrichment
        spatial score
        immune-sector fraction
        TM dominance
        operator entropy

Interpretation:
    This is cross-cancer regime composition, not clinical response prediction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

HODGE = Path("results_spatial_hallmarks/spatial_hallmarks_hodge_interface.csv")
NCG = Path("results_spatial_hallmarks/spatial_hallmarks_ncg_commutators.csv")
OUT = Path("results_spatial_hallmarks/spatial_hallmarks_kts_states.csv")

EPS = 1e-12


def qcut_label(x, q_low, q_high):
    if x <= q_low:
        return "low"
    if x >= q_high:
        return "high"
    return "mid"


def assign_state(row, thresholds):
    """
    KTS-like state labels for external pan-cancer validation.

    S0: weak / low coexact enrichment
    S1: fragmented interface, low spatial structure
    S2: immune-sector active interface
    S3: organized immune-sector coexact interface
    S4: TM-dominated organized interface

    These are section-level analogues, not therapy transition states.
    """
    coexact_high = row["interface_vs_tumor_enrichment"] >= thresholds["R_q66"]
    coexact_low = row["interface_vs_tumor_enrichment"] <= thresholds["R_q33"]

    spatial_high = row["spatial_score"] >= thresholds["spatial_q66"]
    spatial_low = row["spatial_score"] <= thresholds["spatial_q33"]

    immune_high = row["immune_sector_fraction"] >= thresholds["immune_q66"]
    tm_high = row["tm_dominance"] >= thresholds["tm_q66"]

    entropy_high = row["operator_entropy"] >= thresholds["entropy_q66"]

    if coexact_low and spatial_low:
        return "S0_weak"

    if coexact_high and spatial_high and immune_high:
        return "S3_organized_immune_front"

    if spatial_high and tm_high:
        return "S4_TM_dominated_organized"

    if coexact_high and immune_high:
        return "S2_immune_active"

    if spatial_low:
        return "S1_fragmented_or_diffuse"

    return "S2_mixed_interface"


def main():
    if not HODGE.exists():
        raise FileNotFoundError(HODGE)
    if not NCG.exists():
        raise FileNotFoundError(NCG)

    h = pd.read_csv(HODGE)
    n = pd.read_csv(NCG)

    df = h.merge(
        n,
        on=["sample_id", "cancer_type"],
        how="inner",
        suffixes=("", "_ncg"),
    )

    required = [
        "interface_vs_tumor_enrichment",
        "coexact_fraction",
        "spatial_score",
        "immune_sector_fraction",
        "tm_dominance",
        "operator_entropy",
    ]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    thresholds = {
        "R_q33": df["interface_vs_tumor_enrichment"].quantile(0.33),
        "R_q66": df["interface_vs_tumor_enrichment"].quantile(0.66),
        "spatial_q33": df["spatial_score"].quantile(0.33),
        "spatial_q66": df["spatial_score"].quantile(0.66),
        "immune_q33": df["immune_sector_fraction"].quantile(0.33),
        "immune_q66": df["immune_sector_fraction"].quantile(0.66),
        "tm_q33": df["tm_dominance"].quantile(0.33),
        "tm_q66": df["tm_dominance"].quantile(0.66),
        "entropy_q33": df["operator_entropy"].quantile(0.33),
        "entropy_q66": df["operator_entropy"].quantile(0.66),
    }

    df["kts_state"] = df.apply(lambda r: assign_state(r, thresholds), axis=1)

    df["coexact_enrichment_level"] = df["interface_vs_tumor_enrichment"].apply(
        lambda x: qcut_label(x, thresholds["R_q33"], thresholds["R_q66"])
    )
    df["spatial_structure_level"] = df["spatial_score"].apply(
        lambda x: qcut_label(x, thresholds["spatial_q33"], thresholds["spatial_q66"])
    )
    df["immune_sector_level"] = df["immune_sector_fraction"].apply(
        lambda x: qcut_label(x, thresholds["immune_q33"], thresholds["immune_q66"])
    )
    df["tm_dominance_level"] = df["tm_dominance"].apply(
        lambda x: qcut_label(x, thresholds["tm_q33"], thresholds["tm_q66"])
    )
    df["operator_entropy_level"] = df["operator_entropy"].apply(
        lambda x: qcut_label(x, thresholds["entropy_q33"], thresholds["entropy_q66"])
    )

    df.to_csv(OUT, index=False)

    print(f"\nSaved → {OUT}")

    print("\nKTS-like state counts:")
    print(df["kts_state"].value_counts())

    print("\nState counts by cancer type:")
    print(pd.crosstab(df["cancer_type"], df["kts_state"]))

    print("\nState medians:")
    med_cols = [
        "interface_vs_tumor_enrichment",
        "coexact_fraction",
        "spatial_score",
        "immune_sector_fraction",
        "tm_dominance",
        "operator_entropy",
    ]
    print(df.groupby("kts_state")[med_cols].median(numeric_only=True))

    print("\nCancer-type medians:")
    print(df.groupby("cancer_type")[med_cols].median(numeric_only=True))

    print("\nKruskal tests across cancer type:")
    for c in med_cols:
        groups = [g[c].dropna().values for _, g in df.groupby("cancer_type")]
        groups = [x for x in groups if len(x) > 0]
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                print(f"{c:36s} H={stat:.3f} p={p:.4f}")
            except Exception:
                pass

    print("\nInterpretation:")
    print("  S3_organized_immune_front = high coexact enrichment + high spatial score + high immune-sector fraction")
    print("  S4_TM_dominated_organized = organized but TM-dominant interface")
    print("  S0/S1 = weak, diffuse, or fragmented interface organization")
    print("  This is external cross-cancer regime composition, not response-state transition analysis.")


if __name__ == "__main__":
    main()
