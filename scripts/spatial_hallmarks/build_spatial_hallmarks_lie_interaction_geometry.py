#!/usr/bin/env python3

"""
Lie-inspired interaction-sector geometry for Spatial Hallmarks pan-cancer Visium cohort.

Input:
    results_spatial_hallmarks/spatial_hallmarks_ncg_commutators.csv
    results_spatial_hallmarks/spatial_hallmarks_kts_states.csv

Output:
    results_spatial_hallmarks/spatial_hallmarks_lie_interaction_geometry.csv
    results_spatial_hallmarks/spatial_hallmarks_lie_interaction_geometry_summary.csv

Important:
    This is NOT a claim that tumour biology forms a true Lie algebra.
    This script tests interaction-sector geometry:
        - tumour sector
        - immune sector
        - TM dominance
        - immune closure
        - Jacobi-inspired triad inconsistency proxy

The Jacobi-inspired proxy is a negative/control diagnostic.
If it is not significant across states/cancer types, that limits claims to
sector geometry rather than formal Lie-algebraic closure.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

NCG = Path("results_spatial_hallmarks/spatial_hallmarks_ncg_commutators.csv")
KTS = Path("results_spatial_hallmarks/spatial_hallmarks_kts_states.csv")

OUT = Path("results_spatial_hallmarks/spatial_hallmarks_lie_interaction_geometry.csv")
SUMMARY = Path("results_spatial_hallmarks/spatial_hallmarks_lie_interaction_geometry_summary.csv")

PAIRS = ["TI", "TE", "TM", "IE", "IM", "EM"]
COMM = [f"comm_{p}" for p in PAIRS]
EPS = 1e-12


def entropy(v):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0)
    s = v.sum()
    if s <= 0:
        return np.nan
    p = v / s
    return float(-np.sum(p * np.log(p + EPS)))


def compute_lie_metrics(row):
    TI = float(row["comm_TI"])
    TE = float(row["comm_TE"])
    TM = float(row["comm_TM"])
    IE = float(row["comm_IE"])
    IM = float(row["comm_IM"])
    EM = float(row["comm_EM"])

    v = np.array([TI, TE, TM, IE, IM, EM], dtype=float)
    total = v.sum() + EPS

    tumor_sector = np.array([TI, TE, TM], dtype=float)
    immune_sector = np.array([IE, IM, EM], dtype=float)
    exhaustion_sector = np.array([TE, IE, EM], dtype=float)

    tumor_strength = tumor_sector.sum()
    immune_strength = immune_sector.sum()
    exhaustion_strength = exhaustion_sector.sum()

    tm_dominance = TM / total
    immune_sector_fraction = immune_strength / total
    tumor_sector_fraction = tumor_strength / total

    # immune sector relative to all non-TM interactions
    non_tm = TI + TE + IE + IM + EM + EPS
    immune_closure = immune_strength / non_tm

    algebra_entropy = entropy(v)
    immune_sector_entropy = entropy(immune_sector)

    # Triad entropies; variance is a Jacobi-inspired inconsistency proxy.
    tri_TIE = np.array([TI, TE, IE])
    tri_TIM = np.array([TI, TM, IM])
    tri_TEM = np.array([TE, TM, EM])
    tri_IEM = np.array([IE, IM, EM])

    tri_entropies = np.array([
        entropy(tri_TIE),
        entropy(tri_TIM),
        entropy(tri_TEM),
        entropy(tri_IEM),
    ])

    jacobi_inconsistency_proxy = float(np.nanvar(tri_entropies))

    # Interpretive sector score
    lie_immune_geometry_score = (
        immune_closure * immune_sector_entropy / (tm_dominance + EPS)
    )

    return pd.Series({
        "tumor_sector_strength": tumor_strength,
        "immune_sector_strength": immune_strength,
        "exhaustion_sector_strength": exhaustion_strength,

        "tumor_sector_fraction": tumor_sector_fraction,
        "immune_sector_fraction_lie": immune_sector_fraction,

        "tm_dominance_lie": tm_dominance,
        "immune_closure": immune_closure,

        "algebra_entropy_lie": algebra_entropy,
        "immune_sector_entropy": immune_sector_entropy,
        "jacobi_inconsistency_proxy": jacobi_inconsistency_proxy,
        "lie_immune_geometry_score": lie_immune_geometry_score,
    })


def kruskal_by_group(df, group_col, metric):
    groups = [g[metric].dropna().values for _, g in df.groupby(group_col)]
    groups = [x for x in groups if len(x) > 0]
    if len(groups) < 2:
        return np.nan, np.nan
    try:
        stat, p = kruskal(*groups)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def main():
    if not NCG.exists():
        raise FileNotFoundError(NCG)

    df = pd.read_csv(NCG)

    missing = [c for c in COMM if c not in df.columns]
    if missing:
        raise ValueError(f"Missing commutator columns: {missing}")

    lie = df.apply(compute_lie_metrics, axis=1)
    out = pd.concat([df, lie], axis=1)

    # Remove duplicate columns defensively.
    # Duplicate names make out[m].median() return a Series/DataFrame-like object,
    # which causes: TypeError: cannot convert the series to float.
    if out.columns.duplicated().any():
        dupes = out.columns[out.columns.duplicated()].tolist()
        print("Dropping duplicated columns:", dupes)
        out = out.loc[:, ~out.columns.duplicated()].copy()

    if KTS.exists():
        kts = pd.read_csv(KTS)

        # Only import kts_state to avoid duplicate numeric columns
        keep = [c for c in ["sample_id", "kts_state"] if c in kts.columns]

        if len(keep) > 1:
            out = out.merge(
                kts[keep].drop_duplicates("sample_id"),
                on="sample_id",
                how="left",
            )

    out.to_csv(OUT, index=False)

    metrics = [
        "tumor_sector_strength",
        "immune_sector_strength",
        "exhaustion_sector_strength",
        "tumor_sector_fraction",
        "immune_sector_fraction_lie",
        "tm_dominance_lie",
        "immune_closure",
        "algebra_entropy_lie",
        "immune_sector_entropy",
        "jacobi_inconsistency_proxy",
        "lie_immune_geometry_score",
    ]

    rows = []

    for m in metrics:
        h_cancer, p_cancer = kruskal_by_group(out, "cancer_type", m)

        row = {
            "metric": m,
            "median": float(out[m].median()),
            "iqr_low": float(out[m].quantile(0.25)),
            "iqr_high": float(out[m].quantile(0.75)),
            "kruskal_cancer_H": h_cancer,
            "kruskal_cancer_p": p_cancer,
        }

        if "kts_state" in out.columns:
            h_state, p_state = kruskal_by_group(out, "kts_state", m)
            row["kruskal_kts_H"] = h_state
            row["kruskal_kts_p"] = p_state

        if "interface_vs_tumor_enrichment" in out.columns:
            rho, p = spearmanr(out[m], out["interface_vs_tumor_enrichment"], nan_policy="omit")
            row["spearman_vs_enrichment"] = float(rho)
            row["spearman_vs_enrichment_p"] = float(p)

        if "coexact_fraction" in out.columns:
            rho, p = spearmanr(out[m], out["coexact_fraction"], nan_policy="omit")
            row["spearman_vs_coexact_fraction"] = float(rho)
            row["spearman_vs_coexact_fraction_p"] = float(p)

        if "spatial_score" in out.columns:
            rho, p = spearmanr(out[m], out["spatial_score"], nan_policy="omit")
            row["spearman_vs_spatial_score"] = float(rho)
            row["spearman_vs_spatial_score_p"] = float(p)

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("kruskal_cancer_p")
    summary.to_csv(SUMMARY, index=False)

    print("\nSpatial Hallmarks Lie-inspired interaction-sector geometry")
    print("=========================================================")
    print(f"Samples: {len(out)}")

    print("\nGroup medians by cancer type:")
    print(out.groupby("cancer_type")[metrics].median(numeric_only=True))

    if "kts_state" in out.columns:
        print("\nGroup medians by KTS-like state:")
        print(out.groupby("kts_state")[metrics].median(numeric_only=True))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved → {OUT}")
    print(f"Saved → {SUMMARY}")

    print("\nInterpretation:")
    print("  tm_dominance_lie high              -> TM backbone concentration")
    print("  immune_closure high                -> immune sector explains non-TM algebra")
    print("  lie_immune_geometry_score high     -> immune-rich, non-collapsed sector geometry")
    print("  jacobi_inconsistency_proxy null    -> limits claim to sector geometry, not formal Lie algebra")
    print("  This is an external pan-cancer sector-geometry analysis, not response validation.")


if __name__ == "__main__":
    main()
