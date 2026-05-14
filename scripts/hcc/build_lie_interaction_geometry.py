#!/usr/bin/env python3
"""
Lie-algebraic interaction geometry for HCC interface commutator modes.

Purpose:
- Treat six interaction channels TI, TE, TM, IE, IM, EM as generators.
- Test algebraic sector structure:
    TM backbone
    immune-immune sector
    exhaustion-coupling sector
- Compute closure-like diagnostics and Jacobi-inspired inconsistency.

This is NOT claiming genes form a true Lie algebra.
It is an interaction-geometry diagnostic based on commutator-channel structure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

INFILE = Path("results_hcc/results_hcc_interaction_mode_scores.csv")
OUT = Path("results_hcc/results_hcc_lie_interaction_geometry.csv")
SUMMARY = Path("results_hcc/results_hcc_lie_interaction_geometry_summary.csv")

PAIRS = ["TI", "TE", "TM", "IE", "IM", "EM"]
COMM = [f"comm_{p}" for p in PAIRS]
EPS = 1e-12

df = pd.read_csv(INFILE)

missing = [c for c in COMM if c not in df.columns]
if missing:
    raise ValueError(f"Missing commutator columns: {missing}")

def norm(v):
    return float(np.linalg.norm(np.asarray(v, dtype=float)))

def entropy(v):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0)
    p = v / (v.sum() + EPS)
    return float(-np.sum(p * np.log(p + EPS)))

records = []

for _, row in df.iterrows():
    v = row[COMM].astype(float).values
    total = v.sum() + EPS

    TI, TE, TM, IE, IM, EM = v

    tumor_sector = np.array([TI, TE, TM])
    immune_sector = np.array([IE, IM, EM])
    exhaustion_sector = np.array([TE, IE, EM])

    # Sector strengths
    tumor_strength = tumor_sector.sum()
    immune_strength = immune_sector.sum()
    exhaustion_strength = exhaustion_sector.sum()

    # Lie-inspired closure proxy:
    # immune sector relative to all non-TM interactions.
    non_tm = TI + TE + IE + IM + EM + EPS
    immune_closure = immune_strength / non_tm

    # Algebraic collapse proxy:
    tm_dominance = TM / total

    # Interaction diversity
    algebra_entropy = entropy(v)

    # Sector balance: immune sector should not be concentrated in only one channel.
    immune_entropy = entropy(immune_sector)

    # Jacobi-inspired inconsistency proxy:
    # If triple interaction sectors are unbalanced, this rises.
    # Uses three interaction triangles:
    #   T-I-E: TI, TE, IE
    #   T-I-M: TI, TM, IM
    #   T-E-M: TE, TM, EM
    #   I-E-M: IE, IM, EM
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

    jacobi_inconsistency = float(np.var(tri_entropies))

    # Response-associated Lie-sector score:
    # high immune closure + high immune entropy + low TM collapse
    lie_immune_geometry_score = (
        immune_closure * immune_entropy / (tm_dominance + EPS)
    )

    records.append({
        "sample_id": row["sample_id"],
        "response": row["response"],
        "tumor_sector_strength": tumor_strength,
        "immune_sector_strength": immune_strength,
        "exhaustion_sector_strength": exhaustion_strength,
        "tm_dominance": tm_dominance,
        "immune_closure": immune_closure,
        "algebra_entropy": algebra_entropy,
        "immune_sector_entropy": immune_entropy,
        "jacobi_inconsistency_proxy": jacobi_inconsistency,
        "lie_immune_geometry_score": lie_immune_geometry_score,
    })

out = pd.DataFrame(records)

# Attach useful original scores if available
for c in ["kts_state", "coexact_fraction", "spatial_score", "mode_immune_immune_abs", "immune_to_TM_ratio"]:
    if c in df.columns:
        out[c] = df[c].values

out.to_csv(OUT, index=False)

metrics = [
    "tumor_sector_strength",
    "immune_sector_strength",
    "tm_dominance",
    "immune_closure",
    "algebra_entropy",
    "immune_sector_entropy",
    "jacobi_inconsistency_proxy",
    "lie_immune_geometry_score",
]

R = out[out["response"] == "Responder"]
NR = out[out["response"] == "Non_Responder"]

rows = []
for m in metrics:
    x = R[m].dropna()
    y = NR[m].dropna()

    if len(x) and len(y):
        _, pg = mannwhitneyu(x, y, alternative="greater")
        _, pl = mannwhitneyu(x, y, alternative="less")
        _, p2 = mannwhitneyu(x, y, alternative="two-sided")

        rec = {
            "metric": m,
            "R_median": float(x.median()),
            "NR_median": float(y.median()),
            "fold_R_over_NR": float((x.median() + EPS) / (y.median() + EPS)),
            "p_R_greater_NR": float(pg),
            "p_R_less_NR": float(pl),
            "p_two_sided": float(p2),
        }

        if "coexact_fraction" in out.columns:
            rho, p = spearmanr(out[m], out["coexact_fraction"], nan_policy="omit")
            rec["spearman_vs_coexact"] = float(rho)
            rec["spearman_vs_coexact_p"] = float(p)

        rows.append(rec)

summary = pd.DataFrame(rows)
summary.to_csv(SUMMARY, index=False)

print("\nLie-algebraic interaction geometry")
print("==================================")
print(f"Samples: {len(out)}")

print("\nGroup medians:")
print(out.groupby("response")[metrics].median(numeric_only=True))

print("\nTests:")
print(summary.to_string(index=False))

if "kts_state" in out.columns:
    print("\nKTS-state medians:")
    print(out.groupby("kts_state")[metrics].median(numeric_only=True))

print(f"\nSaved → {OUT}")
print(f"Saved → {SUMMARY}")

print("\nInterpretation:")
print("  tm_dominance high              -> algebraic collapse toward tumour-myeloid backbone")
print("  immune_closure high            -> immune sector explains non-TM algebra")
print("  immune_sector_entropy high     -> balanced immune-immune sector")
print("  lie_immune_geometry_score high -> immune-rich non-collapsed interaction geometry")
print("  Use manuscript-safe wording: Lie-inspired interaction-sector geometry.")
