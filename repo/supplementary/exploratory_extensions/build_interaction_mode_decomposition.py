#!/usr/bin/env python3

"""
build_interaction_mode_decomposition.py

Representation-inspired interaction-mode decomposition of HCC interface
commutator algebra.

Purpose
-------
Decompose the 6 pairwise commutator channels:

    TI, TE, TM, IE, IM, EM

into low-dimensional interaction modes.

This is NOT a literal Peter-Weyl theorem application.
It is a representation-inspired decomposition of biological interaction
algebra into interpretable sectors:

    Mode 1: tumor-myeloid backbone
    Mode 2: immune-immune algebra
    Mode 3: exhaustion / mixed coupling
    Mode 4: residual interaction structure

Inputs
------
    results_hcc_ncg_commutators.csv
    results_hcc_kts_states.csv          optional

Outputs
-------
    results_hcc_interaction_mode_scores.csv
    results_hcc_interaction_mode_loadings.csv
    results_hcc_interaction_mode_summary.csv

Main questions
--------------
1. Does the first mode capture the universal TM backbone?
2. Does an immune-immune mode separate responders?
3. Do KTS states differ by mode loading?
4. Does mode decomposition compress NCG into interpretable sectors?
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, NMF
from scipy.stats import mannwhitneyu, spearmanr


# ============================================================
# CONFIG
# ============================================================

NCG = Path("results_hcc/results_hcc_ncg_commutators.csv")
KTS = Path("results_hcc/results_hcc_kts_states.csv")

OUT_SCORES = Path("results_hcc/results_hcc_interaction_mode_scores.csv")
OUT_LOADINGS = Path("results_hcc/results_hcc_interaction_mode_loadings.csv")
OUT_SUMMARY = Path("results_hcc/results_hcc_interaction_mode_summary.csv")

PAIRS = ["TI", "TE", "TM", "IE", "IM", "EM"]
COMM_COLS = [f"comm_{p}" for p in PAIRS]
FRAC_COLS = [f"frac_{p}" for p in PAIRS]

N_COMPONENTS = 3
EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def normalize_response(x):
    x = str(x)
    if x in ["R", "Responder"]:
        return "Responder"
    if x in ["NR", "Non_Responder", "Non-responder", "NonResponder"]:
        return "Non_Responder"
    return x


def hardcoded_response_map():
    return {
        "cytassist_70_pre":  "Responder",
        "cytassist_70_post": "Responder",
        "cytassist_71_pre":  "Responder",
        "cytassist_71_post": "Responder",
        "cytassist_76_pre":  "Responder",
        "cytassist_76_post": "Responder",
        "cytassist_83_pre":  "Responder",
        "cytassist_83_post": "Responder",
        "cytassist_84_pre":  "Responder",
        "cytassist_84_post": "Responder",

        "cytassist_72_pre":  "Non_Responder",
        "cytassist_72_post": "Non_Responder",
        "cytassist_73_pre":  "Non_Responder",
        "cytassist_73_post": "Non_Responder",
        "cytassist_74_pre":  "Non_Responder",
        "cytassist_74_post": "Non_Responder",
        "cytassist_79_pre":  "Non_Responder",
        "cytassist_79_post": "Non_Responder",
        "cytassist_85_pre":  "Non_Responder",
        "cytassist_85_post": "Non_Responder",
        "cytassist_86_pre":  "Non_Responder",
        "cytassist_86_post": "Non_Responder",
    }


def infer_sample_id(df):
    if "sample_id" in df.columns:
        return df
    for c in ["sample", "section", "section_id"]:
        if c in df.columns:
            return df.rename(columns={c: "sample_id"})
    raise ValueError(f"No sample_id column found. Columns: {df.columns.tolist()}")


def ensure_commutator_fractions(df):
    """
    Ensure comm_* and frac_* columns exist.
    If frac columns are absent, compute them from comm columns.
    """
    missing_comm = [c for c in COMM_COLS if c not in df.columns]
    if missing_comm:
        raise ValueError(f"Missing commutator columns: {missing_comm}")

    for c in COMM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    total = df[COMM_COLS].sum(axis=1) + EPS

    for p, cfrac, ccomm in zip(PAIRS, FRAC_COLS, COMM_COLS):
        if cfrac not in df.columns:
            df[cfrac] = df[ccomm] / total
        else:
            df[cfrac] = pd.to_numeric(df[cfrac], errors="coerce")

    return df


def label_pca_mode(loadings_row):
    """
    Assign interpretable label from component loadings.
    """
    vals = loadings_row[PAIRS].astype(float)

    tm = abs(vals["TM"])
    immune = abs(vals["IE"]) + abs(vals["IM"]) + abs(vals["EM"])
    tumor_immune = abs(vals["TI"]) + abs(vals["TE"])
    exhaustion = abs(vals["TE"]) + abs(vals["IE"]) + abs(vals["EM"])

    if tm >= max(immune, tumor_immune, exhaustion):
        return "TM_backbone_mode"
    if immune >= max(tm, tumor_immune):
        return "immune_immune_mode"
    if exhaustion >= max(tm, immune):
        return "exhaustion_coupling_mode"
    return "mixed_interaction_mode"


def label_nmf_mode(loadings_row):
    vals = loadings_row[PAIRS].astype(float)

    top = vals.sort_values(ascending=False).index[:3].tolist()

    if top[0] == "TM":
        return "TM_backbone_mode"
    if all(x in ["IE", "IM", "EM"] for x in top[:2]):
        return "immune_immune_mode"
    if any(x in ["TE", "IE", "EM"] for x in top):
        return "exhaustion_coupling_mode"
    return "mixed_interaction_mode"


def add_manual_scores(df):
    """
    Biologically interpretable fixed decomposition.
    These are not learned components; they are predefined sectors.
    """
    total = df[COMM_COLS].sum(axis=1) + EPS

    df["mode_TM_backbone"] = df["comm_TM"] / total
    df["mode_tumor_immune"] = (df["comm_TI"] + df["comm_TE"]) / total
    df["mode_immune_immune"] = (df["comm_IE"] + df["comm_IM"] + df["comm_EM"]) / total
    df["mode_exhaustion_axis"] = (df["comm_TE"] + df["comm_IE"] + df["comm_EM"]) / total

    # Absolute immune-immune strength, useful because fraction can be small under huge TM.
    df["mode_immune_immune_abs"] = df["comm_IE"] + df["comm_IM"] + df["comm_EM"]

    # Immune algebra diversity within immune-immune sector
    immune_sum = df["mode_immune_immune_abs"] + EPS
    pIE = df["comm_IE"] / immune_sum
    pIM = df["comm_IM"] / immune_sum
    pEM = df["comm_EM"] / immune_sum
    df["mode_immune_immune_entropy"] = -(
        pIE * np.log(pIE + EPS)
        + pIM * np.log(pIM + EPS)
        + pEM * np.log(pEM + EPS)
    )

    # Ratio: immune-immune algebra relative to TM backbone
    df["immune_to_TM_ratio"] = df["mode_immune_immune_abs"] / (df["comm_TM"] + EPS)

    return df


# ============================================================
# LOAD
# ============================================================

if not NCG.exists():
    raise FileNotFoundError(f"Missing NCG CSV: {NCG}")

df = pd.read_csv(NCG)
df = infer_sample_id(df)
df = ensure_commutator_fractions(df)

# response
if "response" not in df.columns:
    resp_map = hardcoded_response_map()
    df["response"] = df["sample_id"].map(resp_map)
else:
    df["response"] = df["response"].map(normalize_response)

missing_resp = df.loc[df["response"].isna(), "sample_id"].tolist()
if missing_resp:
    raise ValueError(f"Missing response mapping for sample IDs: {missing_resp}")

# Attach KTS
if KTS.exists():
    kts = pd.read_csv(KTS)
    kts = infer_sample_id(kts)

    keep = [c for c in ["sample_id", "kts_state", "spatial_score", "coexact_fraction"] if c in kts.columns]

    if len(keep) > 1:
        df = df.merge(kts[keep].drop_duplicates("sample_id"), on="sample_id", how="left")


# ============================================================
# MANUAL BIOLOGICAL MODE SCORES
# ============================================================

df = add_manual_scores(df)


# ============================================================
# PCA ON FRACTIONAL COMMUTATOR ALGEBRA
# ============================================================

Xfrac = df[FRAC_COLS].astype(float).values

scaler = StandardScaler()
Xfrac_z = scaler.fit_transform(Xfrac)

ncomp = min(N_COMPONENTS, Xfrac_z.shape[0], Xfrac_z.shape[1])

pca = PCA(n_components=ncomp, random_state=0)
Zpca = pca.fit_transform(Xfrac_z)

for i in range(ncomp):
    df[f"pca_mode_{i+1}_score"] = Zpca[:, i]

pca_load = pd.DataFrame(
    pca.components_,
    columns=PAIRS,
)

pca_load.insert(0, "component", [f"PCA{i+1}" for i in range(ncomp)])
pca_load["explained_variance_ratio"] = pca.explained_variance_ratio_
pca_load["mode_label"] = pca_load.apply(label_pca_mode, axis=1)


# ============================================================
# NMF ON FRACTIONAL COMMUTATOR ALGEBRA
# ============================================================

# Fractions are nonnegative and sum to 1 -> NMF is interpretable.
ncomp_nmf = min(N_COMPONENTS, Xfrac.shape[0], Xfrac.shape[1])

try:
    nmf = NMF(
        n_components=ncomp_nmf,
        init="nndsvda",
        random_state=0,
        max_iter=2000,
    )

    Znmf = nmf.fit_transform(np.maximum(Xfrac, 0))

    H = nmf.components_

    # Normalize mode loadings to sum 1
    Hnorm = H / (H.sum(axis=1, keepdims=True) + EPS)

    for i in range(ncomp_nmf):
        df[f"nmf_mode_{i+1}_score"] = Znmf[:, i]

    nmf_load = pd.DataFrame(Hnorm, columns=PAIRS)
    nmf_load.insert(0, "component", [f"NMF{i+1}" for i in range(ncomp_nmf)])
    nmf_load["reconstruction_error"] = nmf.reconstruction_err_
    nmf_load["mode_label"] = nmf_load.apply(label_nmf_mode, axis=1)

except Exception as e:
    print(f"NMF failed: {e}")
    nmf_load = pd.DataFrame()


# ============================================================
# SAVE LOADINGS
# ============================================================

loadings = []

pca_tmp = pca_load.copy()
pca_tmp.insert(0, "method", "PCA")
loadings.append(pca_tmp)

if not nmf_load.empty:
    nmf_tmp = nmf_load.copy()
    nmf_tmp.insert(0, "method", "NMF")
    loadings.append(nmf_tmp)

loadings = pd.concat(loadings, ignore_index=True)

loadings.to_csv(OUT_LOADINGS, index=False)


# ============================================================
# TESTS / SUMMARY
# ============================================================

test_metrics = [
    "mode_TM_backbone",
    "mode_tumor_immune",
    "mode_immune_immune",
    "mode_exhaustion_axis",
    "mode_immune_immune_abs",
    "mode_immune_immune_entropy",
    "immune_to_TM_ratio",
]

for c in df.columns:
    if c.startswith("pca_mode_") and c.endswith("_score"):
        test_metrics.append(c)

for c in df.columns:
    if c.startswith("nmf_mode_") and c.endswith("_score"):
        test_metrics.append(c)

summary_rows = []

R = df[df["response"] == "Responder"]
NR = df[df["response"] == "Non_Responder"]

for m in test_metrics:
    if m not in df.columns:
        continue

    x = R[m].dropna()
    y = NR[m].dropna()

    if len(x) == 0 or len(y) == 0:
        continue

    _, p_greater = mannwhitneyu(x, y, alternative="greater")
    _, p_less = mannwhitneyu(x, y, alternative="less")
    _, p_two = mannwhitneyu(x, y, alternative="two-sided")

    row = {
        "metric": m,
        "R_median": float(x.median()),
        "NR_median": float(y.median()),
        "fold_R_over_NR": float((x.median() + EPS) / (y.median() + EPS)),
        "p_R_greater_NR": float(p_greater),
        "p_R_less_NR": float(p_less),
        "p_two_sided": float(p_two),
    }

    if "coexact_fraction" in df.columns:
        rho, p = spearmanr(df[m], df["coexact_fraction"], nan_policy="omit")
        row["spearman_vs_coexact_fraction"] = float(rho)
        row["spearman_vs_coexact_p"] = float(p)

    if "spatial_score" in df.columns:
        rho, p = spearmanr(df[m], df["spatial_score"], nan_policy="omit")
        row["spearman_vs_spatial_score"] = float(rho)
        row["spearman_vs_spatial_score_p"] = float(p)

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_SUMMARY, index=False)

df.to_csv(OUT_SCORES, index=False)


# ============================================================
# PRINT
# ============================================================

print("\nInteraction-mode decomposition")
print("================================")
print(f"Samples: {len(df)}")
print("\nResponse counts:")
print(df["response"].value_counts())

if "kts_state" in df.columns:
    print("\nKTS counts:")
    print(df["kts_state"].value_counts(dropna=False))

print("\nPCA loadings:")
print(pca_load.to_string(index=False))

if not nmf_load.empty:
    print("\nNMF loadings:")
    print(nmf_load.to_string(index=False))

print("\nManual biological mode group medians:")
manual_cols = [
    "mode_TM_backbone",
    "mode_immune_immune",
    "mode_immune_immune_abs",
    "mode_immune_immune_entropy",
    "immune_to_TM_ratio",
]
print(df.groupby("response")[manual_cols].median(numeric_only=True))

print("\nResponder vs NonResponder tests:")
print(summary.to_string(index=False))

if "kts_state" in df.columns:
    print("\nKTS-state medians:")
    kts_cols = [
        "mode_TM_backbone",
        "mode_immune_immune",
        "mode_immune_immune_abs",
        "immune_to_TM_ratio",
    ]
    print(df.groupby("kts_state")[kts_cols].median(numeric_only=True))

print(f"\nSaved → {OUT_SCORES}")
print(f"Saved → {OUT_LOADINGS}")
print(f"Saved → {OUT_SUMMARY}")

print("\nInterpretation guide:")
print("  mode_TM_backbone high       -> tumor-myeloid background algebra")
print("  mode_immune_immune high     -> immune-interaction representation sector")
print("  immune_to_TM_ratio high     -> immune algebra rises relative to HCC TM backbone")
print("  PCA/NMF modes are exploratory compression, not literal Peter-Weyl theory")
print("  Best manuscript-safe wording: interaction-mode decomposition")
