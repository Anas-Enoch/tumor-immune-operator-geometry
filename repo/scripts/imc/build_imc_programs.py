from pathlib import Path

import numpy as np
import pandas as pd

INFILE = Path("results/imc/ici_cell_table.csv")
OUTFILE = Path("results/imc/ici_programs.csv")

df = pd.read_csv(INFILE)

print("Initial cells:", len(df))

# --------------------------------------------------
# QC FILTERS
# --------------------------------------------------

# Remove tiny segmentation fragments
df = df[df["cell_area"] >= 20].copy()

# Remove extreme giant objects
hi = df["cell_area"].quantile(0.995)
df = df[df["cell_area"] <= hi].copy()

print("After QC:", len(df))

# --------------------------------------------------
# PROGRAM SCORES
# --------------------------------------------------

def zscore(x):
    x = np.asarray(x)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-8)

# Tumor program
tumor_markers = [
    "PanKeratin",
    "Ecadherin",
    "Arginase1",
    "CK7",
]

# T-cell program
tcell_markers = [
    "CD3",
    "CD8a",
    "CD4",
]

# Exhaustion program
exhaustion_markers = [
    "PD1",
    "Tim3",
    "TOX",
    "CD39",
    "PDL1",
]

# Myeloid program
myeloid_markers = [
    "CD68",
    "CD163",
    "CD204",
    "CD33",
    "HLA-DR",
]

for markers in [
    tumor_markers,
    tcell_markers,
    exhaustion_markers,
    myeloid_markers,
]:
    for m in markers:
        if m not in df.columns:
            print(f"WARNING missing marker: {m}")

df["tumor_score"] = np.mean(
    [zscore(df[m]) for m in tumor_markers if m in df.columns],
    axis=0,
)

df["tcell_score"] = np.mean(
    [zscore(df[m]) for m in tcell_markers if m in df.columns],
    axis=0,
)

df["exhaustion_score"] = np.mean(
    [zscore(df[m]) for m in exhaustion_markers if m in df.columns],
    axis=0,
)

df["myeloid_score"] = np.mean(
    [zscore(df[m]) for m in myeloid_markers if m in df.columns],
    axis=0,
)

# --------------------------------------------------
# INTERFACE LABEL
# --------------------------------------------------

tumor_q75 = (
    df.groupby("sample_id")["tumor_score"]
    .transform(lambda x: x.quantile(0.75))
)

tcell_q75 = (
    df.groupby("sample_id")["tcell_score"]
    .transform(lambda x: x.quantile(0.75))
)

df["is_interface"] = (
    (df["tumor_score"] > tumor_q75)
    &
    (df["tcell_score"] > tcell_q75)
)

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

summary = (
    df.groupby("sample_id")
    .agg(
        n_cells=("cell_id", "size"),
        n_interface=("is_interface", "sum"),
        tumor_mean=("tumor_score", "mean"),
        tcell_mean=("tcell_score", "mean"),
        exhaustion_mean=("exhaustion_score", "mean"),
        myeloid_mean=("myeloid_score", "mean"),
    )
)

print(summary.head())

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTFILE, index=False)

print("\nSaved:", OUTFILE)
print(df.shape)
