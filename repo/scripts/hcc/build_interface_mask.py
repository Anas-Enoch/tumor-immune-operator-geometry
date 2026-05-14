#!/usr/bin/env python3
"""
build_interface_mask.py — HCC Interface Burden
===============================================
Computes a spatial interface proxy and interface_fraction per sample.

Interface definition
--------------------
interface_like = high tumor_score AND high tcell_score

Both thresholds are derived from the per-sample distribution (Q75),
not the global distribution. This prevents the zero-inflation problem
that occurred when global Q75(tumor_score) = 0.0 under the original
EPCAM/KRT19 markers: a global threshold of 0.0 admits half the tissue
regardless of biological content.

Per-sample thresholds ensure:
  - threshold adapts to local signal range
  - zero-inflated samples produce fewer interface_like spots (correct)
  - high-signal samples use a meaningful positive threshold (correct)

Additionally, a minimum threshold floor is applied: if the per-sample
Q75 is below the floor, the sample is flagged as low-signal and its
interface_fraction is reported with a warning column.

Output
------
results_interface_fraction.csv
  sample_id, patient_id, diagnosis, Response, treatment,
  interface_fraction, tumor_q75, immune_q75, low_tumor_signal
"""

import scanpy as sc
import pandas as pd
import numpy as np

ADATA   = "data/hepatocellular_carcinoma/hcc_scored.h5ad"
OUTFILE = "results_interface_fraction.csv"

# Minimum Q75 floor below which tumor signal is flagged as low
TUMOR_FLOOR = 0.05

print("Loading scored dataset …")
adata = sc.read_h5ad(ADATA)
print(f"  {adata.n_obs:,} spots × {adata.n_vars:,} genes")

# ── Clean sample-level metadata ────────────────────────────────────────────
# adata.obs is spot-level. One row per sample is needed for the merge.
# Drop at sample_id level to avoid Cartesian explosion.
sample_meta = (
    adata.obs[
        ["sample_id", "patient_id", "diagnosis", "Response", "treatment"]
    ]
    .drop_duplicates(subset=["sample_id"])
    .copy()
)
print(f"\n{len(sample_meta)} unique samples found")

# ── Per-sample interface fraction ──────────────────────────────────────────
records = []

for sample_id, grp in adata.obs.groupby("sample_id", observed=True):
    n = len(grp)

    # Per-sample Q75 thresholds
    tumor_q75  = grp["tumor_score"].quantile(0.75)
    immune_q75 = grp["tcell_score"].quantile(0.75)

    # Low-signal flag
    low_tumor = bool(tumor_q75 < TUMOR_FLOOR)

    # Interface mask
    interface_like = (
        (grp["tumor_score"] > tumor_q75) &
        (grp["tcell_score"] > immune_q75)
    )
    frac = float(interface_like.mean())

    records.append({
        "sample_id":          sample_id,
        "n_spots":            n,
        "interface_fraction": frac,
        "tumor_q75":          round(tumor_q75, 4),
        "immune_q75":         round(immune_q75, 4),
        "low_tumor_signal":   low_tumor,
    })

    flag = "  ⚠ LOW TUMOR SIGNAL" if low_tumor else ""
    print(
        f"  {sample_id:<28} frac={frac:.4f}  "
        f"tumor_q75={tumor_q75:.4f}  immune_q75={immune_q75:.4f}{flag}"
    )

fractions = pd.DataFrame(records)

# ── Merge with sample metadata ─────────────────────────────────────────────
summary = sample_meta.merge(fractions, on="sample_id", how="left")
summary = summary.sort_values(["patient_id", "diagnosis", "sample_id"])

# ── Patient-level delta ────────────────────────────────────────────────────
print("\nPer-patient interface fraction delta (post − pre):")
for pid, grp in summary.groupby("patient_id"):
    pre  = grp[grp["diagnosis"] == "Pre"]
    post = grp[grp["diagnosis"] == "Post"]
    if len(pre) and len(post):
        delta   = post["interface_fraction"].values[0] - pre["interface_fraction"].values[0]
        resp    = grp["Response"].values[0]
        flag    = "⚠" if grp["low_tumor_signal"].any() else ""
        print(
            f"  Patient {pid}  ({resp:<13})  "
            f"pre={pre['interface_fraction'].values[0]:.4f}  "
            f"post={post['interface_fraction'].values[0]:.4f}  "
            f"delta={delta:+.4f}  {flag}"
        )

# ── Group summary ──────────────────────────────────────────────────────────
print("\nGroup-level summary (delta):")
deltas = []
for pid, grp in summary.groupby("patient_id"):
    pre  = grp[grp["diagnosis"] == "Pre"]
    post = grp[grp["diagnosis"] == "Post"]
    if len(pre) and len(post):
        deltas.append({
            "patient_id": pid,
            "Response":   grp["Response"].values[0],
            "delta":      post["interface_fraction"].values[0]
                          - pre["interface_fraction"].values[0],
            "low_signal": grp["low_tumor_signal"].any(),
        })
delta_df = pd.DataFrame(deltas)
for resp, g in delta_df.groupby("Response"):
    med = g["delta"].median()
    n_up = (g["delta"] > 0).sum()
    print(f"  {resp:<15} median_delta={med:+.4f}  UP={n_up}/{len(g)}")

# ── Save ───────────────────────────────────────────────────────────────────
summary.to_csv(OUTFILE, index=False)
print(f"\nSaved → {OUTFILE}")
