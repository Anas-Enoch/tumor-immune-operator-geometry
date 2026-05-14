#!/usr/bin/env python3
"""
build_program_scores.py — HCC Program Scoring
==============================================
Computes four biological scalar fields from the HCC Visium AnnData object
using sc.tl.score_genes (Scanpy gene-set scoring).

Program definitions (HCC-appropriate)
--------------------------------------
tumor_score     GPC3, AFP, CPS1, ALDH3A1, FGB
                Hepatocyte-specific markers confirmed expressed in HCC.
                Replaces EPCAM/KRT19 (epithelial keratins — sparse in
                hepatocytes, produce zero-inflated scores in HCC).

tcell_score     CD3D, CD3E, CD8A, CD8B
                Pan-T-cell + cytotoxic T-cell markers.

exhaustion_score PDCD1, LAG3, TIGIT, CXCL13, TOX
                Core transcriptional exhaustion program. CXCL13 marks
                tumour-reactive exhausted T cells in HCC specifically.

myeloid_score   LYZ, S100A8, S100A9, CTSB, FCN1, TYMP
                6-gene myeloid panel. Replaces single-gene LYZ which
                produced pathological scaling (0–300 range) due to
                extreme Kupffer cell outliers.

Circularity boundary
--------------------
These gene sets define the wedge operator in build_wedge_field.py.
No gene from these sets should appear as a downstream validation marker.

Output
------
data/hepatocellular_carcinoma/hcc_scored.h5ad
"""

import scanpy as sc
import numpy as np
import pandas as pd

ADATA   = "data/hepatocellular_carcinoma/Visium-ST/visium_all.h5ad"
OUTFILE = "data/hepatocellular_carcinoma/hcc_scored.h5ad"

print("Loading dataset …")
adata = sc.read_h5ad(ADATA)
print(f"  {adata.n_obs:,} spots × {adata.n_vars:,} genes")

# ── Program definitions ────────────────────────────────────────────────────
PROGRAMS = {
    "tumor_score": [
        "GPC3",    # Glypican-3  — canonical HCC surface marker
        "AFP",     # Alpha-fetoprotein — secreted by malignant hepatocytes
        "CPS1",    # Carbamoyl-phosphate synthase — hepatocyte metabolic enzyme
        "ALDH3A1", # Aldehyde dehydrogenase — enriched in HCC cells
        "FGB",     # Fibrinogen beta — liver-specific synthesis
    ],
    "tcell_score": [
        "CD3D",   # Pan-T-cell
        "CD3E",   # Pan-T-cell
        "CD8A",   # Cytotoxic T-cell
        "CD8B",   # Cytotoxic T-cell
    ],
    "exhaustion_score": [
        "PDCD1",  # PD-1 — checkpoint receptor, canonical exhaustion
        "LAG3",   # LAG-3 — co-inhibitory receptor
        "TIGIT",  # TIGIT — co-inhibitory receptor
        "CXCL13", # B-cell attracting chemokine; marks tumour-reactive
                  # exhausted T cells in HCC specifically
        "TOX",    # Transcription factor driving exhaustion programme
    ],
    "myeloid_score": [
        "LYZ",    # Lysozyme — myeloid
        "S100A8", # Calprotectin subunit — monocyte/macrophage
        "S100A9", # Calprotectin subunit — monocyte/macrophage
        "CTSB",   # Cathepsin B — tumour-associated macrophage
        "FCN1",   # Ficolin-1 — classical monocyte marker
        "TYMP",   # Thymidine phosphorylase — macrophage activation
    ],
}

# ── Availability check ─────────────────────────────────────────────────────
print("\nGene availability check:")
available_var = set(adata.var_names)
for program, genes in PROGRAMS.items():
    found   = [g for g in genes if g in available_var]
    missing = [g for g in genes if g not in available_var]
    print(f"  {program}: {len(found)}/{len(genes)} found", end="")
    if missing:
        print(f"  — missing: {missing}", end="")
    print()
    # Trim to available genes only
    PROGRAMS[program] = found

# ── Score programs ─────────────────────────────────────────────────────────
print("\nScoring programs …")
for name, genes in PROGRAMS.items():
    if len(genes) == 0:
        print(f"  SKIP {name} — no genes available")
        adata.obs[name] = 0.0
        continue
    sc.tl.score_genes(adata, genes, score_name=name)
    print(f"  {name}: scored with {len(genes)} genes")

# ── Summary statistics ─────────────────────────────────────────────────────
print("\nProgram score summary:")
cols = list(PROGRAMS.keys())
summary = adata.obs[cols].describe().round(4)
print(summary.to_string())

# ── Q75 threshold preview (used by build_interface_mask.py) ───────────────
print("\nQ75 thresholds (interface mask inputs):")
for col in cols:
    q75 = adata.obs[col].quantile(0.75)
    print(f"  {col}: Q75 = {q75:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
print(f"\nSaving → {OUTFILE}")
adata.write(OUTFILE)
print("Done.")
