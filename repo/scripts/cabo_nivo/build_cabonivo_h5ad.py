#!/usr/bin/env python3

"""
build_cabonivo_h5ad.py

Build unified AnnData object for the external CABO/NIVO HCC
Visium cohort (GSE238264).

Inputs:
    data/GSE238264/extracted/
        GSM7661255_HCC1R/HCC1R/
        GSM7661256_HCC2R/HCC2R/
        ...

Per sample directory must contain:
    filtered_feature_bc_matrix.h5
    spatial/tissue_positions_list.csv
    data_SME_*_identity.csv

Outputs:
    data/GSE238264/cabonivo_combined.h5ad

The script:
    - loads all 7 Visium samples
    - attaches spatial coordinates
    - attaches response labels
    - attaches sample metadata
    - merges into one AnnData object
"""

from pathlib import Path
import pandas as pd
import scanpy as sc
import anndata as ad
import numpy as np


# ============================================================
# CONFIG
# ============================================================

BASE = Path("data/GSE238264/extracted")

OUTPUT = Path("data/GSE238264/cabonivo_combined.h5ad")

RESPONSES = {
    "HCC1R": "Responder",
    "HCC2R": "Responder",
    "HCC3R": "Responder",
    "HCC4R": "Responder",
    "HCC5NR": "Non_Responder",
    "HCC6NR": "Non_Responder",
    "HCC7NR": "Non_Responder",
}


# ============================================================
# HELPERS
# ============================================================

def find_sample_dir(gsm_dir: Path):
    """
    Inside GSM7661255_HCC1R/
    locate HCC1R/
    """
    subdirs = [x for x in gsm_dir.iterdir() if x.is_dir()]
    if len(subdirs) != 1:
        raise RuntimeError(f"Expected one sample dir inside {gsm_dir}")
    return subdirs[0]


def load_spatial_positions(spatial_dir: Path):

    pos_file = spatial_dir / "tissue_positions_list.csv"

    cols = [
        "barcode",
        "in_tissue",
        "array_row",
        "array_col",
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
    ]

    df = pd.read_csv(
        pos_file,
        header=None,
        names=cols,
    )

    df = df.set_index("barcode")

    return df


def load_identity_csv(sample_dir: Path):

    files = list(sample_dir.glob("data_SME_*_identity.csv"))

    if len(files) != 1:
        raise RuntimeError(f"identity csv missing in {sample_dir}")

    df = pd.read_csv(files[0])

    df.columns = ["barcode", "x_pca_kmeans"]

    df = df.set_index("barcode")

    return df


# ============================================================
# MAIN
# ============================================================

all_adatas = []

gsm_dirs = sorted([x for x in BASE.iterdir() if x.is_dir()])

print("\n===================================================")
print("Building CABO/NIVO unified AnnData")
print("===================================================\n")

for gsm_dir in gsm_dirs:

    sample_dir = find_sample_dir(gsm_dir)

    sample_name = sample_dir.name

    print(f"[{sample_name}]")

    h5_path = sample_dir / "filtered_feature_bc_matrix.h5"

    spatial_dir = sample_dir / "spatial"

    # --------------------------------------------------------
    # Load Visium counts
    # --------------------------------------------------------

    adata = sc.read_10x_h5(h5_path)

    adata.var_names_make_unique()

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    adata.obs["sample_id"] = sample_name

    adata.obs["response"] = RESPONSES[sample_name]

    # --------------------------------------------------------
    # Spatial coordinates
    # --------------------------------------------------------

    pos = load_spatial_positions(spatial_dir)

    common = adata.obs_names.intersection(pos.index)

    adata = adata[common].copy()

    pos = pos.loc[common]

    adata.obs["in_tissue"] = pos["in_tissue"].values
    adata.obs["array_row"] = pos["array_row"].values
    adata.obs["array_col"] = pos["array_col"].values

    adata.obs["pxl_row"] = pos["pxl_row_in_fullres"].values
    adata.obs["pxl_col"] = pos["pxl_col_in_fullres"].values

    # spatial coordinates used by Scanpy
    adata.obsm["spatial"] = np.vstack([
        pos["pxl_col_in_fullres"].values,
        pos["pxl_row_in_fullres"].values,
    ]).T

    # --------------------------------------------------------
    # Domain labels
    # --------------------------------------------------------

    ident = load_identity_csv(sample_dir)

    common2 = adata.obs_names.intersection(ident.index)

    adata = adata[common2].copy()

    ident = ident.loc[common2]

    adata.obs["x_pca_kmeans"] = ident["x_pca_kmeans"].astype(str).values

    # --------------------------------------------------------
    # Basic QC
    # --------------------------------------------------------

    adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel()

    adata.obs["n_genes"] = np.asarray((adata.X > 0).sum(axis=1)).ravel()

    print(adata)
    print(
        f"  response={adata.obs['response'].iloc[0]}  "
        f"spots={adata.n_obs}"
    )

    all_adatas.append(adata)

    print()

# ============================================================
# CONCATENATE
# ============================================================

print("Concatenating samples ...")

adata_all = ad.concat(
    all_adatas,
    join="outer",
    label="batch",
    keys=[a.obs["sample_id"].iloc[0] for a in all_adatas],
    index_unique="-",
)

# preserve spatial
adata_all.obsm["spatial"] = np.vstack(
    [a.obsm["spatial"] for a in all_adatas]
)

print("\n===================================================")
print("FINAL OBJECT")
print("===================================================\n")

print(adata_all)

print("\nResponse counts:")
print(adata_all.obs["response"].value_counts())

print("\nSample counts:")
print(adata_all.obs["sample_id"].value_counts())

# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

adata_all.write_h5ad(OUTPUT)

print(f"\nSaved → {OUTPUT}")
