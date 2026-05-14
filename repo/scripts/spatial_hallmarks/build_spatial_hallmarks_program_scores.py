#!/usr/bin/env python3

"""
Build program scores for the Spatial Hallmarks pan-cancer Visium cohort.

Input:
    data/spatial_hallmarks/<SampleName>/

Expected Visium-like files per sample:
    filtered_feature_bc_matrix.h5
    OR filtered_feature_bc_matrix/
    spatial/tissue_positions*.csv
    spatial/scalefactors_json.json

Output:
    results_spatial_hallmarks/spatial_hallmarks_scored.h5ad
    results_spatial_hallmarks/spatial_hallmarks_metadata.csv

This script is the first step for external pan-cancer validation of the
original HCC operator-regime pipeline.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import scanpy as sc

BASE = Path("data/spatial_hallmarks")
OUTDIR = Path("results_spatial_hallmarks")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_H5AD = OUTDIR / "spatial_hallmarks_scored.h5ad"
OUT_META = OUTDIR / "spatial_hallmarks_metadata.csv"

PROGRAMS = {
    "tumor": [
        "EPCAM", "KRT8", "KRT18", "KRT19", "MUC1", "TACSTD2",
        "KRT7", "KRT17", "KRT20", "ERBB2",
    ],
    "tcell": [
        "CD3D", "CD3E", "CD3G", "CD2", "TRAC", "TRBC1", "TRBC2",
        "CD8A", "CD8B", "GZMB", "NKG7", "PRF1",
    ],
    "exhaustion": [
        "PDCD1", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "TOX",
        "ENTPD1", "LAYN", "CXCL13",
    ],
    "myeloid": [
        "LYZ", "CD68", "CD163", "C1QA", "C1QB", "C1QC",
        "LST1", "FCGR3A", "MS4A7", "TYROBP", "AIF1",
    ],
    "stroma": [
        "COL1A1", "COL1A2", "DCN", "LUM", "TAGLN", "FAP",
        "POSTN", "ACTA2",
    ],
}

def infer_cancer_type(sample_name: str) -> str:
    m = re.match(r"([A-Za-z]+)", sample_name)
    if not m:
        return "Unknown"
    raw = m.group(1)
    fixes = {
        "Glioblstoma": "Glioblastoma",
    }
    return fixes.get(raw, raw)

def find_matrix(sample_dir: Path):
    candidates = [
        sample_dir / "filtered_feature_bc_matrix.h5",
        sample_dir / "filtered_feature_bc_matrix" / "matrix.mtx.gz",
        sample_dir / "filtered_feature_bc_matrix" / "matrix.mtx",
    ]
    for p in candidates:
        if p.exists():
            return p
    hits = list(sample_dir.rglob("filtered_feature_bc_matrix.h5"))
    if hits:
        return hits[0]
    hits = list(sample_dir.rglob("matrix.mtx.gz")) + list(sample_dir.rglob("matrix.mtx"))
    if hits:
        return hits[0]
    return None

def find_spatial_dir(sample_dir: Path):
    candidates = [sample_dir / "spatial"]
    for c in candidates:
        if c.exists():
            return c
    hits = [p for p in sample_dir.rglob("spatial") if p.is_dir()]
    return hits[0] if hits else None

def read_visium_sample(sample_dir: Path):
    matrix = find_matrix(sample_dir)
    spatial_dir = find_spatial_dir(sample_dir)

    if matrix is None:
        raise FileNotFoundError(f"No matrix found in {sample_dir}")

    if matrix.name.endswith(".h5"):
        ad = sc.read_10x_h5(matrix)
    else:
        # matrix.mtx is inside filtered_feature_bc_matrix/
        ad = sc.read_10x_mtx(matrix.parent, var_names="gene_symbols", cache=False)

    ad.var_names_make_unique()

    sample_id = sample_dir.name
    cancer_type = infer_cancer_type(sample_id)

    ad.obs["sample_id"] = sample_id
    ad.obs["cancer_type"] = cancer_type

    # Try to load spatial coordinates
    if spatial_dir is not None:
        pos_files = (
            list(spatial_dir.glob("tissue_positions*.csv"))
            + list(spatial_dir.glob("*positions*.csv"))
        )

        if pos_files:
            pos = pd.read_csv(pos_files[0], header=None)

            # 10x old/new formats differ. Detect by width.
            # old: barcode,in_tissue,array_row,array_col,pxl_row,pxl_col
            if pos.shape[1] >= 6:
                pos = pos.iloc[:, :6]
                pos.columns = [
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "pxl_row_in_fullres",
                    "pxl_col_in_fullres",
                ]
            else:
                raise ValueError(f"Unexpected tissue_positions format: {pos_files[0]} shape={pos.shape}")

            pos = pos.set_index("barcode")
            common = ad.obs_names.intersection(pos.index)

            if len(common) == 0:
                print(f"[WARN] No coordinate barcode overlap for {sample_id}")
                ad.obsm["spatial"] = np.zeros((ad.n_obs, 2), dtype=float)
                ad.obs["in_tissue"] = True
            else:
                ad = ad[common].copy()
                pos = pos.loc[common]
                coords = pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values.astype(float)
                ad.obsm["spatial"] = coords
                ad.obs["in_tissue"] = pos["in_tissue"].astype(int).values
                ad.obs["array_row"] = pos["array_row"].values
                ad.obs["array_col"] = pos["array_col"].values
        else:
            print(f"[WARN] No tissue_positions file for {sample_id}")
            ad.obsm["spatial"] = np.zeros((ad.n_obs, 2), dtype=float)
            ad.obs["in_tissue"] = True
    else:
        print(f"[WARN] No spatial dir for {sample_id}")
        ad.obsm["spatial"] = np.zeros((ad.n_obs, 2), dtype=float)
        ad.obs["in_tissue"] = True

    return ad

def zscore(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    return (x - mu) / sd

def add_program_scores(ad):
    genes = set(ad.var_names)

    # log-normalize per sample
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    X = ad.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    for program, markers in PROGRAMS.items():
        present = [g for g in markers if g in genes]
        ad.uns[f"{program}_markers_used"] = present

        if len(present) == 0:
            print(f"[WARN] {ad.obs['sample_id'].iloc[0]}: no markers for {program}")
            score = np.zeros(ad.n_obs)
        else:
            idx = [ad.var_names.get_loc(g) for g in present]
            score = X[:, idx].mean(axis=1)

        ad.obs[f"{program}_score_raw"] = score
        ad.obs[f"{program}_score"] = zscore(score)

    # composite useful for interface
    ad.obs["immune_score"] = zscore(
        ad.obs["tcell_score"].values
        + ad.obs["myeloid_score"].values
        + ad.obs["exhaustion_score"].values
    )

    return ad

def main():
    sample_dirs = sorted([p for p in BASE.iterdir() if p.is_dir()])

    if not sample_dirs:
        raise FileNotFoundError(f"No sample folders found in {BASE}")

    adatas = []
    meta = []

    print(f"Found {len(sample_dirs)} sample folders")

    for sd in sample_dirs:
        print(f"\nProcessing {sd.name}")

        try:
            ad = read_visium_sample(sd)
            ad = add_program_scores(ad)

            adatas.append(ad)

            meta.append({
                "sample_id": sd.name,
                "cancer_type": infer_cancer_type(sd.name),
                "n_spots": ad.n_obs,
                "n_genes": ad.n_vars,
                "tumor_markers_used": ",".join(ad.uns.get("tumor_markers_used", [])),
                "tcell_markers_used": ",".join(ad.uns.get("tcell_markers_used", [])),
                "exhaustion_markers_used": ",".join(ad.uns.get("exhaustion_markers_used", [])),
                "myeloid_markers_used": ",".join(ad.uns.get("myeloid_markers_used", [])),
                "stroma_markers_used": ",".join(ad.uns.get("stroma_markers_used", [])),
            })

            print(
                f"  OK n_spots={ad.n_obs} n_genes={ad.n_vars} "
                f"cancer={infer_cancer_type(sd.name)}"
            )

        except Exception as e:
            print(f"  FAILED {sd.name}: {e}")

    if not adatas:
        raise RuntimeError("No samples successfully processed")

    combined = sc.concat(
        adatas,
        join="outer",
        label="batch",
        keys=[a.obs["sample_id"].iloc[0] for a in adatas],
        index_unique="-",
    )

    pd.DataFrame(meta).to_csv(OUT_META, index=False)
    combined.write_h5ad(OUT_H5AD)

    print("\nSaved:")
    print(f"  {OUT_H5AD}")
    print(f"  {OUT_META}")

    print("\nCohort summary:")
    print(pd.DataFrame(meta).groupby("cancer_type")["sample_id"].count())

if __name__ == "__main__":
    main()
