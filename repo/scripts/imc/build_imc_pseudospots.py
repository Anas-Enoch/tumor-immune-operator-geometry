from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

INFILE = Path("results/imc/ici_programs.csv")
OUTFILE = Path("results/imc/ici_pseudospots.csv")

BIN_SIZE = 160      # pixels; later test 50, 100, 150
MIN_CELLS = 5
K_NEIGHBORS = 6    # spatial neighbors for adjacency-defined interface

df = pd.read_csv(INFILE)

records = []

for sid, sub in df.groupby("sample_id"):
    sub = sub.copy()

    # --------------------------------------------------
    # Build pseudo-spots by spatial binning
    # --------------------------------------------------
    sub["bin_x"] = (sub["x"] // BIN_SIZE).astype(int)
    sub["bin_y"] = (sub["y"] // BIN_SIZE).astype(int)

    grouped = (
        sub.groupby(["sample_id", "patient_id", "roi_id", "bin_x", "bin_y"])
        .agg(
            n_cells=("cell_id", "size"),
            x=("x", "mean"),
            y=("y", "mean"),
            tumor_score=("tumor_score", "mean"),
            tcell_score=("tcell_score", "mean"),
            exhaustion_score=("exhaustion_score", "mean"),
            myeloid_score=("myeloid_score", "mean"),
        )
        .reset_index()
    )

    grouped = grouped[grouped["n_cells"] >= MIN_CELLS].copy()

    if len(grouped) < K_NEIGHBORS + 1:
        grouped["is_tumor_bin"] = False
        grouped["is_immune_bin"] = False
        grouped["is_interface"] = False
        records.append(grouped)
        continue

    # --------------------------------------------------
    # Per-ROI tumor and immune enrichment
    # --------------------------------------------------
    tumor_q75 = grouped["tumor_score"].quantile(0.75)
    tcell_q75 = grouped["tcell_score"].quantile(0.75)

    tumor_mask = grouped["tumor_score"].values > tumor_q75
    immune_mask = grouped["tcell_score"].values > tcell_q75

    # --------------------------------------------------
    # Adjacency-defined interface
    # --------------------------------------------------
    # IMC pseudo-spots are sparse. Unlike Visium, tumor and immune
    # signals often occupy adjacent bins rather than the same mixed bin.
    #
    # A bin is interface-like if:
    #   1. it is tumor-enriched,
    #   2. at least one kNN spatial neighbor is immune-enriched.
    #
    # This defines a geometric tumor-immune boundary.
    # --------------------------------------------------

    xy = grouped[["x", "y"]].values

    nn = NearestNeighbors(
        n_neighbors=min(K_NEIGHBORS + 1, len(grouped)),
        algorithm="kd_tree",
    )
    nn.fit(xy)

    neigh = nn.kneighbors(xy, return_distance=False)

    interface = np.zeros(len(grouped), dtype=bool)

    for i in range(len(grouped)):
        if not tumor_mask[i]:
            continue

        # exclude self at neigh[i, 0]
        nb = neigh[i, 1:]

        if immune_mask[nb].any():
            interface[i] = True

    grouped["is_tumor_bin"] = tumor_mask
    grouped["is_immune_bin"] = immune_mask
    grouped["is_interface"] = interface

    records.append(grouped)

out = pd.concat(records, ignore_index=True)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTFILE, index=False)

summary = (
    out.groupby("sample_id")
    .agg(
        n_bins=("bin_x", "size"),
        n_tumor_bins=("is_tumor_bin", "sum"),
        n_immune_bins=("is_immune_bin", "sum"),
        n_interface=("is_interface", "sum"),
    )
)

print("Saved:", OUTFILE)
print("Shape:", out.shape)
print(summary.head(20))
print("\nInterface count summary:")
print(summary["n_interface"].describe())
print("\nROIs with >=20 interface bins:", (summary["n_interface"] >= 20).sum())
