#!/usr/bin/env python3
"""
build_wedge_field.py — HCC Antisymmetric Wedge Field
=====================================================
Computes the antisymmetric edge interaction operator (wedge field)
on a kNN spatial graph for one or more HCC samples.

Wedge operator
--------------
For each directed edge (u → v):

    F_AB(u, v) = A(u)·B(v) − A(v)·B(u)

where A = tumor_score and B = exhaustion_score.
This is the minimal antisymmetric bilinear operator on the program
fields. It is zero when programs are locally balanced and large when
one spot is tumor-dominant and its neighbor is exhaustion-dominant
(or vice versa).

Node-level summary
------------------
wedge_intensity_i = mean |F_AB(i, j)| over all edges incident to i.

This collapses the edge-level antisymmetric field to a node-level
scalar suitable for spatial plotting and downstream Hodge decomposition.

Fixes applied vs. original script
----------------------------------
1. .loc → .iloc bug:
   node_intensity.index contains integer positions (0, 1, 2, …).
   sub.obs.index contains AnnData observation labels (barcode strings).
   .loc does label-based lookup → KeyError: None of [0, 1, 2 …].
   Fixed: use .iloc with .columns.get_loc() for position-based assignment.

2. Runs all samples by default; accepts --sample-id to run one.

3. Saves per-sample wedge h5ad and a cohort-level wedge summary CSV.

Output
------
data/hcc_{sample_id}_wedge.h5ad        per-sample AnnData with wedge fields
results_hcc_wedge_summary.csv          cohort-level node wedge statistics

Usage
-----
  # All samples
  python build_wedge_field.py

  # One sample
  python build_wedge_field.py --sample-id cytassist_84_post

  # Override programs and graph k
  python build_wedge_field.py --prog-a tumor_score --prog-b exhaustion_score --k 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors


ADATA_PATH = "data/hepatocellular_carcinoma/hcc_scored.h5ad"


def build_knn_edges(coords: np.ndarray, k: int) -> pd.DataFrame:
    """Return a DataFrame of directed edges with columns source, target."""
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    _, indices = nbrs.kneighbors(coords)
    rows = []
    for i, nbr in enumerate(indices):
        for j in nbr[1:]:          # skip self (index 0)
            rows.append({"source": i, "target": int(j)})
    return pd.DataFrame(rows)


def compute_wedge(edges: pd.DataFrame,
                  A: np.ndarray,
                  B: np.ndarray) -> pd.DataFrame:
    """
    Compute F_AB(u,v) = A[u]·B[v] − A[v]·B[u] for each directed edge.
    Returns edges with a 'wedge' column added.
    """
    src = edges["source"].values
    tgt = edges["target"].values
    edges = edges.copy()
    edges["wedge"] = A[src] * B[tgt] - A[tgt] * B[src]
    return edges


def node_wedge_intensity(edges: pd.DataFrame, n: int) -> np.ndarray:
    """
    Mean |wedge| over all edges incident to each node.
    Returns an array of length n.
    """
    intensity = np.zeros(n, dtype=float)
    counts    = np.zeros(n, dtype=float)
    src = edges["source"].values
    tgt = edges["target"].values
    w   = np.abs(edges["wedge"].values)
    np.add.at(intensity, src, w)
    np.add.at(intensity, tgt, w)
    np.add.at(counts,    src, 1)
    np.add.at(counts,    tgt, 1)
    mask = counts > 0
    intensity[mask] /= counts[mask]
    return intensity


def process_sample(sub: sc.AnnData,
                   sample_id: str,
                   prog_a: str,
                   prog_b: str,
                   k: int,
                   out_dir: Path) -> dict:
    """Run the wedge pipeline for a single sample."""
    n      = sub.n_obs
    coords = sub.obsm["spatial"]

    # ── Programs ──────────────────────────────────────────────────────────
    for prog in (prog_a, prog_b):
        if prog not in sub.obs.columns:
            raise ValueError(
                f"[{sample_id}] Program '{prog}' not in .obs. "
                f"Run build_program_scores.py first."
            )

    A = sub.obs[prog_a].values.astype(float)
    B = sub.obs[prog_b].values.astype(float)

    # ── kNN graph ─────────────────────────────────────────────────────────
    edges = build_knn_edges(coords, k)

    # ── Wedge field ───────────────────────────────────────────────────────
    edges = compute_wedge(edges, A, B)

    # ── Node-level intensity ──────────────────────────────────────────────
    intensity = node_wedge_intensity(edges, n)

    # ── Assign to .obs ────────────────────────────────────────────────────
    # FIX: node indices are INTEGER POSITIONS (0, 1, …, n-1).
    # sub.obs.index contains AnnData barcode labels, not integers.
    # .loc does label-based lookup → KeyError.
    # .iloc does position-based lookup → correct.
    sub = sub.copy()
    sub.obs["wedge_intensity"]    = intensity
    sub.obs["wedge_prog_a"]       = prog_a
    sub.obs["wedge_prog_b"]       = prog_b

    # Also store the raw program values for reference
    sub.obs["wedge_A"] = A
    sub.obs["wedge_B"] = B

    # ── Save per-sample h5ad ──────────────────────────────────────────────
    out_path = out_dir / f"hcc_{sample_id}_wedge.h5ad"
    sub.write(out_path)

    # ── Summary statistics ────────────────────────────────────────────────
    stats = {
        "sample_id":             sample_id,
        "n_spots":               n,
        "n_edges":               len(edges),
        "prog_a":                prog_a,
        "prog_b":                prog_b,
        "k":                     k,
        "wedge_mean":            float(edges["wedge"].mean()),
        "wedge_std":             float(edges["wedge"].std()),
        "wedge_abs_mean":        float(edges["wedge"].abs().mean()),
        "wedge_abs_median":      float(edges["wedge"].abs().median()),
        "wedge_max":             float(edges["wedge"].max()),
        "wedge_min":             float(edges["wedge"].min()),
        "node_intensity_mean":   float(intensity.mean()),
        "node_intensity_median": float(np.median(intensity)),
        "node_intensity_max":    float(intensity.max()),
        "node_intensity_p95":    float(np.percentile(intensity, 95)),
    }

    print(
        f"  [{sample_id}]  spots={n:5d}  edges={len(edges):6d}  "
        f"wedge_abs_mean={stats['wedge_abs_mean']:.4f}  "
        f"intensity_p95={stats['node_intensity_p95']:.4f}  "
        f"→ {out_path.name}"
    )
    return stats


def main():
    ap = argparse.ArgumentParser(description="HCC wedge field computation")
    ap.add_argument("--adata",     default=ADATA_PATH,
                    help="Path to hcc_scored.h5ad")
    ap.add_argument("--sample-id", default=None,
                    help="Single sample ID; omit to run all samples")
    ap.add_argument("--prog-a",    default="tumor_score",
                    help="Program A for wedge (default: tumor_score)")
    ap.add_argument("--prog-b",    default="exhaustion_score",
                    help="Program B for wedge (default: exhaustion_score)")
    ap.add_argument("--k",         type=int, default=6,
                    help="Number of nearest neighbours for kNN graph (default: 6)")
    ap.add_argument("--out-dir",   default="data/hepatocellular_carcinoma",
                    help="Directory to write per-sample .h5ad files")
    ap.add_argument("--summary",   default="results_hcc_wedge_summary.csv",
                    help="Path for cohort-level summary CSV")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.adata} …")
    adata = sc.read_h5ad(args.adata)
    print(f"  {adata.n_obs:,} spots  programs: {args.prog_a} × {args.prog_b}  k={args.k}")

    # Select samples to process
    all_samples = sorted(adata.obs["sample_id"].unique())
    if args.sample_id:
        if args.sample_id not in all_samples:
            raise ValueError(
                f"Sample '{args.sample_id}' not found. "
                f"Available: {all_samples}"
            )
        samples = [args.sample_id]
    else:
        samples = all_samples

    print(f"\nProcessing {len(samples)} sample(s) …\n")
    all_stats = []
    for sid in samples:
        sub = adata[adata.obs["sample_id"] == sid].copy()
        stats = process_sample(
            sub, sid, args.prog_a, args.prog_b, args.k, out_dir
        )
        all_stats.append(stats)

    # ── Cohort summary ────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(args.summary, index=False)
    print(f"\nCohort summary → {args.summary}")
    print(summary_df[
        ["sample_id", "n_spots", "wedge_abs_mean", "node_intensity_median"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
