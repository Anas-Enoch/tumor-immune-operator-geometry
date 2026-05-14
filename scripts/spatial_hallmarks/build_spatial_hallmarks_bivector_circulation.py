#!/usr/bin/env python3

"""
Face-level bivector / circulation analysis for Spatial Hallmarks pan-cancer Visium cohort.

Input:
    results_spatial_hallmarks/spatial_hallmarks_interface.h5ad

Output:
    results_spatial_hallmarks/spatial_hallmarks_bivector_circulation.csv
    results_spatial_hallmarks/spatial_hallmarks_bivector_face_scores.csv.gz

Purpose:
    Extend the edge-level wedge/Hodge validation to an explicit face-level
    circulation diagnostic.

Core object:
    Edge wedge:
        omega_ij = A_i B_j - A_j B_i

    Face circulation:
        C_ijk = omega_ij + omega_jk + omega_ki

Interpretation:
    abs(C_ijk) measures local oriented circulation-like organization
    on spatial triangles.

Important:
    This is an exploratory geometric layer.
    It is not required for the primary Hodge validation claim.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from scipy.spatial import Delaunay
from scipy.stats import mannwhitneyu, binomtest, kruskal


INFILE = Path("results_spatial_hallmarks/spatial_hallmarks_interface.h5ad")

OUT_SUMMARY = Path("results_spatial_hallmarks/spatial_hallmarks_bivector_circulation.csv")
OUT_FACES = Path("results_spatial_hallmarks/spatial_hallmarks_bivector_face_scores.csv.gz")

MIN_INTERFACE_FACES = 10
MIN_TUMOR_CORE_FACES = 10
EPS = 1e-12


def triangle_area(coords):
    a, b, c = coords
    v1 = b - a
    v2 = c - a
    return 0.5 * (v1[0] * v2[1] - v1[1] * v2[0])


def wedge(A, B, i, j):
    return A[i] * B[j] - A[j] * B[i]


def safe_mwu_greater(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan

    try:
        return float(mannwhitneyu(x, y, alternative="greater").pvalue)
    except Exception:
        return np.nan


def face_region(labels):
    labels = list(labels)

    n_interface = sum(x == "interface" for x in labels)
    n_tumor = sum(x == "tumor_core" for x in labels)
    n_immune = sum(x == "immune" for x in labels)

    if n_interface >= 1:
        return "interface"
    if n_tumor == 3:
        return "tumor_core"
    if n_immune == 3:
        return "immune"
    return "mixed_other"


def circular_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "signed_mean": np.nan,
            "abs_mean": np.nan,
            "signed_bias": np.nan,
            "positive_fraction": np.nan,
        }

    signed_mean = float(np.mean(x))
    abs_mean = float(np.mean(np.abs(x)))
    signed_bias = signed_mean / (abs_mean + EPS)
    positive_fraction = float(np.mean(x > 0))

    return {
        "signed_mean": signed_mean,
        "abs_mean": abs_mean,
        "signed_bias": signed_bias,
        "positive_fraction": positive_fraction,
    }


def process_sample(ad, sid):
    idx = np.where(ad.obs["sample_id"].values == sid)[0]
    obs = ad.obs.iloc[idx].copy()
    coords = np.asarray(ad.obsm["spatial"][idx], dtype=float)

    if len(idx) < 10:
        return None, None

    A = obs["tumor_score"].values.astype(float)
    B = obs["immune_score"].values.astype(float)

    labels = obs["interface_region"].astype(str).values

    # Delaunay triangulation over Visium spot coordinates
    try:
        tri = Delaunay(coords)
    except Exception as e:
        print(f"[{sid}] Delaunay failed: {e}")
        return None, None

    faces = tri.simplices

    face_records = []

    for face_id, simplex in enumerate(faces):
        simplex = list(map(int, simplex))
        pts = coords[simplex]

        area = triangle_area(pts)

        # Orient all triangles counterclockwise for consistent sign convention.
        if area < 0:
            simplex = [simplex[0], simplex[2], simplex[1]]
            pts = coords[simplex]
            area = -area

        i, j, k = simplex

        omega_ij = wedge(A, B, i, j)
        omega_jk = wedge(A, B, j, k)
        omega_ki = wedge(A, B, k, i)

        circulation = omega_ij + omega_jk + omega_ki

        # Area-normalized circulation density
        bivector_density = circulation / (area + EPS)

        reg = face_region(labels[[i, j, k]])

        face_records.append({
            "sample_id": sid,
            "cancer_type": obs["cancer_type"].iloc[0],
            "face_id": face_id,

            "i": i,
            "j": j,
            "k": k,

            "x_centroid": float(np.mean(coords[[i, j, k], 0])),
            "y_centroid": float(np.mean(coords[[i, j, k], 1])),

            "face_region": reg,
            "area": float(area),

            "circulation": float(circulation),
            "abs_circulation": float(abs(circulation)),
            "bivector_density": float(bivector_density),
            "abs_bivector_density": float(abs(bivector_density)),

            "omega_ij": float(omega_ij),
            "omega_jk": float(omega_jk),
            "omega_ki": float(omega_ki),
        })

    faces_df = pd.DataFrame(face_records)

    if faces_df.empty:
        return None, None

    interface = faces_df[faces_df["face_region"] == "interface"]
    tumor = faces_df[faces_df["face_region"] == "tumor_core"]

    n_interface = len(interface)
    n_tumor = len(tumor)

    if n_interface < MIN_INTERFACE_FACES or n_tumor < MIN_TUMOR_CORE_FACES:
        status = "LOW_FACE_N"
    else:
        status = "OK"

    interface_abs = interface["abs_bivector_density"].values
    tumor_abs = tumor["abs_bivector_density"].values

    interface_mean = float(np.nanmean(interface_abs)) if len(interface_abs) else np.nan
    tumor_mean = float(np.nanmean(tumor_abs)) if len(tumor_abs) else np.nan
    ratio = interface_mean / (tumor_mean + EPS) if np.isfinite(interface_mean) and np.isfinite(tumor_mean) else np.nan

    p = safe_mwu_greater(interface_abs, tumor_abs)

    istats = circular_stats(interface["bivector_density"].values)
    tstats = circular_stats(tumor["bivector_density"].values)

    rec = {
        "sample_id": sid,
        "cancer_type": obs["cancer_type"].iloc[0],
        "n_spots": len(idx),
        "n_faces": len(faces_df),
        "n_interface_faces": n_interface,
        "n_tumor_core_faces": n_tumor,
        "status": status,

        "interface_abs_bivector_mean": interface_mean,
        "tumor_core_abs_bivector_mean": tumor_mean,
        "interface_vs_tumor_abs_bivector_ratio": ratio,
        "log2_interface_vs_tumor_abs_bivector_ratio": float(np.log2(ratio + EPS)) if np.isfinite(ratio) else np.nan,
        "mwu_p_interface_gt_tumor": p,

        "interface_signed_mean": istats["signed_mean"],
        "interface_signed_bias": istats["signed_bias"],
        "interface_positive_fraction": istats["positive_fraction"],

        "tumor_signed_mean": tstats["signed_mean"],
        "tumor_signed_bias": tstats["signed_bias"],
        "tumor_positive_fraction": tstats["positive_fraction"],
    }

    return rec, faces_df


def main():
    ad = sc.read_h5ad(INFILE)

    required = [
        "sample_id",
        "cancer_type",
        "tumor_score",
        "immune_score",
        "interface_region",
    ]

    for c in required:
        if c not in ad.obs.columns:
            raise ValueError(f"Missing required obs column: {c}")

    if "spatial" not in ad.obsm:
        raise ValueError("Missing ad.obsm['spatial']")

    summary_records = []
    all_faces = []

    print("\nRunning Spatial Hallmarks face-level bivector circulation analysis\n")

    for sid in sorted(ad.obs["sample_id"].unique()):
        rec, faces = process_sample(ad, sid)

        if rec is None:
            continue

        summary_records.append(rec)
        all_faces.append(faces)

        verdict = "R>1" if rec["interface_vs_tumor_abs_bivector_ratio"] > 1 else "R<=1"

        print(
            f"[{sid:14s}] {rec['cancer_type']:12s} "
            f"faces={rec['n_faces']:5d} "
            f"R={rec['interface_vs_tumor_abs_bivector_ratio']:.3f} "
            f"p={rec['mwu_p_interface_gt_tumor']:.2e} "
            f"bias={rec['interface_signed_bias']:.3f} "
            f"{rec['status']} {verdict}"
        )

    summary = pd.DataFrame(summary_records)
    faces_all = pd.concat(all_faces, ignore_index=True) if all_faces else pd.DataFrame()

    summary.to_csv(OUT_SUMMARY, index=False)
    faces_all.to_csv(OUT_FACES, index=False, compression="gzip")

    print("\nSaved:")
    print(f"  {OUT_SUMMARY}")
    print(f"  {OUT_FACES}")

    valid = summary[summary["status"] == "OK"].copy()

    if len(valid):
        n = len(valid)
        n_gt = int((valid["interface_vs_tumor_abs_bivector_ratio"] > 1).sum())
        n_sig = int((valid["mwu_p_interface_gt_tumor"] < 0.05).sum())

        try:
            p_sign = binomtest(n_gt, n, p=0.5, alternative="greater").pvalue
        except Exception:
            p_sign = np.nan

        print("\nCohort summary:")
        print(f"  Valid samples: {n}")
        print(f"  R>1: {n_gt}/{n}")
        print(f"  significant p<0.05: {n_sig}/{n}")
        print(f"  sign test p: {p_sign:.3e}")
        print(f"  median R: {valid['interface_vs_tumor_abs_bivector_ratio'].median():.3f}")
        print(f"  median signed bias: {valid['interface_signed_bias'].median():.3f}")

        print("\nBy cancer type:")
        by = valid.groupby("cancer_type").agg(
            n=("sample_id", "count"),
            n_R_gt_1=("interface_vs_tumor_abs_bivector_ratio", lambda x: int((x > 1).sum())),
            median_R=("interface_vs_tumor_abs_bivector_ratio", "median"),
            median_signed_bias=("interface_signed_bias", "median"),
            median_positive_fraction=("interface_positive_fraction", "median"),
        )
        print(by)

        print("\nKruskal across cancer type:")
        groups = [
            g["interface_vs_tumor_abs_bivector_ratio"].dropna().values
            for _, g in valid.groupby("cancer_type")
        ]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                print(f"R ratio: H={stat:.3f}, p={p:.4f}")
            except Exception:
                pass

    print("\nInterpretation:")
    print("  abs_bivector ratio > 1 means interface triangles carry stronger oriented circulation magnitude than tumor-core triangles.")
    print("  signed_bias near 0 means no dominant CW/CCW orientation; high absolute signal without signed bias means bidirectional/local circulation.")
    print("  This is an exploratory face-level circulation layer, not required for the primary edge-Hodge validation.")


if __name__ == "__main__":
    main()
