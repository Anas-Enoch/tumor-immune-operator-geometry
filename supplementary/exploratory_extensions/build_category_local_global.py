#!/usr/bin/env python3

"""
Yoneda/Grothendieck-inspired local-to-global analysis for HCC interfaces.

This is NOT a literal proof/use of Yoneda lemma.
This is a computational analogue:

Yoneda-inspired question:
    Is each interface state identifiable by its relational profile
    to local neighborhoods?

Grothendieck/sheaf-inspired question:
    Do local neighborhood operator signatures glue consistently into
    a coherent global interface regime?

Outputs:
    results_hcc_category_local_global.csv

Main metrics:
    local_to_global_consistency
    neighborhood_signature_variance
    local_regime_entropy
    restriction_stability
    relational_identifiability_score

Interpretation:
    high consistency + low entropy -> local patches glue into coherent regime
    low consistency + high entropy -> fragmented / incompatible local patches
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity


INFILE = Path("data/hepatocellular_carcinoma/hcc_scored.h5ad")
KTS = Path("results_hcc_kts_states.csv")
OUT = Path("results_hcc_category_local_global.csv")

K = 6
PATCH_RADIUS = 1
MIN_INTERFACE = 20
RIDGE = 1e-8
EPS = 1e-12


adata = sc.read_h5ad(INFILE)
kts = pd.read_csv(KTS) if KTS.exists() else None

required = [
    "sample_id",
    "tumor_score",
    "tcell_score",
    "exhaustion_score",
    "myeloid_score",
]

for c in required:
    if c not in adata.obs.columns:
        raise ValueError(f"Missing required column: {c}")

if "spatial" not in adata.obsm:
    raise ValueError("Missing adata.obsm['spatial']")


if "response" not in adata.obs.columns:
    RESPONSE = {
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

    adata.obs["response"] = adata.obs["sample_id"].map(RESPONSE)

    missing = adata.obs.loc[adata.obs["response"].isna(), "sample_id"].unique().tolist()
    if missing:
        raise ValueError(f"Response map missing sample IDs: {missing}")

    print("Attached response labels from hard-coded discovery map")


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or np.sum(x) <= 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    return float((2 * np.sum(np.arange(1, n + 1) * x) / (n * np.sum(x))) - (n + 1) / n)


def entropy_from_counts(labels):
    labels = [x for x in labels if x is not None]
    if not labels:
        return np.nan
    c = Counter(labels)
    p = np.array(list(c.values()), dtype=float)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p + EPS)))


def build_graph(coords, k=6):
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]

    nbrs = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        algorithm="kd_tree"
    ).fit(coords)

    _, knn = nbrs.kneighbors(coords)

    directed = []
    undirected = set()
    adj = [set() for _ in range(n)]

    for i in range(n):
        for j in knn[i, 1:]:
            j = int(j)
            if i == j:
                continue

            directed.append((i, j))
            a, b = sorted((i, j))
            undirected.add((a, b))
            adj[i].add(j)
            adj[j].add(i)

    return directed, sorted(undirected), [sorted(x) for x in adj]


def hodge_density(A, B, directed_edges, n):
    m = len(directed_edges)
    omega = np.zeros(m)

    for e, (i, j) in enumerate(directed_edges):
        omega[e] = A[i] * B[j] - A[j] * B[i]

    rows, cols, vals = [], [], []

    for e, (i, j) in enumerate(directed_edges):
        rows.extend([e, e])
        cols.extend([i, j])
        vals.extend([-1.0, 1.0])

    B1 = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n))
    L0 = B1.T @ B1 + RIDGE * sparse.eye(n, format="csr")
    rhs = B1.T @ omega

    alpha = lsqr(L0, rhs, atol=1e-8, btol=1e-8, iter_lim=1000)[0]

    omega_exact = B1 @ alpha
    omega_coexact = omega - omega_exact

    total_E = float(np.sum(omega ** 2))
    coexact_E = float(np.sum(omega_coexact ** 2))
    coexact_fraction = coexact_E / (total_E + EPS)

    density_sum = np.zeros(n)
    degree = np.zeros(n)
    abs_c = np.abs(omega_coexact)

    for val, (i, j) in zip(abs_c, directed_edges):
        density_sum[i] += val
        density_sum[j] += val
        degree[i] += 1
        degree[j] += 1

    density = density_sum / np.maximum(degree, 1)

    return density, coexact_fraction


def patch_nodes(center, adj, radius=1):
    current = {center}
    visited = {center}

    for _ in range(radius):
        nxt = set()
        for u in current:
            nxt.update(adj[u])
        nxt = nxt - visited
        visited |= nxt
        current = nxt

    return sorted(visited)


def local_patch_signature(nodes, tumor, tcell, exhaustion, myeloid, density, adj):
    nodes = np.array(nodes, dtype=int)

    if len(nodes) < 2:
        return None

    vals = {
        "tumor": tumor[nodes],
        "tcell": tcell[nodes],
        "exhaustion": exhaustion[nodes],
        "myeloid": myeloid[nodes],
        "density": density[nodes],
    }

    # local commutator-like pairwise antisymmetry inside patch
    def pair_energy(a, b):
        e = 0.0
        count = 0
        node_set = set(nodes.tolist())

        for i in nodes:
            for j in adj[i]:
                if j in node_set and i < j:
                    e += (a[i] * b[j] - a[j] * b[i]) ** 2
                    count += 1

        return np.sqrt(e / max(count, 1))

    sig = np.array([
        np.mean(vals["tumor"]),
        np.mean(vals["tcell"]),
        np.mean(vals["exhaustion"]),
        np.mean(vals["myeloid"]),
        np.mean(vals["density"]),
        np.std(vals["density"]),
        gini(vals["density"]),
        pair_energy(tumor, exhaustion),
        pair_energy(tcell, exhaustion),
        pair_energy(tcell, myeloid),
        pair_energy(exhaustion, myeloid),
    ], dtype=float)

    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
    return sig


def classify_patch(sig, all_sigs):
    """
    Local regime labels from patch signature quantiles.

    This is a computational analogue of local sections/restrictions,
    not a replacement for the KTS state classifier.
    """
    density_mean = sig[4]
    density_gini = sig[6]
    te = sig[7]
    ie = sig[8]
    im = sig[9]
    em = sig[10]
    immune_alg = ie + im + em

    q = np.nanquantile(all_sigs, [0.33, 0.66], axis=0)

    high_density = density_mean >= q[1, 4]
    high_gini = density_gini >= q[1, 6]
    high_te = te >= q[1, 7]
    high_immune = immune_alg >= np.nanquantile(all_sigs[:, 8] + all_sigs[:, 9] + all_sigs[:, 10], 0.66)

    if high_density and high_gini and high_immune:
        return "local_front"
    if high_density and high_te and not high_immune:
        return "tumor_dominated"
    if not high_density and not high_immune:
        return "weak_fragment"
    return "mixed"


def signature_variance(sigs):
    if sigs.shape[0] < 2:
        return np.nan
    return float(np.mean(np.var(sigs, axis=0)))


records = []

print("\nProcessing category-inspired local-to-global consistency\n")

for sid in sorted(adata.obs["sample_id"].unique()):

    idx = np.where(adata.obs["sample_id"].values == sid)[0]
    obs = adata.obs.iloc[idx].copy()

    if "is_interface" in obs.columns:
        iface_mask = obs["is_interface"].values.astype(bool)
    else:
        tq = obs["tumor_score"].quantile(0.75)
        iq = obs["tcell_score"].quantile(0.75)
        iface_mask = (
            (obs["tumor_score"].values > tq)
            &
            (obs["tcell_score"].values > iq)
        )

    iface_idx = idx[iface_mask]
    n = len(iface_idx)

    if n < MIN_INTERFACE:
        print(f"[{sid}] SMALL INTERFACE n={n}")
        continue

    sub = adata.obs.iloc[iface_idx]
    coords = np.asarray(adata.obsm["spatial"][iface_idx], dtype=float)

    tumor = sub["tumor_score"].values.astype(float)
    tcell = sub["tcell_score"].values.astype(float)
    exhaustion = sub["exhaustion_score"].values.astype(float)
    myeloid = sub["myeloid_score"].values.astype(float)

    directed, undirected, adj = build_graph(coords, k=K)

    density, cf = hodge_density(tumor, exhaustion, directed, n)

    patch_sigs = []
    valid_centers = []

    for i in range(n):
        nodes = patch_nodes(i, adj, radius=PATCH_RADIUS)
        sig = local_patch_signature(
            nodes, tumor, tcell, exhaustion, myeloid, density, adj
        )
        if sig is not None:
            patch_sigs.append(sig)
            valid_centers.append(i)

    if len(patch_sigs) < 5:
        print(f"[{sid}] too few patch signatures")
        continue

    patch_sigs = np.vstack(patch_sigs)

    # Global signature = mean relational profile of all local patches.
    global_sig = np.mean(patch_sigs, axis=0, keepdims=True)

    sim = cosine_similarity(patch_sigs, global_sig).ravel()

    local_to_global_consistency = float(np.mean(sim))
    restriction_stability = float(np.quantile(sim, 0.25))
    sig_var = signature_variance(patch_sigs)

    labels = [classify_patch(sig, patch_sigs) for sig in patch_sigs]
    local_entropy = entropy_from_counts(labels)

    counts = Counter(labels)
    dominant_label, dominant_count = counts.most_common(1)[0]
    dominant_fraction = dominant_count / len(labels)

    # Yoneda-inspired relational identifiability:
    # how sharply the sample is represented by one relational local-patch type
    # while maintaining high local-global consistency.
    relational_identifiability_score = float(
        local_to_global_consistency * dominant_fraction / (1.0 + local_entropy)
    )

    rec = {
        "sample_id": sid,
        "response": sub["response"].iloc[0],
        "n_interface": n,
        "n_patches": len(patch_sigs),
        "coexact_fraction": cf,
        "density_gini": gini(density),
        "density_p95": float(np.quantile(density, 0.95)),

        "local_to_global_consistency": local_to_global_consistency,
        "restriction_stability_q25": restriction_stability,
        "neighborhood_signature_variance": sig_var,
        "local_regime_entropy": local_entropy,
        "dominant_local_regime": dominant_label,
        "dominant_local_regime_fraction": dominant_fraction,
        "relational_identifiability_score": relational_identifiability_score,

        "frac_local_front": counts.get("local_front", 0) / len(labels),
        "frac_tumor_dominated": counts.get("tumor_dominated", 0) / len(labels),
        "frac_weak_fragment": counts.get("weak_fragment", 0) / len(labels),
        "frac_mixed": counts.get("mixed", 0) / len(labels),
    }

    records.append(rec)

    print(
        f"[{sid:18s}] {rec['response']:14s} "
        f"n={n:4d} "
        f"cons={local_to_global_consistency:.3f} "
        f"entropy={local_entropy:.3f} "
        f"dom={dominant_label}:{dominant_fraction:.2f} "
        f"RIS={relational_identifiability_score:.3f}"
    )


out = pd.DataFrame(records)

if kts is not None and "sample_id" in kts.columns:
    keep = [c for c in ["sample_id", "kts_state", "spatial_score", "lcc_fraction"] if c in kts.columns]
    if len(keep) > 1:
        out = out.merge(kts[keep], on="sample_id", how="left")

out.to_csv(OUT, index=False)

print(f"\nSaved → {OUT}")

metrics = [
    "local_to_global_consistency",
    "restriction_stability_q25",
    "neighborhood_signature_variance",
    "local_regime_entropy",
    "dominant_local_regime_fraction",
    "relational_identifiability_score",
    "frac_local_front",
    "frac_tumor_dominated",
    "frac_weak_fragment",
    "coexact_fraction",
]

print("\nGroup medians:")
print(out.groupby("response")[metrics].median(numeric_only=True))

if "kts_state" in out.columns:
    print("\nKTS-state medians:")
    print(out.groupby("kts_state")[metrics].median(numeric_only=True))

print("\nResponder vs NonResponder tests:")
R = out[out["response"] == "Responder"]
NR = out[out["response"] == "Non_Responder"]

for m in metrics:
    x = R[m].dropna()
    y = NR[m].dropna()
    if len(x) and len(y):
        _, pg = mannwhitneyu(x, y, alternative="greater")
        _, pl = mannwhitneyu(x, y, alternative="less")
        print(
            f"{m:38s} "
            f"R={x.median():.4f} NR={y.median():.4f} "
            f"p(R>NR)={pg:.4f} p(R<NR)={pl:.4f}"
        )

for m in ["relational_identifiability_score", "local_to_global_consistency", "local_regime_entropy"]:
    rho, p = spearmanr(out[m], out["coexact_fraction"], nan_policy="omit")
    print(f"\nSpearman {m} vs coexact_fraction: rho={rho:.3f}, p={p:.4f}")

print("\nInterpretation guide:")
print("  local_to_global_consistency high -> local patches glue coherently into global regime")
print("  local_regime_entropy high        -> heterogeneous/incompatible patch types")
print("  relational_identifiability high  -> interface identity is stable from local relational profiles")
print("  This is a category-inspired local-to-global diagnostic, not literal Yoneda/Grothendieck theory.")
