from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mannwhitneyu

INFILE = Path("data/GSE238264/cabonivo_interface.h5ad")
KTS = Path("results/cabonivo/cabonivo_kts_states.csv")
OUT = Path("results/cabonivo/cabonivo_ncg_commutators.csv")

K = 6
MIN_INTERFACE = 20

adata = sc.read_h5ad(INFILE)
kts = pd.read_csv(KTS)

program_cols = {
    "T": "tumor_score",
    "I": "tcell_score",
    "E": "exhaustion_score",
    "M": "myeloid_score",
}

pairs = [
    ("T", "I"),
    ("T", "E"),
    ("T", "M"),
    ("I", "E"),
    ("I", "M"),
    ("E", "M"),
]

records = []

print("Processing CABO/NIVO NCG commutators")

for sid in sorted(adata.obs["sample_id"].unique()):

    idx_sample = np.where(adata.obs["sample_id"].values == sid)[0]
    obs = adata.obs.iloc[idx_sample].copy()

    iface_local = obs["is_interface"].values.astype(bool)
    iface_idx = idx_sample[iface_local]

    n = len(iface_idx)

    if n < MIN_INTERFACE:
        print(f"[{sid}] SMALL INTERFACE n={n}")
        continue

    coords = adata.obsm["spatial"][iface_idx]

    nbrs = NearestNeighbors(
        n_neighbors=min(K + 1, n),
        algorithm="kd_tree"
    ).fit(coords)

    _, knn = nbrs.kneighbors(coords)

    edge_src = []
    edge_tgt = []

    for i in range(n):
        for j in knn[i, 1:]:
            edge_src.append(i)
            edge_tgt.append(int(j))

    edge_src = np.asarray(edge_src, dtype=int)
    edge_tgt = np.asarray(edge_tgt, dtype=int)

    vals = {
        key: adata.obs.iloc[iface_idx][col].values.astype(float)
        for key, col in program_cols.items()
    }

    rec = {
        "sample_id": sid,
        "response": obs["response"].iloc[0],
        "n_interface": n,
        "n_edges": len(edge_src),
    }

    comm_values = []

    for a, b in pairs:
        A = vals[a]
        B = vals[b]

        omega = A[edge_src] * B[edge_tgt] - A[edge_tgt] * B[edge_src]

        # Frobenius-like edge norm of commutator field
        norm = float(np.sqrt(np.sum(omega ** 2)))

        key = f"comm_{a}{b}"
        rec[key] = norm
        comm_values.append(norm)

    total = float(np.sum(comm_values))
    rec["total_comm_energy"] = total

    if total > 0:
        p = np.asarray(comm_values) / total
        p = p[p > 0]
        rec["operator_entropy"] = float(-np.sum(p * np.log(p)))
    else:
        rec["operator_entropy"] = 0.0

    dominant_idx = int(np.argmax(comm_values))
    rec["dominant_pair"] = f"comm_{pairs[dominant_idx][0]}{pairs[dominant_idx][1]}"

    # Fractions
    for (a, b), v in zip(pairs, comm_values):
        rec[f"frac_{a}{b}"] = v / (total + 1e-12)

    records.append(rec)

    print(
        f"[{sid:6s}] {rec['response']:14s} "
        f"total={total:.3f} "
        f"entropy={rec['operator_entropy']:.3f} "
        f"dominant={rec['dominant_pair']}"
    )

out = pd.DataFrame(records)

# attach KTS state
out = out.merge(
    kts[["sample_id", "kts_state", "coexact_fraction", "spatial_score"]],
    on="sample_id",
    how="left",
)

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print("\nSaved:")
print(OUT)

print("\nCommutator table:")
cols = [
    "sample_id", "response", "kts_state",
    "comm_TI", "comm_TE", "comm_TM",
    "comm_IE", "comm_IM", "comm_EM",
    "operator_entropy", "dominant_pair",
]
print(out[cols].to_string(index=False))

print("\nGroup medians:")
metrics = [
    "comm_TI", "comm_TE", "comm_TM",
    "comm_IE", "comm_IM", "comm_EM",
    "total_comm_energy", "operator_entropy",
    "frac_TI", "frac_TE", "frac_TM", "frac_IE", "frac_IM", "frac_EM",
]

print(out.groupby("response")[metrics].median())

print("\nResponder > NonResponder tests:")
R = out[out["response"] == "Responder"]
NR = out[out["response"] == "Non_Responder"]

for m in metrics:
    x = R[m].dropna()
    y = NR[m].dropna()

    if len(x) == 0 or len(y) == 0:
        continue

    _, p_greater = mannwhitneyu(x, y, alternative="greater")
    _, p_two = mannwhitneyu(x, y, alternative="two-sided")

    print(
        f"{m:20s} "
        f"R={x.median():.4f} "
        f"NR={y.median():.4f} "
        f"p(R>NR)={p_greater:.4f} "
        f"p2={p_two:.4f}"
    )

print("\nHCC5NR diagnostic:")
print(
    out[out["sample_id"] == "HCC5NR"][
        [
            "sample_id", "response", "kts_state",
            "comm_TI", "comm_TE", "comm_TM",
            "comm_IE", "comm_IM", "comm_EM",
            "operator_entropy",
            "dominant_pair",
            "frac_TM", "frac_IE", "frac_IM", "frac_EM",
        ]
    ].to_string(index=False)
)
