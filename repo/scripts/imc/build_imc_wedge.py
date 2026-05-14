from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

INFILE = Path("results/imc/ici_programs.csv")
OUTFILE = Path("results/imc/ici_wedge_summary.csv")

K = 6

df = pd.read_csv(INFILE)

records = []

samples = sorted(df["sample_id"].unique())

print(f"{len(samples)} ROIs")

for sid in samples:

    sub = df[df["sample_id"] == sid].copy()

    iface = sub[sub["is_interface"]].copy()

    n = len(iface)

    if n < 20:
        print(f"[{sid}] SMALL INTERFACE n={n}")
        continue

    xy = iface[["x", "y"]].values

    nbrs = NearestNeighbors(
        n_neighbors=min(K + 1, n),
        algorithm="kd_tree"
    ).fit(xy)

    dist, idx = nbrs.kneighbors(xy)

    A = iface["tumor_score"].values
    B = iface["exhaustion_score"].values

    wedge_vals = []

    for i in range(n):

        for j in idx[i, 1:]:

            w = A[i] * B[j] - B[i] * A[j]
            wedge_vals.append(abs(w))

    wedge_vals = np.asarray(wedge_vals)

    rec = {
        "sample_id": sid,
        "n_interface": n,
        "wedge_mean": np.mean(wedge_vals),
        "wedge_median": np.median(wedge_vals),
        "wedge_p95": np.quantile(wedge_vals, 0.95),
        "wedge_max": np.max(wedge_vals),
    }

    records.append(rec)

    print(
        f"[{sid}] "
        f"iface={n:5d} "
        f"mean={rec['wedge_mean']:.3f} "
        f"p95={rec['wedge_p95']:.3f}"
    )

out = pd.DataFrame(records)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUTFILE, index=False)

print("\nSaved:", OUTFILE)
print(out.head())
