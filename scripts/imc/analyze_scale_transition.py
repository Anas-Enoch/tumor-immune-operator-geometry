from pathlib import Path
import pandas as pd
from scipy.stats import mannwhitneyu

SCALES = [60, 80, 100, 120, 140, 160]

metrics = [
    "coexact_fraction_median",
    "density_gini_median",
    "lcc_fraction_median",
    "spatial_score_median",
    "density_p95_median",
]

rows = []

for s in SCALES:

    f = Path(
        f"results/imc/ici_pseudospot_patient_level_summary_{s}.csv"
    )

    if not f.exists():
        continue

    df = pd.read_csv(f)

    R = df[df["response"] == 1]
    NR = df[df["response"] == 0]

    for m in metrics:

        if m not in df.columns:
            continue

        x = R[m].dropna()
        y = NR[m].dropna()

        if len(x) == 0 or len(y) == 0:
            continue

        _, p = mannwhitneyu(
            x, y,
            alternative="greater"
        )

        effect = x.median() - y.median()

        rows.append({
            "scale": s,
            "metric": m,
            "R_median": x.median(),
            "NR_median": y.median(),
            "effect_size": effect,
            "p": p,
            "n_R": len(x),
            "n_NR": len(y),
        })

out = pd.DataFrame(rows)

out.to_csv(
    "results/imc/imc_scale_transition_summary.csv",
    index=False
)

print(out)
