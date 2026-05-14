from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

INFILE = Path("results/cabonivo/cabonivo_hodge_hotspots.csv")
OUTFILE = Path("results/cabonivo/cabonivo_kts_states.csv")

df = pd.read_csv(INFILE)

# --------------------------------------------------
# KTS state definitions
# --------------------------------------------------
# External cohort version:
# Use cohort-relative quantile thresholds because absolute energies
# differ across datasets and exhaustion signal is sparse.
#
# Variables:
#   coexact_fraction = non-gradient purity
#   spatial_score    = LCC × Gini connected concentration
#   density_p95      = hotspot amplitude
#
# Interpretation:
#   S1 = fragmented/weak
#   S2 = connected moderate
#   S3 = amplified connected front
#   S4 = residual/concentrated high-purity object
# --------------------------------------------------

cf_med = df["coexact_fraction"].median()
score_med = df["spatial_score"].median()
p95_med = df["density_p95"].median()
gini_med = df["density_gini"].median()

print("Thresholds:")
print(f"  cf_med    = {cf_med:.6f}")
print(f"  score_med = {score_med:.6f}")
print(f"  p95_med   = {p95_med:.6f}")
print(f"  gini_med  = {gini_med:.6f}")

def assign_state(row):
    cf = row["coexact_fraction"]
    score = row["spatial_score"]
    p95 = row["density_p95"]
    gini = row["density_gini"]
    lcc = row["lcc_fraction"]

    # S3: amplified connected front
    if (score >= score_med) and (p95 >= p95_med) and (lcc >= df["lcc_fraction"].median()):
        return "S3"

    # S4: high purity/concentration but not necessarily high amplitude
    if (cf >= cf_med) and (gini >= gini_med):
        return "S4"

    # S2: connected moderate organization
    if score >= score_med:
        return "S2"

    # S1: fragmented/weak operator regime
    return "S1"

df["kts_state"] = df.apply(assign_state, axis=1)
df["is_S1"] = df["kts_state"] == "S1"
df["is_organized"] = ~df["is_S1"]

df.to_csv(OUTFILE, index=False)

print("\nKTS state table:")
print(
    df[
        [
            "sample_id",
            "response",
            "coexact_fraction",
            "density_gini",
            "lcc_fraction",
            "spatial_score",
            "density_p95",
            "kts_state",
        ]
    ].to_string(index=False)
)

print("\nState counts:")
print(pd.crosstab(df["response"], df["kts_state"]))

print("\nS1 vs organized:")
tab = pd.crosstab(df["response"], df["is_S1"])
print(tab)

# Fisher test:
# rows: Responder / Non_Responder
# cols: S1 true / S1 false
if set(df["response"]) == {"Responder", "Non_Responder"}:
    r_s1 = int(((df["response"] == "Responder") & df["is_S1"]).sum())
    r_org = int(((df["response"] == "Responder") & (~df["is_S1"])).sum())
    nr_s1 = int(((df["response"] == "Non_Responder") & df["is_S1"]).sum())
    nr_org = int(((df["response"] == "Non_Responder") & (~df["is_S1"])).sum())

    table = [[r_s1, r_org], [nr_s1, nr_org]]

    _, p_two = fisher_exact(table, alternative="two-sided")
    _, p_less = fisher_exact(table, alternative="less")  # R less likely S1

    print("\nFisher table [[R_S1, R_org], [NR_S1, NR_org]]:")
    print(table)
    print(f"p(two-sided)={p_two:.4f}")
    print(f"p(R less S1 than NR)={p_less:.4f}")

print(f"\nSaved → {OUTFILE}")
