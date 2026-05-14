from pathlib import Path
import pandas as pd

HOTSPOTS = Path("results/imc/ici_pseudospot_hodge_hotspots.csv")
WEDGE = Path("results/imc/ici_wedge_summary.csv")
META = Path("data/HCC_ICICohort/imc/ICI_metadata.csv")

OUT_ROI = Path("results/imc/ici_pseudospot_hodge_hotspots_with_response.csv")
OUT_PATIENT = Path("results/imc/ici_pseudospot_patient_level_summary.csv")

try:
    hot = pd.read_csv(HOTSPOTS)
except Exception:
    print("No valid hotspot ROIs found.")
    exit()
wedge = pd.read_csv(WEDGE)
meta = pd.read_csv(META)

print("Hotspots:", hot.shape)
print("Wedge:", wedge.shape)
print("Metadata:", meta.shape)

# --------------------------------------------------
# Normalize keys
# --------------------------------------------------

hot["patient_id"] = hot["patient_id"].astype(int)
meta["patient_id"] = meta["patient_id"].astype(int)

# --------------------------------------------------
# Merge ROI-level operator features with metadata
# --------------------------------------------------

df = hot.merge(
    meta,
    on="patient_id",
    how="left",
    validate="many_to_one",
)

# Add wedge metrics
df = df.merge(
    wedge,
    on="sample_id",
    how="left",
    suffixes=("", "_wedge"),
)

# Response label
df["response_label"] = df["response"].map({
    1: "Responder",
    0: "Non_Responder",
})

# Sanity checks
missing = df["response"].isna().sum()
print("Missing response labels:", missing)

if missing > 0:
    print(df.loc[df["response"].isna(), ["sample_id", "patient_id"]])

print("\nROI-level response counts:")
print(df["response_label"].value_counts(dropna=False))

print("\nPatient-level response counts:")
print(meta["response"].map({1: "Responder", 0: "Non_Responder"}).value_counts())

# --------------------------------------------------
# Patient-level aggregation
# --------------------------------------------------
# Important: response is patient-level; ROI-level p-values are pseudo-replicated.
# Use patient-level median as primary validation.

metrics = [
    "coexact_fraction",
    "density_gini",
    "lcc_fraction",
    "spatial_score",
    "density_p95",
    "wedge_mean",
    "wedge_p95",
]

patient = (
    df.groupby("patient_id")
    .agg(
        response=("response", "first"),
        response_label=("response_label", "first"),
        best_response=("best_response", "first"),
        immunotype=("immunotype", "first"),
        biopsy_or_resectate=("biopsy_or_resectate", "first"),
        n_rois_metadata=("n_rois", "first"),
        n_rois_processed=("sample_id", "nunique"),
        **{f"{m}_median": (m, "median") for m in metrics if m in df.columns},
        **{f"{m}_max": (m, "max") for m in metrics if m in df.columns},
    )
    .reset_index()
)

OUT_ROI.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_ROI, index=False)
patient.to_csv(OUT_PATIENT, index=False)

print("\nSaved ROI-level:")
print(OUT_ROI)
print(df.shape)

print("\nSaved patient-level:")
print(OUT_PATIENT)
print(patient.shape)

print("\nPatient-level preview:")
print(patient.head().to_string(index=False))
