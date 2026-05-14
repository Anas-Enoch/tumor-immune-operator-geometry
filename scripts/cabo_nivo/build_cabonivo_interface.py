from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

INFILE = Path("data/GSE238264/cabonivo_scored.h5ad")
OUTFILE = Path("data/GSE238264/cabonivo_interface.h5ad")
SUMMARY = Path("results/cabonivo/cabonivo_interface_summary.csv")

adata = sc.read_h5ad(INFILE)

records = []

adata.obs["is_interface"] = False
adata.obs["low_tcell_signal"] = False
adata.obs["low_exhaustion_signal"] = False

for sid in sorted(adata.obs["sample_id"].unique()):
    mask = adata.obs["sample_id"] == sid
    sub = adata.obs.loc[mask]

    tumor_q75 = sub["tumor_score"].quantile(0.75)
    tcell_q75 = sub["tcell_score"].quantile(0.75)
    exhaustion_q75 = sub["exhaustion_score"].quantile(0.75)

    # Primary interface: tumor-high adjacent immune-high surrogate.
    # At Visium scale, use tumor-high AND tcell-high.
    iface = (
        (sub["tumor_score"] > tumor_q75)
        &
        (sub["tcell_score"] > tcell_q75)
    )

    adata.obs.loc[mask, "is_interface"] = iface.values

    low_tcell = tcell_q75 <= 0
    low_exh = exhaustion_q75 <= 0

    adata.obs.loc[mask, "low_tcell_signal"] = low_tcell
    adata.obs.loc[mask, "low_exhaustion_signal"] = low_exh

    rec = {
        "sample_id": sid,
        "response": sub["response"].iloc[0],
        "n_spots": len(sub),
        "n_interface": int(iface.sum()),
        "interface_fraction": float(iface.mean()),
        "tumor_q75": float(tumor_q75),
        "tcell_q75": float(tcell_q75),
        "exhaustion_q75": float(exhaustion_q75),
        "low_tcell_signal": bool(low_tcell),
        "low_exhaustion_signal": bool(low_exh),
    }

    records.append(rec)

    flag = ""
    if low_tcell:
        flag += " LOW_TCELL"
    if low_exh:
        flag += " LOW_EXHAUSTION"

    print(
        f"[{sid:6s}] {sub['response'].iloc[0]:14s} "
        f"spots={len(sub):4d} interface={iface.sum():4d} "
        f"frac={iface.mean():.4f} "
        f"tumorQ75={tumor_q75:.3f} "
        f"tcellQ75={tcell_q75:.3f} "
        f"exhQ75={exhaustion_q75:.3f}"
        f"{flag}"
    )

out = pd.DataFrame(records)
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(SUMMARY, index=False)

adata.write_h5ad(OUTFILE)

print("\nSaved:")
print(OUTFILE)
print(SUMMARY)

print("\nGroup summary:")
print(out.groupby("response")[["interface_fraction", "n_interface"]].median())
