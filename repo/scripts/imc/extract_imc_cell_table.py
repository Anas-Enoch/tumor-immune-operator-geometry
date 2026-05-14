import re
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from skimage.measure import regionprops_table
from tqdm import tqdm

BASE = Path("data/HCC_ICICohort")
IMG_DIR = BASE / "ICI_img_normalized"
CELL_DIR = BASE / "ICI_cellmasks"
STROMA_DIR = BASE / "ICI_stromamasks"
PANEL = BASE / "ICICohort_panel.xlsx"
OUT = Path("results/imc/ici_cell_table.csv")

panel = pd.read_excel(PANEL)
targets = panel["Target"].astype(str).tolist()

records = []
files = sorted(CELL_DIR.glob("*.tiff"))

print(f"Found {len(files)} ROI masks")

for mask_path in tqdm(files):
    sample_id = mask_path.stem

    img_path = IMG_DIR / mask_path.name
    stroma_path = STROMA_DIR / mask_path.name

    if not img_path.exists():
        print(f"Missing image: {img_path}")
        continue

    img = tiff.imread(img_path)
    mask = tiff.imread(mask_path)

    stroma = tiff.imread(stroma_path) if stroma_path.exists() else None

    m = re.match(r"ICI_(\d+)_ROI(\d+)", sample_id)
    patient_id = int(m.group(1)) if m else None
    roi_id = int(m.group(2)) if m else None

    props = regionprops_table(
        mask,
        properties=("label", "centroid", "area")
    )

    df = pd.DataFrame(props).rename(
        columns={
            "label": "cell_id",
            "centroid-0": "y",
            "centroid-1": "x",
            "area": "cell_area",
        }
    )

    df["sample_id"] = sample_id
    df["patient_id"] = patient_id
    df["roi_id"] = roi_id

    for ch, marker in enumerate(targets):
        ch_img = img[ch]

        vals = regionprops_table(
            mask,
            intensity_image=ch_img,
            properties=("label", "mean_intensity")
        )

        tmp = pd.DataFrame(vals).rename(
            columns={
                "label": "cell_id",
                "mean_intensity": marker,
            }
        )

        df = df.merge(
            tmp[["cell_id", marker]],
            on="cell_id",
            how="left"
        )

    if stroma is not None:
        yy = np.clip(df["y"].round().astype(int).values, 0, stroma.shape[0] - 1)
        xx = np.clip(df["x"].round().astype(int).values, 0, stroma.shape[1] - 1)
        df["stroma_mask_value"] = stroma[yy, xx]
    else:
        df["stroma_mask_value"] = np.nan

    records.append(df)

out = pd.concat(records, ignore_index=True)
out = out.loc[:, ~out.columns.duplicated()]

OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT, index=False)

print(f"\nSaved: {OUT}")
print("Shape:", out.shape)
print(out.head())
