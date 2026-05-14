# Data Availability

## Discovery HCC Cohort

**Description:** Matched pre- and post-immunotherapy Visium CytAssist spatial
transcriptomics sections from 11 HCC patients (5 responders, 6 non-responders),
yielding 22 sections, 104,418 spots, and 18,085 genes.

**Access:** Currently under controlled release. Will be deposited to GEO upon
manuscript acceptance (accession pending). Contact anas_nour@um5.ac.ma for
data sharing enquiries in the interim.

**Format:** `hcc_scored.h5ad` — AnnData object with:
- `adata.X`: normalised expression matrix
- `adata.obs`: spot metadata including `sample_id`, `patient_id`, `timepoint`,
  `response`, `tumor_score`, `tcell_score`, `exhaustion_score`, `myeloid_score`
- `adata.obsm['spatial']`: spot centroid coordinates (µm)
- `adata.uns['sample_metadata']`: patient-level clinical annotations

**Programme scoring markers:**

| Programme | Markers |
|---|---|
| Tumour | GPC3, AFP, CPS1, ALDH3A1, FGB |
| T-cell | CD3D, CD3E, CD8A |
| Exhaustion | PDCD1, CTLA4, HAVCR2, LAG3, TIGIT |
| Myeloid | LYZ, S100A8, CTSB, SPP1, FCGR3A |

---

## CABO/NIVO External Validation Cohort

**Description:** 7 Visium sections from HCC patients treated with
cabozantinib plus nivolumab (4 responders, 3 non-responders);
17,292 spots, 36,601 genes.

**Access:** Publicly available at GEO:
[GSE238264](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE238264)

**Download:**
```bash
# Using GEOparse
python -c "
import GEOparse
gse = GEOparse.get_GEO('GSE238264', destdir='data/GSE238264/')
"
```

**Directory structure after download:**
```
data/GSE238264/
    HCC1R/
        filtered_feature_bc_matrix.h5
        data_SME_*_identity.csv
        spatial/
            tissue_positions_list.csv
            scalefactors_json.json
    HCC2R/ HCC3R/ HCC4R/   # responders
    HCC5NR/ HCC6NR/ HCC7NR/ # non-responders
```

**Programme scoring markers (CABO/NIVO):**

| Programme | Markers |
|---|---|
| Tumour | EPCAM, KRT19, AFP, GPC3, ALB, CPS1 |
| T-cell | CD3D, CD3E, CD8A, TRAC, TRBC1, NKG7 |
| Exhaustion | PDCD1, HAVCR2, TIGIT, LAG3, TOX, CTLA4 |
| Myeloid | LYZ, APOE, FCER1G, TYROBP, C1QA, C1QB, C1QC |

---

## IMC Cross-Modality Cohort

**Description:** 42 HCC patients treated with immune checkpoint inhibitors,
86 ROIs, 45-channel single-cell protein imaging with cell segmentation.
456,214 QC-filtered cells (area 20–2000 px).

**Access:** Publicly available at Zenodo:
[doi: 10.5281/zenodo.7624874](https://doi.org/10.5281/zenodo.7624874)

**Download:**
```bash
wget https://zenodo.org/record/7624874/files/HCC_ICICohort.zip
unzip HCC_ICICohort.zip -d data/HCC_ICICohort/
```

**Directory structure:**
```
data/HCC_ICICohort/
    ICICohort_panel.xlsx       # 45-channel protein panel
    ICI_img_normalized/        # normalized TIFF stacks
        ICI_1_ROI1.tiff
        ...
    ICI_cellmasks/             # cell segmentation masks
        ICI_1_ROI1_cellmask.tiff
        ...
    ICI_stromamasks/           # stromal/parenchymal masks
```

**IMC protein panel (programme mapping):**

| Programme | Protein markers |
|---|---|
| Tumour | PanKeratin, E-cadherin, Arginase1, CK7 |
| T-cell | CD3, CD8a, CD4 |
| Exhaustion | PD1, Tim3, TOX, CD39, PDL1 |
| Myeloid | CD68, CD163, CD204, CD33, HLA-DR |

---

---

## Spatial Hallmarks Pan-Cancer Visium Cohort

**Description:** 26 Visium sections from six cancer types (breast,
colorectal, glioblastoma, lung, ovarian, prostate), comprising 118,526
spots. No immunotherapy response annotations. Used to test generality of
the coexact operator framework.

**Access:** Data sourced from the 10x Genomics Visium Spatial Gene Expression public datasets, the Human Cell Atlas spatial data portal, and the Spatial Hallmarks pan-cancer spatial transcriptomics collection publicly archived at Zenodo:
https://zenodo.org/records/14044964

The cohort comprised 26 Visium sections across six cancer types. Individual section accession numbers and metadata are listed in:
`results/spatial_hallmarks/spatial_hallmarks_metadata.csv`.

**Programme scoring markers (pan-cancer):**
Same four-programme panel as the HCC discovery cohort
(tumour, T-cell, exhaustion, myeloid), supplemented with stromal (COL1A1,
COL1A2, DCN, FAP) and pan-immune (PTPRC/CD45) scores for interface
definition across diverse cancer types.

**Key validation result:** Interface-localised coexact enrichment was
present in 26/26 sections across all six cancer types
(sign test p = 1.49×10⁻⁸; median enrichment ratio 1.703).


## Processed Results

Pre-computed result CSVs are available in `results/` for reproducibility
without re-running the full pipeline.

| File | Description |
|---|---|
| `results/hcc/results_hcc_hodge_interface_summary.csv` | Per-section coexact metrics |
| `results/hcc/results_hcc_ncg_commutators.csv` | 4×4 commutator matrix per section |
| `results/hcc/results_hcc_kts_states.csv` | KTS state assignments |
| `results/hcc/results_hcc_kts_transitions.csv` | Pre/post paired transitions |
| `results/hcc/results_hcc_constraint_range.csv` | Local/global R² per section |
| `results/cabo_nivo/cabonivo_hodge_hotspots.csv` | CABO/NIVO Hodge metrics |
| `results/cabo_nivo/cabonivo_ncg_commutators.csv` | CABO/NIVO commutator algebra |
| `results/cabo_nivo/cabonivo_kts_states.csv` | CABO/NIVO KTS states |
| `results/cabo_nivo/cabonivo_step23_local_global.csv` | CABO/NIVO Step 23 |
| `results/hcc/results_hcc_interaction_mode_summary.csv` | Exploratory: NCG sector decomposition |
| `results/hcc/results_hcc_harmonic_biological_manifold.csv` | Exploratory: harmonic mode analysis |
| `results/hcc/results_hcc_category_local_global.csv` | Exploratory: local-to-global consistency |
| `results/hcc/results_hcc_lie_interaction_geometry_summary.csv` | Exploratory: Lie-sector geometry |

---

## Code Availability

All analysis code is available at:
**https://github.com/Anas-Enoch/tumor-immune-operator-geometry**

Please cite the manuscript if you use this code.

---

## Ethics and Data Use

This study used de-identified public datasets processed under the ethics
approvals of their originating studies.
The discovery HCC cohort was collected under IRB approval at the originating
institution and will be shared in accordance with GEO data use policies.
No new patient data were collected. Secondary analysis of de-identified
public data was conducted in accordance with applicable regulations.
