# Operator-Regime Analysis of Tumour–Immune Interfaces in HCC

**Paper:** Locally constrained immune-interaction fronts at tumour–immune interfaces
identify operator-regime states associated with immunotherapy response in
hepatocellular carcinoma

**Author:** Anas Enoch, MD — Mohammed VI University of Health Sciences (UM6SS),
Casablanca, Morocco — anas_nour@um5.ac.ma

---

## Overview

This repository contains all code for the operator-regime framework applied
to hepatocellular carcinoma (HCC) spatial transcriptomics.
The framework decomposes tumour–immune interaction fields into gradient (exact)
and non-gradient (coexact) components using discrete Hodge operators, then
characterises interface organisation through four analytical layers:

| Layer | Script prefix | Key output |
|---|---|---|
| **Hodge geometry** | `build_hodge_*` | Coexact fraction, LCC, spatial score |
| **NCG commutator algebra** | `build_ncg_*` | 4×4 commutator matrix, operator entropy |
| **KTS regime transitions** | `build_kts_*` | S0–S4 state per section, pre/post transitions |
| **Finite-range analysis** | `build_constraint_*` | Local R², global R², predictability gap |

External validation adds:

| Cohort | Script prefix | Dataset |
|---|---|---|
| CABO/NIVO | `build_cabonivo_*` | GSE238264 (GEO) |
| IMC scale sweep | `build_imc_*` | Zenodo 10.5281/zenodo.7624874 |
| **Spatial Hallmarks** | `build_spatial_hallmarks_*` | Pan-cancer Visium (26 sections, 6 cancer types) |

A dedicated baseline benchmarking analysis (`build_baseline_benchmarking.py`)
compares the operator-derived coexact metrics against six simpler abundance
and proximity baselines (immune fraction, T-cell/tumour ratio, CD8 density,
nearest-neighbour overlap, Moran's I, Ripley's K LCC proxy), demonstrating
that coexact metrics outperform all baselines while retaining
abundance-independent information (partial Spearman ρ = 0.535, p = 0.040
after controlling for immune fraction).

---

## Repository Structure

```
operator-regime-hcc/
│
├── README.md                       ← This file
├── README_FIGURE_CAPTIONS.md       ← Full figure legends for all 8 main figures
├── requirements.txt                ← Python package dependencies (pip)
├── environment.yml                 ← Conda environment specification
├── data_availability.md            ← Dataset access and accession numbers
│
├── scripts/
│   ├── hcc/                        ← Discovery HCC cohort pipeline
│   │   ├── build_program_scores.py
│   │   ├── build_interface_mask.py
│   │   ├── build_wedge_field.py
│   │   ├── build_hodge_decomposition.py
│   │   ├── build_hodge_interface.py
│   │   ├── build_coexact_spatial.py
│   │   ├── build_ncg_commutators.py
│   │   ├── build_kts_transitions.py
│   │   ├── build_constraint_range.py
│   │   ├── build_baseline_benchmarking.py  ← NEW: AUC/ROC vs simpler baselines
│   │   ├── build_ks_instability.py
│   │   ├── build_interface_transport.py
│   │   ├── build_interaction_mode_decomposition.py  ← exploratory
│   │   ├── build_harmonic_biological_manifold.py     ← exploratory
│   │   ├── build_category_local_global.py            ← exploratory
│   │   └── build_lie_interaction_geometry.py         ← exploratory
│   │
│   ├── cabo_nivo/                  ← External CABO/NIVO replication (GSE238264)
│   │   ├── build_cabonivo_h5ad.py
│   │   ├── build_cabonivo_program_scores.py
│   │   ├── build_cabonivo_interface.py
│   │   ├── build_cabonivo_hodge_hotspots.py
│   │   ├── build_cabonivo_ncg_commutators.py
│   │   ├── build_cabonivo_kts_states.py
│   │   └── build_cabonivo_step23.py
│   │
│   ├── imc/                        ← IMC cross-modality stress test
│   │   ├── extract_imc_cell_table.py
│   │   ├── build_imc_programs.py
│   │   ├── build_imc_wedge.py
│   │   ├── build_imc_pseudospots.py
│   │   ├── build_imc_pseudospot_hodge_hotspots.py
│   │   ├── merge_imc_pseudospot_response.py
│   │   ├── analyze_scale_transition.py
│   │   └── run_imc_scale_sweep.sh
│   │
│   ├── spatial_hallmarks/          ← Pan-cancer Visium validation
│   │   ├── build_spatial_hallmarks_program_scores.py
│   │   ├── build_spatial_hallmarks_interface_mask.py
│   │   ├── build_spatial_hallmarks_hodge_interface.py
│   │   ├── build_spatial_hallmarks_ncg_commutators.py
│   │   ├── build_spatial_hallmarks_kts_states.py
│   │   ├── build_spatial_hallmarks_lie_interaction_geometry.py
│   │   ├── build_spatial_hallmarks_ks_instability.py
│   │   └── build_spatial_hallmarks_bivector_circulation.py
│   │
│   └── figures/                    ← Figure generation
│       ├── fig_globals.py          ← Shared style constants, helpers
│       ├── build_figures.py        ← Figures 1–8 (main manuscript)
│       └── build_supplement_figs.py
│
├── results/                        ← Analysis output CSVs
│   ├── hcc/
│   │   ├── results_hcc_hodge_summary.csv
│   │   ├── results_hcc_hodge_interface_summary.csv
│   │   ├── results_hcc_coexact_hotspots.csv
│   │   ├── results_hcc_ncg_commutators.csv
│   │   ├── results_hcc_kts_states.csv
│   │   ├── results_hcc_kts_transitions.csv
│   │   ├── results_hcc_constraint_range.csv
│   │   ├── results_hcc_baseline_benchmarking.csv  ← NEW: AUC vs baselines
│   │   ├── results_hcc_interaction_mode_summary.csv    ← exploratory
│   │   ├── results_hcc_harmonic_biological_manifold.csv ← exploratory
│   │   ├── results_hcc_category_local_global.csv        ← exploratory
│   │   └── results_hcc_lie_interaction_geometry_summary.csv ← exploratory
│   │
│   ├── spatial_hallmarks/
│   │   ├── spatial_hallmarks_hodge_interface.csv
│   │   ├── spatial_hallmarks_ncg_commutators.csv
│   │   ├── spatial_hallmarks_kts_states.csv
│   │   ├── spatial_hallmarks_ks_instability_summary.csv
│   │   ├── spatial_hallmarks_lie_interaction_geometry_summary.csv
│   │   └── spatial_hallmarks_bivector_circulation.csv
│   │
│   └── cabo_nivo/
│       ├── cabonivo_hodge_hotspots.csv
│       ├── cabonivo_ncg_commutators.csv
│       ├── cabonivo_kts_states.csv
│       └── cabonivo_step23_local_global.csv
│
├── figures/                        ← Publication figures (200 dpi PNG)
│   ├── fig1_hcc_baseline_benchmarking.png  ← NEW: AUC/ROC comparison
│   ├── fig2_conceptual.png
│   ├── fig3_hodge_geometry.png
│   ├── fig4_ncg_algebra.png
│   ├── fig5_kts_transitions.png
│   ├── fig6_finite_range.png
│   ├── fig7_external_validation.png
│   ├── fig8_imc_scale.png
│   └── figS1_spatial_hallmarks.png         ← Supplementary Figure S1
│
└── supplementary/
    ├── exploratory_extensions/
    ├── figures/
    ├── supp_spectral_entropy.py
    ├── supp_robustness_k_sweep.py
    ├── supp_interface_threshold.py
    ├── supp_adf_isolation_index.py
    └── supp_persistence_topology.py
```

---

## Analysis Pipeline

```
Baseline benchmarking (simpler metrics)
→ Hodge coexact organisation
→ immune–immune commutator algebra
→ KTS regime structure
→ finite-range local/global analysis
→ external CABO/NIVO replication
→ IMC scale-emergence stress test
→ pan-cancer Spatial Hallmarks validation
```

---

## Quick Start

### 1  Install dependencies

```bash
# Conda (recommended)
conda env create -f environment.yml
conda activate operator-regime-hcc

# Or pip
pip install -r requirements.txt
```

### 2  Prepare data

```bash
# Discovery HCC cohort — place raw Visium files in:
data/hepatocellular_carcinoma/
    hcc_scored.h5ad          # (generated by build_program_scores.py)

# CABO/NIVO external cohort (GSE238264)
data/GSE238264/
    HCC1R/ HCC2R/ HCC3R/ HCC4R/     # responders
    HCC5NR/ HCC6NR/ HCC7NR/         # non-responders

# IMC cohort (Zenodo 10.5281/zenodo.7624874)
data/HCC_ICICohort/
    ICI_img_normalized/
    ICI_cellmasks/
    ICI_stromamasks/
    ICICohort_panel.xlsx
```

### 3  Run discovery pipeline

```bash
cd scripts/hcc

python build_program_scores.py \
    --input data/hepatocellular_carcinoma/raw/ \
    --output data/hepatocellular_carcinoma/hcc_scored.h5ad

python build_interface_mask.py --adata data/.../hcc_scored.h5ad
python build_wedge_field.py
python build_hodge_decomposition.py
python build_hodge_interface.py --min-interface 20
python build_coexact_spatial.py
python build_ncg_commutators.py \
    --adata data/hepatocellular_carcinoma/hcc_scored.h5ad \
    --kts-csv results/hcc/results_hcc_kts_states.csv
python build_kts_transitions.py
python build_constraint_range.py \
    --kts-csv results/hcc/results_hcc_kts_states.csv \
    --h5ad-dir data/hepatocellular_carcinoma
```

### 4  Run baseline benchmarking (NEW)

```bash
python build_baseline_benchmarking.py \
    --adata   data/hepatocellular_carcinoma/hcc_scored.h5ad \
    --results results/hcc/results_hcc_hodge_interface_summary.csv \
    --kts     results/hcc/results_hcc_kts_states.csv \
    --out     results/hcc/results_hcc_baseline_benchmarking.csv \
    --fig     figures/fig1_hcc_baseline_benchmarking.png
```

### 5  Run external validation

```bash
# CABO/NIVO
cd scripts/cabo_nivo && bash run_cabonivo_pipeline.sh

# IMC scale sweep
cd scripts/imc && bash run_imc_scale_sweep.sh   # sweeps 40–240 px

# Spatial Hallmarks pan-cancer
cd scripts/spatial_hallmarks && bash run_spatial_hallmarks.sh
```

### 6  Generate all figures

```bash
cd scripts/figures
python build_figures.py          # saves fig1–8 to figures/
python build_supplement_figs.py  # saves figS1 to figures/
```

---

## Mathematical Framework

### Wedge field

For two biological programme scores $A$, $B$ on neighbouring spots $(u, v)$:

$$\omega(u,v) = A(u)B(v) - A(v)B(u)$$

This antisymmetric edge field captures the failure of programmes to align
across the spatial graph.

### Hodge decomposition

Solving $L_0 \alpha = B_1^\top \omega$ (where $L_0 = B_1^\top B_1$ is the
node Laplacian) separates:

$$\omega = \underbrace{B_1 \alpha}_{\omega_\text{exact}} + \underbrace{\omega - B_1\alpha}_{\omega_\text{coexact}}$$

**Coexact fraction** $= \|\omega_\text{coexact}\|^2 / \|\omega\|^2$ measures
the non-gradient fraction of interface interaction structure.

### Multi-field commutator

For programme pair $(p, q)$ with diagonal operators $M_p$, $M_q$ and graph
Laplacian $L$:

$$\|M_p L M_q - M_q L M_p\|_F = \sqrt{\sum_{ij} L_{ij}^2 (p_i q_j - q_i p_j)^2}$$

This equals the interface-weighted norm of the wedge field $\omega_{pq}$,
connecting the commutator algebra directly to the Hodge decomposition.

### KTS state classification

| State | Condition | Biological interpretation |
|---|---|---|
| S0 | Tumour Q75 < 0.05 | Complete response / absent interface |
| S1 | Score < 0.10, P95 < 50 | Fragmented low-organisation |
| S2 | Score ≥ 0.10, P95 < 50 | Connected-concentrated |
| S3 | P95 ≥ 50, score ≥ 0.08 | Amplified high-intensity front |
| S4 | Gini ≥ 0.60, P95 < 20 | Residual concentrated |

> **Threshold calibration note:** state boundaries were defined empirically
> on the discovery cohort and have not been pre-registered or externally
> validated. They require recalibration before application to independent
> datasets.

### Finite-range analysis

- **Local R²**: leave-one-out k-NN mean prediction of coexact density
- **Global R²**: projection onto first 5 graph Laplacian eigenmodes
- **Locally constrained interaction fronts**: high local R², negative global R²
  (Gibbs-like eigenmode overshoot at non-cluster nodes)

---

## Key Results Summary

> **Statistical note:** all discovery-cohort statistics are exploratory
> (n = 15 sections, 6R/9NR). No correction for multiple comparisons was
> applied. At Bonferroni threshold α/11 = 0.0045, NCG purity (p = 0.0016)
> clearly survives correction; IE and EM commutator statistics lie near
> the threshold and should be treated as strong exploratory signals.
> The Spatial Hallmarks pan-cancer result is robust under any correction.

### Baseline benchmarking (Figure 1)

| Metric | AUC | p (two-sided) | Type |
|---|---|---|---|
| Interface coexact energy | **0.926** | **0.005** | Operator |
| Coexact fraction | **0.907** | **0.008** | Operator |
| Immune fraction | 0.889 | 0.012 | Abundance |
| T-cell/tumour ratio | 0.870 | 0.018 | Abundance |
| CD8 density proxy | 0.852 | 0.026 | Abundance |
| Moran's I (immune) | 0.796 | 0.066 | Spatial |
| LCC (Ripley's K proxy) | 0.759 | 0.114 | Spatial |
| NN overlap (proximity) | 0.500 | 1.000 | Proximity |

Coexact fraction retains significant association after controlling for immune
fraction (partial Spearman ρ = 0.535, p = 0.040), demonstrating
abundance-independent predictive information.

### Primary operator-regime findings

| Finding | Statistic | n |
|---|---|---|
| Spatial score R > NR (discovery) | p = 0.006 | 6R vs 9NR sections |
| IE commutator R > NR | p = 0.0048, fold 7.1× | 6R vs 9NR |
| EM commutator R > NR | p = 0.0048, fold 3.9× | 6R vs 9NR |
| IM commutator R > NR | p = 0.012, fold 3.1× | 6R vs 9NR |
| Operator entropy R > NR | p = 0.050 | 6R vs 9NR |
| KTS S1 convergence NR | Fisher p = 0.015 | 11 patients |
| Local R² R > NR | p = 0.033 | 6R vs 9NR |
| IE fraction replicated (CABO/NIVO) | p = 0.029 | 4R vs 3NR |
| Operator entropy replicated | p = 0.029 | 4R vs 3NR |
| IMC emergence window | ~80px / 55µm | 42 patients, 86 ROIs |
| **Pan-cancer coexact enrichment** | **26/26, p = 1.49×10⁻⁸** | **26 sections, 6 cancer types** |

---

## HCC5NR — Mechanistic Falsification Control

Sample HCC5NR (CABO/NIVO cohort) has higher spatial score (0.504) than 3 of
4 responders but did not respond to therapy.
It demonstrates that **spatial organisation is necessary but not sufficient**
for the response phenotype.

HCC5NR lacks the two additional required properties:
1. Immune–immune commutator richness (IE = 0.018, below all responders)
2. Locally constrained front geometry (local R² = 0.657, global R² = 0.604;
   gap = 0.053 vs responder range 0.228–0.459)

The full response-associated regime requires all three: connected organisation,
immune–immune algebra, and locally constrained interaction front geometry.

---

## Pre-Registered Validation Hypotheses

For prospective validation in an independent pre-therapy Visium cohort:

1. Pre-therapy coexact fraction ≥ 0.10
2. Spatial score (LCC × Gini) ≥ 0.10
3. H1 topological loop max persistence ≤ 25

All three are computable from a single pre-therapy section and are size-agnostic.
These thresholds were defined on the discovery cohort and require external
calibration before any clinical application.

---

## Exploratory Operator-Geometry Analyses

In addition to the primary Hodge, NCG, KTS, external validation, and IMC
analyses, this repository contains exploratory operator-geometry extensions.
These analyses support interpretation and robustness; they do not replace the
primary manuscript results.

### Interaction-mode decomposition

**Script:** `scripts/hcc/build_interaction_mode_decomposition.py`  
**Output:** `results/hcc/results_hcc_interaction_mode_summary.csv`

Decomposes six pairwise commutator channels (TI, TE, TM, IE, IM, EM)
into interpretable interaction sectors, distinguishing the tumour–myeloid
backbone from the immune–immune interaction sector.

### Harmonic biological manifold analysis

**Script:** `scripts/hcc/build_harmonic_biological_manifold.py`  
**Output:** `results/hcc/results_hcc_harmonic_biological_manifold.csv`

Cautious harmonic decomposition of coexact-density profiles around the
interface centroid. **Not** a literal Fourier analysis of biological signals.
Low harmonics = globally smooth organisation; mid = finite-range mesoscopic;
high = punctate fragmentation.

### Local-to-global relational consistency

**Script:** `scripts/hcc/build_category_local_global.py`  
**Output:** `results/hcc/results_hcc_category_local_global.csv`

Tests whether local neighbourhood operator signatures compress into a
homogeneous global interface profile. Responder interfaces show lower
local-to-global consistency and higher neighbourhood signature variance,
consistent with locally heterogeneous finite-range interaction fronts.

### Lie-inspired interaction-sector geometry

**Script:** `scripts/hcc/build_lie_interaction_geometry.py`  
**Output:** `results/hcc/results_hcc_lie_interaction_geometry_summary.csv`

Reformulates the commutator layer into interaction sectors: a tumour–myeloid
backbone and an immune sector (IE + IM + EM).

> **Important:** interpret as Lie-inspired sector geometry, **not** as
> evidence that the biological system forms a true Lie algebra. The
> Jacobi-inspired inconsistency proxy was not significant (p = 0.69),
> appropriately limiting the claim to sector-level interaction strength
> rather than formal Lie-algebraic closure (see Supplementary Note S1).

| Analysis | Status |
|---|---|
| Interaction-mode decomposition | Useful interpretive extension of NCG |
| Harmonic biological manifold | Exploratory support for finite-range organisation |
| Local-to-global consistency | Support for local heterogeneity |
| Lie-inspired sector geometry | Interpretive framing; TM-dominance diagnostic significant |

---

## Citation

```bibtex
@article{enoch2025operator,
  title   = {Locally constrained immune-interaction fronts at tumour–immune
             interfaces identify operator-regime states associated with
             immunotherapy response in hepatocellular carcinoma},
  author  = {Enoch, Anas},
  journal = {(submitted)},
  year    = {2025}
}
```
