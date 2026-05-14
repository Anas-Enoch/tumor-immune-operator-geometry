# Supplementary Analyses

| Script | Figure | Content |
|---|---|---|
| `supp_spectral_entropy.py` | S1 | Graph spectral entropy and participation ratio by KTS state |
| `supp_robustness_k_sweep.py` | S2 | Sensitivity of coexact fraction to kNN k (k = 4,6,8,10) |
| `supp_interface_threshold.py` | S3 | Sensitivity to Q75 interface threshold (Q70–Q80) |
| `supp_adf_isolation_index.py` | S4 | ADF reframed as hotspot isolation index |
| `supp_persistence_topology.py` | S5 | H1 persistent loop analysis (patient 71: 0→45 loops) |

Run any script directly; figures saved to `supplementary/figures/`.

---

## Exploratory operator-geometry analyses

These scripts live in `scripts/hcc/` but their outputs belong to the
exploratory-support tier, not the primary evidence chain.

| Script | Output | Interpretation |
|---|---|---|
| `build_interaction_mode_decomposition.py` | `results_hcc_interaction_mode_summary.csv` | NCG sector framing; immune sector expansion |
| `build_harmonic_biological_manifold.py` | `results_hcc_harmonic_biological_manifold.csv` | Mid-harmonic mesoscopic enrichment in R |
| `build_category_local_global.py` | `results_hcc_category_local_global.csv` | Higher neighbourhood variance in R |
| `build_lie_interaction_geometry.py` | `results_hcc_lie_interaction_geometry_summary.csv` | TM dominance NR>R (p=0.025); Jacobi null (p=0.69) |

**Claim levels:**
- `build_interaction_mode_decomposition.py` and `build_category_local_global.py`:
  useful interpretive support for the primary NCG and Step-23 results.
- `build_harmonic_biological_manifold.py`: exploratory; mid-harmonic
  enrichment is suggestive but not robustly validated.
- `build_lie_interaction_geometry.py`: the TM-dominance result is new and
  reported in the manuscript; the Lie-algebra claim is explicitly bounded
  by the Jacobi null.

None of these analyses replace the primary evidence chain
(Hodge → NCG → KTS → Step 23 → CABO/NIVO replication → IMC scale test).
