# results/spatial_hallmarks — Pan-Cancer Visium Results

| File | Description |
|---|---|
| `spatial_hallmarks_scored.h5ad` | AnnData with programme scores (118,526 spots) |
| `spatial_hallmarks_metadata.csv` | Section-level metadata and cancer-type labels |
| `spatial_hallmarks_interface_summary.csv` | Per-section interface size and validity |
| `spatial_hallmarks_hodge_interface.csv` | Coexact enrichment ratios, fractions, sign tests |
| `spatial_hallmarks_ncg_commutators.csv` | 6-pair commutator matrix per section |
| `spatial_hallmarks_kts_states.csv` | KTS-like state per section |
| `spatial_hallmarks_ks_instability.csv` | Per-section KS_R and graph-operator metrics |
| `spatial_hallmarks_ks_instability_summary.csv` | Cancer-type aggregated KS summary |
| `spatial_hallmarks_lie_interaction_geometry.csv` | Per-section Lie-sector metrics |
| `spatial_hallmarks_lie_interaction_geometry_summary.csv` | Cancer-type summary |
| `spatial_hallmarks_bivector_circulation.csv` | Face-level bivector magnitude and orientation |

## Interpreting sector differences vs HCC

The pan-cancer cohort shows immune-sector dominance (median fraction 0.513)
whereas the HCC discovery cohort shows TM dominance (>85%). This reflects
cancer-context and therapeutic-state dependence of the interface algebra,
not a methodological inconsistency.
