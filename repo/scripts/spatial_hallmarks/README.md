# scripts/spatial_hallmarks — Pan-Cancer Visium Validation

**Cohort:** 26 Visium sections, 118,526 spots, 6 cancer types
(breast, colorectal, glioblastoma, lung, ovarian, prostate).
No therapy-response annotations — tests *generality* of the operator
framework, not response prediction.

## Run order

```bash
python build_spatial_hallmarks_program_scores.py   # score 4 programmes
python build_spatial_hallmarks_interface_mask.py   # tumor-adj-immune interface
python build_spatial_hallmarks_hodge_interface.py  # Hodge coexact enrichment
python build_spatial_hallmarks_ncg_commutators.py  # 6-pair commutator matrix
python build_spatial_hallmarks_kts_states.py       # KTS-like state assignment
python build_spatial_hallmarks_ks_instability.py   # graph-KS operator
python build_spatial_hallmarks_lie_interaction_geometry.py
python build_spatial_hallmarks_bivector_circulation.py
```

## Key results

| Layer | Result | p-value |
|---|---|---|
| Coexact enrichment | 26/26 interface > tumour-core | p = 1.49×10⁻⁸ |
| Coexact enrichment by cancer | 4/4–7/7 in every type | — |
| Median enrichment ratio | 1.703 | — |
| Median coexact fraction | 0.253 | — |
| NCG: immune-sector > TM | 26/26 | — |
| NCG: median immune-sector fraction | 0.513 (vs HCC TM > 85%) | — |
| Graph-KS instability KS_R > 1 | 26/26 | p = 1.49×10⁻⁸ |
| KS_R cancer-type variation | — | Kruskal p = 0.0077 |
| Lie immune-sector variation | — | Kruskal p = 0.0063 |
| Lie Jacobi inconsistency | not significant | p = 0.386 |
| Bivector magnitude enriched | 24/26 | p = 5.2×10⁻⁶ |
| Bivector signed orientation | not directional | bias ≈ 0.039 |

## Important nuances

**NCG sector composition differs from HCC:**
In HCC (therapy cohort), TM > 85% of commutator energy — TM collapse
distinguishes non-responders. In the pan-cancer untreated cohort, the
immune sector dominates (median 0.513) and TM is minority (0.148).
This is expected and consistent: interface algebra is conserved as a
*framework*; its sector composition reflects cancer context and
therapeutic state.

**Graph-KS is an operator proxy:**
`KS_R > 1` means the interface coexact field shows higher
instability-like graph-operator activity than tumour core.
It is **not** a literal PDE dynamics claim. Longitudinal HCC PDE
identification did not support a KS-class PDE, so this remains
framed as graph-operator instability activity.

**Bivector circulation is supplementary:**
Magnitude enrichment (24/26) supports the interface geometry finding.
Signed orientation is near zero — no global rotational direction.
Report only magnitude; do not claim directionality.
