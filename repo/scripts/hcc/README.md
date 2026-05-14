# scripts/hcc — Discovery HCC Pipeline

Run in order:

```bash
python build_program_scores.py        # score 4 programmes → hcc_scored.h5ad
python build_interface_mask.py        # define interface subgraphs
python build_wedge_field.py           # wedge fields per sample
python build_hodge_decomposition.py   # full-tissue Hodge (negative control)
python build_hodge_interface.py       # interface-restricted Hodge
python build_coexact_spatial.py       # coexact density, LCC, Gini, spatial score
python build_ncg_commutators.py       # 4×4 programme commutator matrix
python build_kts_transitions.py       # KTS state classification and transitions
python build_constraint_range.py      # local vs global R² (Step 23)
python build_ks_instability.py        # KS-inspired instability proxies
python build_interface_transport.py   # persistent topology (H1 loops)
```

All scripts accept `--help`. Default data path: `data/hepatocellular_carcinoma/`.

---

## Exploratory operator-geometry extensions

These four scripts are exploratory and supplement the primary pipeline.
Run after the main pipeline is complete.

```bash
# Interaction-mode decomposition (NCG sector framing)
python build_interaction_mode_decomposition.py

# Harmonic biological manifold (mesoscopic harmonic modes)
python build_harmonic_biological_manifold.py

# Local-to-global relational consistency
python build_category_local_global.py

# Lie-inspired interaction-sector geometry
python build_lie_interaction_geometry.py
```

| Script | Claim level | New vs NCG |
|---|---|---|
| `build_interaction_mode_decomposition.py` | Interpretive extension | Sector framing of pairwise commutators |
| `build_harmonic_biological_manifold.py` | Exploratory | Mid-harmonic mesoscopic enrichment |
| `build_category_local_global.py` | Exploratory | Local signature variance |
| `build_lie_interaction_geometry.py` | Interpretive framing | TM dominance (p=0.025); Jacobi null (p=0.69) |

See `README.md` (root) → **Additional exploratory operator-geometry analyses**
for full interpretation guidance.
