# results/hcc — Discovery Cohort Output CSVs

| File | Columns | Description |
|---|---|---|
| `results_hcc_hodge_interface_summary.csv` | sample_id, response, coexact_fraction, lcc_fraction, spatial_score, density_p95, density_gini | Per-section interface Hodge metrics |
| `results_hcc_ncg_commutators.csv` | sample_id, response, kts_state, comm_TI, comm_TE, comm_TM, comm_IE, comm_IM, comm_EM, operator_entropy | Programme commutator matrix |
| `results_hcc_kts_states.csv` | sample_id, response, kts_state, is_S1, is_organized | KTS state per section |
| `results_hcc_kts_transitions.csv` | patient_id, response, pre_state, post_state | Pre/post paired transitions |
| `results_hcc_constraint_range.csv` | sample_id, response, kts_state, r2_local, r2_global, local_minus_global | Step 23 predictability |
| `results_hcc_ks_instability.csv` | sample_id, response, cdis, roughness_cv, adf, nai | KS-inspired instability proxies |
