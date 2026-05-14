# scripts/imc — IMC Cross-Modality Stress Test

Run in order:

```bash
python extract_imc_cell_table.py              # extract cell features from TIFF/masks
python build_imc_programs.py                  # score programmes from protein intensities
python build_imc_pseudospots.py               # coarse-grain to pseudospots (B px)
python build_imc_pseudospot_hodge_hotspots.py # Hodge + hotspot analysis
python merge_imc_pseudospot_response.py       # merge with clinical metadata
bash run_imc_scale_sweep.sh                   # full sweep B ∈ {40,60,80,100,120,160,240}
```

Scale sweep runtime: ~30 min (86 ROIs × 7 scales).
