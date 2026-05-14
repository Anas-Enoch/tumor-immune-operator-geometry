# scripts/figures — Figure Generation

```bash
python build_figures.py          # Figures 1–7 (300 dpi PNG) → figures/
python build_supplement_figs.py  # Supplementary figures → supplementary/figures/
```

`fig_globals.py` exports all shared style constants, colour semantics,
and helper functions. Import at the top of any figure script.
