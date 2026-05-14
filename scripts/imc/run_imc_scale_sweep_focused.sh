#!/bin/bash

SCALES=(60 80 100 120 140 160)

for S in "${SCALES[@]}"
do
    echo "=================================================="
    echo "BIN SIZE = $S"
    echo "=================================================="

    python3 - <<PY
from pathlib import Path
import re

p = Path("scripts/imc/build_imc_pseudospots.py")
s = p.read_text()
s = re.sub(r"BIN_SIZE = \\d+", "BIN_SIZE = $S", s)
p.write_text(s)
print("Set BIN_SIZE =", $S)
PY

    python3 scripts/imc/build_imc_pseudospots.py
    python3 scripts/imc/build_imc_pseudospot_hodge_hotspots.py
    python3 scripts/imc/merge_imc_pseudospot_response.py

    if [ -f results/imc/ici_pseudospot_hodge_hotspots.csv ]; then
        cp results/imc/ici_pseudospot_hodge_hotspots.csv \
           results/imc/ici_pseudospot_hodge_hotspots_${S}.csv
    fi

    if [ -f results/imc/ici_pseudospot_hodge_hotspots_with_response.csv ]; then
        cp results/imc/ici_pseudospot_hodge_hotspots_with_response.csv \
           results/imc/ici_pseudospot_hodge_hotspots_with_response_${S}.csv
    fi

    if [ -f results/imc/ici_pseudospot_patient_level_summary.csv ]; then
        cp results/imc/ici_pseudospot_patient_level_summary.csv \
           results/imc/ici_pseudospot_patient_level_summary_${S}.csv
    fi
done
