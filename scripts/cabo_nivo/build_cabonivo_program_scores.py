from pathlib import Path
import numpy as np
import scanpy as sc

INFILE = Path("data/GSE238264/cabonivo_combined.h5ad")
OUTFILE = Path("data/GSE238264/cabonivo_scored.h5ad")

adata = sc.read_h5ad(INFILE)

# Normalize/log only if raw counts-like
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

programs = {
    "tumor_score": ["GPC3", "AFP", "EPCAM", "KRT19", "ALB", "CPS1"],
    "tcell_score": ["CD3D", "CD3E", "CD8A", "TRAC", "TRBC1", "NKG7"],
    "exhaustion_score": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4"],
    "myeloid_score": ["LYZ", "C1QA", "C1QB", "C1QC", "APOE", "FCER1G", "TYROBP"],
}

print("Gene availability:")
for score, genes in programs.items():
    found = [g for g in genes if g in adata.var_names]
    missing = [g for g in genes if g not in adata.var_names]
    print(f"{score}: {len(found)}/{len(genes)} found", "missing="+str(missing) if missing else "")
    sc.tl.score_genes(
        adata,
        gene_list=found,
        score_name=score,
        use_raw=False,
    )

print("\nProgram summaries:")
for score in programs:
    vals = adata.obs[score].values
    print(
        f"{score:18s} "
        f"mean={np.mean(vals):.4f} "
        f"median={np.median(vals):.4f} "
        f"q75={np.quantile(vals,0.75):.4f} "
        f"max={np.max(vals):.4f}"
    )

adata.write_h5ad(OUTFILE)
print(f"\nSaved → {OUTFILE}")
