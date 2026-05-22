"""
Path configuration for the dms_analysis pipeline.

Every path is resolved relative to this repository (``dms_analysis/``), so the
pipeline needs no machine-specific setup — the data ships in ``data/``.
Pipeline scripts import what they need, e.g. ``from config import MOESM6, RESULTS``.

Steps 02-11 are CPU-only and depend only on the files under ``data/``.
Step 01 (``01_compute_sae_features.py``) additionally requires:
  * the EvolutionaryScale ``esm`` package and ``cookbook`` snippets importable;
  * the ``ESM_API_KEY`` environment variable set to a Forge API token.
"""

from pathlib import Path

# --- repository layout -------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # dms_analysis/
DATA = ROOT / "data"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

# --- raw inputs (shipped under data/) ----------------------------------------
MOESM5 = DATA / "41586_2023_6954_MOESM5_ESM.xlsx"       # Weng et al. TableS4 — fitness
MOESM6 = DATA / "41586_2023_6954_MOESM6_ESM.xlsx"       # Weng et al. TableS5 — inferred ddG
PDB = DATA / "6vjj.pdb"                                 # KRAS (chain A) + RAF1-RBD (chain B)
FEATURE_TABLE = DATA / "uniref90_feature_table.parquet" # 16,384 ESMC-SAE feature labels
SAE_FEATURES = DATA / "kras_sae_features.pt"            # (3741, 16384) raw SAE output of step 01
ARIAL = DATA / "arial.ttf"                              # figure font (used if present)

# --- derived artifacts (written by steps 02-04, under data/) -----------------
DMS_MATRICES = DATA / "dms_matrices"
SAE_TENSOR = DMS_MATRICES / "kras_sae_tensor.pt"        # (20, 187, 16384) per-mutant SAE
SAE_TENSOR_DIFF = DMS_MATRICES / "kras_sae_tensor_diff.pt"  # same shape, mutant - WT
WT_SAE_VECTOR = DMS_MATRICES / "kras_wt_sae_vector.pt"  # (16384,) WT pooled SAE vector
SAE_META = DMS_MATRICES / "kras_sae_tensor_meta.json"   # axis / index documentation
KRAS_ANNO = DATA / "kras_anno.csv"                      # per-residue structural annotations
