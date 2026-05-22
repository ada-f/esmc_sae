"""Step 01 — compute ESMC-6B SAE features for every KRAS DMS variant.

This is the only GPU/API step of the pipeline. For each unique (position,
mutant amino acid) pair in the DMS library it builds the full mutant sequence,
runs it (and wild-type KRAS) through the ESM Cambrian 6B Sparse Autoencoder via
the EvolutionaryScale Forge API, and saves the pooled feature matrix.

Output: data/kras_sae_features.pt  -- tensor (3741, 16384), one pooled 16,384-d
SAE vector per variant. Row 0 is wild-type; rows 1..3740 follow the row order of
``MOESM6.drop_duplicates(['Pos_real', 'mt_codon'])``. Step 02 reshapes this into
the (20, 187, 16384) DMS-layout tensor.

NOT part of the routine re-run: the API call costs Forge credits and the output
ships with the repository. Steps 02-11 consume data/kras_sae_features.pt
directly. Only run this to regenerate the SAE features from scratch.

Prerequisites (this step only):
  * the EvolutionaryScale ``esm`` package and ``cookbook`` snippets importable;
  * the ESM_API_KEY environment variable set to a Forge API token.

Usage:
  export ESM_API_KEY=<your Forge token>
  python 01_compute_sae_features.py
"""

import os

import pandas as pd
import torch

from config import MOESM6, SAE_FEATURES

# ESM Cambrian 6B + its layer-60 SAE (k=64, codebook 16,384).
ESMC_6B_MODEL = "esmc-6b-2024-12"
ESMC_6B_SAE_MODEL = "esmc-6b-2024-12_k64_codebook16384_layer60"

# Wild-type KRAS (188 aa; UniProt P01116-2, KRAS-4B isoform). The DMS construct
# skips Met1, so DMS canonical positions run 2..188.
KRAS_WT = ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY"
           "SAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTK"
           "QAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM")
assert len(KRAS_WT) == 188


def build_sequences():
    """Wild-type sequence followed by one mutant sequence per (Pos_real,
    mt_codon) pair in MOESM6. The row order matches step 02's reshape:
    row 0 = wild-type, rows 1.. = the unique (position, mutant) pairs."""
    measurements = pd.read_excel(MOESM6, sheet_name="TableS5")
    pos_mts = measurements[["Pos_real", "mt_codon"]].drop_duplicates()

    sequences = [KRAS_WT]
    for pos_real, mt_codon in pos_mts[["Pos_real", "mt_codon"]].values:
        if pos_real != pos_real:                 # NaN -> the wild-type placeholder row
            continue
        pos = int(pos_real)
        assert len(mt_codon) == 1
        sequences.append(KRAS_WT[:pos - 1] + mt_codon + KRAS_WT[pos:])
    return sequences


def main():
    token = os.environ.get("ESM_API_KEY")
    if not token:
        raise SystemExit("ESM_API_KEY is not set — export a Forge API token first.")

    from esm.sdk.api import SAEConfig
    from esm.sdk.forge import ESMCForgeInferenceClient
    from cookbook.snippets.sae import get_sae_features

    sequences = build_sequences()
    print(f"[build] {len(sequences)} sequences (1 wild-type + {len(sequences) - 1} mutants)")

    client = ESMCForgeInferenceClient(
        model=ESMC_6B_MODEL,
        url="https://forge.evolutionaryscale.ai",
        token=token,
    )
    # normalize_features=True applies TF-IDF weighting — upweights activations of
    # the more highly specific features.
    sae_config = SAEConfig(model=ESMC_6B_SAE_MODEL, normalize_features=True)

    # get_sae_features pools to one vector per sequence by default (per-residue
    # features for all variants would be ~90 GB and are not needed downstream).
    features = get_sae_features(client, sae_config, sequences)
    print(f"[sae] got SAE features for {len(features)} sequences, "
          f"each of shape {tuple(features[0].shape)}")

    stacked = torch.stack(features)
    torch.save(stacked, SAE_FEATURES)
    print(f"[save] {SAE_FEATURES}  shape={tuple(stacked.shape)}")


if __name__ == "__main__":
    main()
