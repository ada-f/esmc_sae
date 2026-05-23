# DMS × ESM-C SAE skills

Reusable skills distilled from the DMS × ESM-C SAE analysis in `dms_analysis/`.
Each skill is a self-contained workflow for **any deep-mutational-scanning (DMS)
screen** paired with ESM-C sparse-autoencoder (SAE) features — the KRAS study in
this repository is the worked example, not the scope.

A skill's "Reference implementation" line points at the `dms_analysis/scripts/`
step it was distilled from; read that script for a concrete, runnable version.

| Skill | What it does |
|---|---|
| [protein_structural_annotations](protein_structural_annotations.md) | Per-residue binding-interface / ligand-pocket / core-surface / secondary-structure annotation from a PDB structure |
| [build_per_mutant_sae_tensor](build_per_mutant_sae_tensor.md) | Compute ESM-C SAE features for every single mutant of a DMS library and assemble the (allele × position × feature) tensor |
| [annotated_dms_heatmap](annotated_dms_heatmap.md) | Plot a DMS effect matrix as a heatmap aligned to the sequence and a structural-annotation track |
| [sae_global_drop_vs_dms](sae_global_drop_vs_dms.md) | Test whether DMS-disruptive substitutions perturb SAE features more than neutral ones |
| [sae_hotspot_feature_enrichment](sae_hotspot_feature_enrichment.md) | Find the SAE features most dropped at DMS loss-of-function hotspots — permutation test + descriptive ranking |
| [retrieve_dms_data_mavedb](retrieve_dms_data_mavedb.md) | Fetch a DMS score set from MaveDB (via ToolUniverse's MaveDB tools) and normalise it into the effect-matrix layout |

A natural pipeline order: `retrieve_dms_data_mavedb` and the SAE tensor supply
the DMS effect matrix and the feature tensor; `protein_structural_annotations`
adds the structural prior; `sae_global_drop_vs_dms` is the foundational test;
`sae_hotspot_feature_enrichment` is the interpretive payload; `annotated_dms_heatmap`
is the figure that ties the SAE callouts to the structural track.

[`tooluniverse_tools_plan.md`](tooluniverse_tools_plan.md) is not a skill — it
records which ToolUniverse tools these skills would need, and which already
exist.

## Shared conventions

These hold across every skill — get them right once, per screen.

- **SAE feature drop.** `drop[f] = max(0, activation_WT[f] − activation_mutant[f])`
  — the activation a mutation *removes* from feature `f`. Feature *gains* are a
  separate signal and are not mixed into "drop".
- **Canonical numbering.** Pin every coordinate (DMS positions, structure
  residues, sequence) to **one** 1-based residue index and verify it with an
  explicit residue-identity check before any cross-source join. Silent
  off-by-N joins are the most common and most damaging error in this work;
  crystal constructs routinely add N-terminal cloning residues.
- **Sign of the DMS effect.** Decide once which tail is "disruptive". For
  folding/binding ΔΔG, positive = destabilising, so disruptive = the top
  quantile; a fitness/growth score is usually the opposite. State it explicitly
  in code — a flipped sign silently inverts every result.
- **Feature labels are interpretive hints.** The ESM-C SAE feature table
  (`uniref90_feature_table.parquet`, 16,384 rows: `feature_id`, `category`,
  `summary`, …) is derived from UniRef90 activation patterns, not per-protein
  expert curation. A feature labelled "Catalytic function" fires on residues
  that *tend* to be catalytic across many proteins — treat it as a hypothesis,
  not an annotation of the specific protein under study.
- **SAE activations are sparse.** k≈64 features are active per residue out of
  16,384; aggregations should use a small top-K (1/3/10), not a broad mean.
