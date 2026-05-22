# ToolUniverse integration plan — DMS × ESM-C SAE

The skills in this folder describe deep-mutational-scanning (DMS) × sparse-
autoencoder (SAE) analysis workflows. Running them inside ToolUniverse (TU)
requires each step to be available as a TU tool. This document lists the
capabilities TU already provides and the new tools that would need to be added.

In short: TU already covers DMS retrieval, ESM-C embeddings, structure
fetching, pre-computed variant-effect scores, and general compute. The missing
pieces are the **SAE layer** (extracting SAE features and interpreting them)
and **per-residue structural annotation**.

---

## Capabilities ToolUniverse already provides

These cover parts of the workflow and should be used as-is:

| Capability | ToolUniverse tool(s) |
|---|---|
| Retrieve DMS / variant-effect score sets | `MaveDB_search_score_sets`, `MaveDB_get_score_set`, `MaveDB_get_variant_scores`, `MaveDB_search_experiments` |
| ESM-C protein embeddings (mean-pooled or per-residue) | `ESM_get_protein_embedding` |
| ESM-C sequence likelihood / zero-shot variant score | `ESM_score_sequence` |
| Structure prediction and fetching | `ESM_fold_protein`, `ESMFold_predict_structure`, the RCSB PDB tools, `alphafold_get_prediction` |
| Pre-computed variant-effect scores | `AlphaMissense_*`, `ProtVar_*`, `OpenCRAVAT_annotate_variant`, `CADD_get_variant_score` |
| Protein domain ranges | `InterPro_get_protein_domains`, RCSB `get_polymer_entity_annotations` |
| Statistics, plotting, and dataframe operations | `python_code_executor` |

---

## New tools ToolUniverse needs

### Tool 1 — SAE feature extraction *(critical)*

Decompose an ESM-C embedding into its ~16,384 sparse SAE features. Every SAE
skill depends on this, and no current TU tool exposes it.

- **Where it fits:** alongside `ESM_get_protein_embedding` in `esm_tool.py`.
- **Input:** a protein sequence; optionally the model and SAE choice
  (`esmc-6b`, `esmc-600m`), a `pooled` flag (one vector per sequence, or one
  per residue), and a `normalize_features` flag (TF-IDF feature weighting).
- **Output:** SAE feature activations — pooled `(16384,)` or per-residue
  `(L, 16384)`, sparse (about 64 active features per residue), with the
  beginning- and end-of-sequence tokens removed.
- **Used by:** `build_per_mutant_sae_tensor` directly, and through the
  per-mutant tensor by `sae_global_drop_vs_dms` and
  `sae_hotspot_feature_enrichment`.

### Tool 2 — SAE feature interpretation *(critical)*

Map an SAE feature ID to its meaning. Tool 1 produces 16,384 activation values
per protein; without a way to label them, those values cannot be interpreted.
No current TU tool provides this.

- **Where it fits:** a method on `esm_tool.py`, or a small reference tool that
  serves the SAE feature table.
- **Input:** one or more feature IDs (0–16,383) for a given SAE model.
- **Output:** for each feature — its `category`, `summary`, `description`, and
  activation threshold; optionally the protein families it is most associated
  with.
- **Used by:** `sae_hotspot_feature_enrichment`, and any interpretive use of
  Tool 1's output.

### Tool 3 — per-residue structural annotation *(high priority)*

For each residue of a protein structure, report whether it lies at a binding
interface, lines a ligand pocket, is buried or solvent-exposed, and which
secondary-structure element it belongs to. TU can fetch structures and provides
domain ranges, but it does not compute these per-residue properties.

- **Where it fits:** an extension of `structural_biology_tools`, or a new
  `structure_annotation_tool.py`.
- **Input:** a PDB ID or file; the target chain; partner chain(s) for the
  interface; ligand residue names for the pocket; distance and solvent-
  accessibility cutoffs.
- **Output:** for each residue — interface and ligand-pocket flags (by minimum
  side-chain heavy-atom distance), a core/surface call (by relative solvent
  accessibility), and the secondary-structure element.
- **Used by:** `protein_structural_annotations`, `annotated_dms_heatmap`.

### Tool 4 — DMS effect-matrix figure *(optional)*

A plotting tool for the annotated DMS heatmap. This is optional —
`python_code_executor` already supports custom plotting — and is worth
formalising only if an annotated DMS heatmap becomes a frequently requested
figure.

---

## Orchestration workflows: skills, not tools

Two of the workflows in this folder are orchestration: they chain the tools
above with `python_code_executor` rather than introducing new computation.
They belong in TU's `skills/` directory as workflow definitions, not as new
atomic tools:

- **`sae_global_drop_vs_dms`** and **`sae_hotspot_feature_enrichment`** each
  combine Tool 1 (and, for interpretation, Tool 2) with a MaveDB-retrieved DMS
  table and a statistical test.

Tool 1 and Tool 2 also extend the existing `tooluniverse-variant-functional-
annotation` skill: for a single variant, extract the SAE features of the
wild-type and mutant sequences and report the labelled features the mutation
most disrupts. This adds an interpretable, mechanism-oriented explanation
alongside the pathogenicity scores that skill already aggregates
(AlphaMissense, CADD, and others).

---

## Out of scope

Atlas-wide nearest-neighbour retrieval and SAE-similarity nearest-neighbour
retrieval would support a broader protein-function annotation goal. They are
not required by any of the six skills in this folder and are therefore out of
scope for this plan; they should be specified together with that broader
workflow.
