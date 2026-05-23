# Skill: Build a per-mutant ESM-C SAE tensor for a DMS library

## Purpose

For any protein with a deep-mutational-scanning (DMS) library, compute ESM
Cambrian SAE features for every single amino-acid substitution and assemble them
into one tensor aligned to the DMS layout, plus the wild-type reference vector.
This tensor is the input every downstream SAE-vs-DMS skill consumes
(`sae_global_drop_vs_dms`, `sae_hotspot_feature_enrichment`).

Reference implementation: `dms_analysis/scripts/01_compute_sae_features.py`
(API calls → raw feature matrix) and `02_export_sae_tensors.py` (reshape into
the DMS-layout tensor). Could be formalised as a ToolUniverse tool that takes a
wild-type sequence + a mutant list and returns the tensor.

## Inputs

| Input | Notes |
|---|---|
| Wild-type sequence | canonical, 1-based; this is the coordinate system everything else pins to |
| List of single mutants | every `(position, mutant_aa)` the DMS covers |
| ESM-C SAE access | EvolutionaryScale Forge API (`ESM_API_KEY`), or precomputed features |

ESM-C 6B with its layer-60 SAE (`esmc-6b-2024-12` /
`esmc-6b-2024-12_k64_codebook16384_layer60`) is the model used here; the 600M
model has an analogous layer-27 SAE.

## Method

1. **Build mutant sequences.** For each `(position, mutant_aa)`, splice the
   substitution into the wild-type sequence. Keep a strict, recorded ordering —
   wild-type first, then mutants in a fixed iteration order. This ordering *is*
   the contract between the API call and the reshape in step 2.

2. **Run the SAE.** Send wild-type + every mutant through the ESM-C SAE. Use
   **pooled** (one vector per sequence) features — per-residue features for a
   whole library are ~100× larger and are not needed for per-variant analyses.
   `normalize_features=True` applies TF-IDF weighting (upweights specific
   features); keep it consistent across all variants and the wild-type.

3. **Assemble the tensor** `T[a, p, f]` — allele `a` (20 amino acids in a fixed
   order), position `p`, SAE feature `f` (16,384). Reindex the flat API output
   into this layout using the recorded ordering from step 1.

4. **Store the wild-type vector separately** and, optionally, the difference
   tensor `diff[a,p,f] = T[a,p,f] − WT[f]` (some downstream skills want the
   raw tensor + WT, others want the diff — the WT subtraction is cheap to redo).

5. **WT-diagonal cells.** Set cells where `mutant_aa == wild_type_aa` at that
   position to NaN — they are layout placeholders, not real mutants. Stop-codon
   substitutions are dropped (the grid is 20 canonical amino acids).

6. **Write a metadata JSON** recording the amino-acid order, the position list,
   the axis meaning, and the WT-cell convention — so the tensor is decodable
   without re-reading the code.

## Output

- `sae_tensor.pt` — `(20, n_positions, 16384)`, NaN at WT-diagonal cells.
- `sae_tensor_diff.pt` — same shape, `mutant − WT`.
- `wt_sae_vector.pt` — `(16384,)`.
- `sae_tensor_meta.json` — amino-acid order, position list, axis conventions.

## Pitfalls

- **Row ordering.** The order in which mutants were sent to the API must be
  reproduced *exactly* when reshaping; an unrecorded `drop_duplicates` or sort
  scrambles positions with no error raised. Verify with a landmark (below).
- **API cost.** One call per mutant — a full library is thousands of calls.
  Cache aggressively and run the wild-type only once. The step is resumable in
  principle; budget for it.
- **Verify the tensor before trusting it.** A known benign mutation should give
  a near-zero diff; a known disruptive one a large diff; every WT-diagonal cell
  must be NaN; and the WT vector's norm should be stable across reruns.
