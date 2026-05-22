# Skill: Retrieve DMS data from MaveDB

## Purpose

Fetch a deep-mutational-scanning (DMS) score set from
[MaveDB](https://www.mavedb.org) — the public repository of multiplexed
variant-effect assays — and normalise it into the `(position × substitution)`
effect matrix that every other skill in this set consumes. This is what lets
the SAE-vs-DMS workflow run on any protein MaveDB hosts, not only the local
KRAS files.

ToolUniverse already provides the MaveDB API tools (see below), so this skill
is **orchestration, not a tool to build** — it calls those tools and adds the
parsing / numbering step they do not do.

## ToolUniverse tools used

| Tool | Role |
|---|---|
| `MaveDB_search_score_sets` | find score sets by gene / keyword → URNs |
| `MaveDB_get_score_set` | score-set metadata: assay method, variant count, publication |
| `MaveDB_get_variant_scores` | the variant table: HGVS string + functional score |
| `MaveDB_search_experiments` | (optional) browse experiments grouping several score sets |

## Method

1. **Find the score set.** `MaveDB_search_score_sets` with the gene symbol or
   protein name; inspect results and pick the URN whose assay matches what you
   need (a stability assay, a binding assay, …). `MaveDB_get_score_set`
   confirms the assay method and variant count before committing.

2. **Pull the scores.** `MaveDB_get_variant_scores` returns one row per
   variant: an HGVS string and a functional score.

3. **Parse to `(position, wild_type_aa, mutant_aa)`.** Variants come as HGVS
   protein strings (e.g. `p.Arg175His`). Parse them, convert three-letter to
   one-letter amino acids, and **keep single missense variants only** — drop
   synonymous, nonsense, indel, and multi-mutant rows.

4. **Pin to canonical numbering.** MaveDB targets carry their own coordinate
   system, which may not equal the canonical UniProt sequence. Resolve the
   offset and verify it with a residue-identity check (the shared numbering
   convention in `skills/README.md`) before emitting anything.

5. **Emit the effect matrix** — rows = 20 amino acids, columns = positions,
   cells = the functional score — the layout `build_per_mutant_sae_tensor`
   and `sae_global_drop_vs_dms` expect.

## Output

A `(position × substitution)` effect matrix for one assay, on canonical
numbering, plus a note recording the MaveDB URN, the assay method, and the
score column used (for provenance).

## Pitfalls

- **Score semantics vary per score set.** One MaveDB score set may report a
  stability/abundance score, another a binding score, with different signs and
  scales. Read `MaveDB_get_score_set` metadata and decide explicitly which
  tail is "disruptive" — the downstream tests depend on that sign (see the
  effect-sign convention in `skills/README.md`).
- **Numbering offsets are silent.** Always verify residue identity at a
  landmark before joining MaveDB positions to a sequence or structure.
- **Multi-mutant rows.** Many score sets include double and higher mutants;
  the SAE-vs-DMS skills here assume single substitutions — filter to those, or
  extend the tensor layout deliberately.
- **Coverage is partial.** A score set rarely measures all 20 substitutions at
  all positions; expect a sparse matrix and carry missing cells as missing,
  not zero.
