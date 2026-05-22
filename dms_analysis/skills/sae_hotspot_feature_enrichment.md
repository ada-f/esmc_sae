# Skill: SAE features dropped at DMS hotspots

## Purpose

At the loss-of-function (LOF) hotspots of a DMS map, identify which SAE features
the mutations most disrupt — as a permutation-tested shortlist and as a plain
descriptive ranking — and surface their interpretive labels. The output is a
per-hotspot "callout" of SAE features that should reflect the protein's known
biology (its domains, motifs, catalytic/binding residues).

Reference implementation: `dms_analysis/scripts/08_validation2_hotspot_enrichment.py`
(permutation test), `09_validation2_descriptive.py` (descriptive ranking),
`10_validation2_plot_callouts.py` (figure). This is an orchestration workflow —
it composes the SAE tensor, a DMS table, the SAE feature labels, and a
permutation test — so it belongs as a ToolUniverse *skill*; it depends on the
SAE-extraction and SAE-interpretation tools in `tooluniverse_tools_plan.md`.

## Inputs

| Input | Notes |
|---|---|
| Per-mutant SAE tensor (or diff tensor) + WT vector | from `build_per_mutant_sae_tensor` |
| DMS effect table | per `(position, mutant_aa)` ΔΔG |
| Feature label table | the ESM-C SAE `category` + `summary` per feature |

## Method

1. **Per-position drop statistic** —
   `max_drop[p, f] = max over alleles of max(0, WT[f] − mutant[a,p,f])`:
   the biggest drop any substitution at position `p` causes in feature `f`.

2. **Hotspot clusters** — take the top-K positions by max ΔΔG; chain adjacent
   ones (gap ≤ 2) into domain-level clusters; keep the top clusters by
   within-cluster max ΔΔG. The max-ΔΔG position in each cluster is its rep.

3. **Permutation test** (per cluster, per feature):
   - the observed statistic is the mean of `max_drop` over the cluster's
     positions;
   - the null is the same statistic on equally-many positions drawn at random
     from all DMS positions of the protein, the cluster's own positions
     included — excluding them would bias the test toward significance;
   - the empirical p-value is corrected within each cluster with a
     Benjamini–Hochberg false-discovery-rate adjustment.
   Use the mean of `max_drop`, not the maximum: under this null a maximum-based
   statistic returns almost no significant features. The test asks whether a
   feature responds to one cluster more than to a random same-size set of
   positions; it is not a cluster-versus-cluster comparison.

4. **Descriptive ranking** — independently, per cluster rank features by
   `mean(max_drop)` across the cluster's positions; take the top 5. No test.
   This is the fast "what drops most here" view; it needs only the tensor, not
   the (slow) permutation run.

5. **Annotate** every shortlisted feature with its `category` + `summary`, and
   read the shortlist against the cluster's known structural role.

## Output

- `clusters.parquet`, `feature_pvalues.parquet`, `cluster_features.parquet`
  (permutation test); `descriptive_features.parquet` (descriptive ranking).
- `callouts_permutation_<assay>.png` — heatmap + per-cluster callout boxes of
  the significant features.
- `callouts_descriptive_<assay>.png` — same layout, most-dropped features.

## Interpretation & limits

- A good result shows the SAE *surfaces the right features* at hotspots — e.g.
  a nucleotide-binding-loop hotspot calling Walker-A / P-loop features. The
  value is *which specific features* light up.
- Caveat: hotspots and SAE features both track domains, so this recovers
  biology that sequence or structure annotation alone would also give — it is
  not, on its own, a DMS-specific claim. Pair it with `sae_global_drop_vs_dms`,
  which makes the DMS effect load-bearing.
- Single-position clusters return no significant features — the statistic is
  then a single value with no resolving power. This is an expected sample-size
  limit. Multi-position clusters carry the signal.
- No circularity: the cluster is defined by ΔΔG (a fitness/stability
  measurement), the statistic by SAE drop (an embedding measurement). Their
  correlation is the empirical finding — a random SAE would yield ~no hits.
