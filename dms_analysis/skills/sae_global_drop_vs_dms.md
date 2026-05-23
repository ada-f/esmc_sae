# Skill: Global SAE drop vs DMS effect

## Purpose

Test the foundational question of any DMS × SAE study: do DMS-disruptive
substitutions perturb the SAE feature representation more than DMS-neutral
substitutions? A positive result establishes that the SAE-drop axis carries DMS
signal — the prerequisite for any per-position / per-feature interpretation.

Reference implementation: `dms_analysis/scripts/05_validation1_global.py`
(per-assay statistics) and `06_validation1_aggregations.py` (figures);
`07_validation1_sweep.py` sweeps the parameter grid. This is an orchestration
workflow — it composes the SAE tensor, a DMS table, and a statistical test —
so it belongs as a ToolUniverse *skill*, not an atomic tool
(see `tooluniverse_tools_plan.md`).

## Inputs

| Input | Notes |
|---|---|
| Per-mutant SAE tensor + WT vector | from `build_per_mutant_sae_tensor` |
| DMS effect table | per `(position, mutant_aa)`, one or more assays of ΔΔG (or fitness) |
| Aggregation K | mean of the top-K feature drops; sweep `K ∈ {1, 3, 10}` |

## Method

1. **Per-substitution drop** — `drop[f] = max(0, WT[f] − mutant[f])` over all
   16,384 features (the lost activation only).

2. **Global score** — `topK_drop = mean of the K largest drops`. SAE
   activations are sparse, so a small K (1 = the single max, 3, 10) is the
   right summary, not a mean over all features. Report several K — the best K
   varies by assay and the choice should be transparent, not tuned.

3. **DMS categories** (per assay):
   - neutral = a tight band around zero, e.g. `|ΔΔG| ≤ 0.1`;
   - disruptive = the most-disruptive tail, e.g. the top 5 % of substitutions.
   A loose neutral band leaks weakly-disruptive variants into the "neutral"
   group and erodes the contrast — keep it tight. *Sign:* for folding/binding
   ΔΔG, positive = destabilising, so disruptive is the **top** tail; flip for a
   fitness/growth score.

4. **Test** — one-sided Mann–Whitney U: is `topK_drop` larger for disruptive
   than neutral substitutions? Report p per assay, and check robustness across
   K and across the neutral-band / disruptive-cutoff grid (the sweep).

## Output

- `global_drop.parquet` — per-variant global score.
- `category_stats.csv` — per-assay U statistic, group medians, p-value.
- `aggregation_<K>.png` — per-assay neutral-vs-disruptive box/strip panels
  with one shared y-axis label ("SAE feature drop").
- a parameter sweep (aggregation × neutral-band) for robustness.

## Interpretation & limits

- A significant result shows the SAE *responds to mutational disruptiveness* —
  a necessary foundation, but a **coarse** one: it does not by itself show the
  SAE has captured protein-specific biology, since any larger functional
  perturbation moves the embedding more.
- Treat this as groundwork. The interpretive payload — *which* features, and
  *where* — is in `sae_hotspot_feature_enrichment`.
- Resist over-reading per-assay differences in the best K (e.g. "folding wants
  top-10, binding wants max"). A neutral-vs-disruptive contrast cannot support
  claims about how *broad* the perturbation is across assays; that needs a
  separate analysis comparing disruptive sets directly.
