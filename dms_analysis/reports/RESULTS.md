# Validating ESM-C SAE against KRAS deep mutational scanning

## 1. Goal

A sparse autoencoder (SAE) decomposes a protein language model's embedding of a
sequence into thousands of sparse features, each of which can be given a short
human-readable label. If those SAE features are biologically meaningful, they
could explain *why* a mutation is damaging, not just predict *that* it is.

This report tests the prerequisite for that use. The question is:

> When a mutation alters a protein, do its SAE features change in a way that
> (a) tracks the mutation's measured functional effect, and (b) points to the
> protein's known functional sites?

We test this on KRAS because a deep mutational scanning (DMS) study measured
the functional effect of nearly every possible single amino-acid substitution
across one folding readout and six binding readouts — a dense, quantitative
ground truth to compare the SAE against.

The analysis is organised as two validations of increasing resolution:

1. **Validation 1** — across all substitutions, does the *size* of the SAE
   disruption track the measured functional effect?
2. **Validation 2** — at the mutational hotspots, *which* SAE features are most
   disrupted, and do they match the local biology?

---

## 2. Data and setup

### 2.1 The deep mutational scanning dataset

Source: Weng et al., *Nature* 2024 (KRAS DMS atlas). Seven assays were used.
Each is a protein-fragment complementation assay read out as yeast growth;
all share the KRAS half and differ in the partner the mutation is tested
against:

| Assay | Partner | What it measures |
|---|---|---|
| `folding` | KRAS itself (abundance assay) | intrinsic stability of the folded protein |
| `RAF1`, `PIK3CG`, `RALGDS` | three effector domains | strength of effector binding |
| `SOS1` | the SOS1 exchange factor | strength of binding to the nucleotide-exchange machinery |
| `DARPin K27`, `DARPin K55` | two engineered binders | binding to two designed, non-natural surfaces |

Each assay reports, per substitution, a **ΔΔG value** — the change in folding
or binding free energy the mutation causes (kcal/mol). A **positive ΔΔG means
the mutation destabilises the protein or weakens binding**, i.e. a loss of
function (LOF). The seven assays jointly cover positions 2–188 of KRAS.

### 2.2 SAE features and the "feature drop"

Every variant was run through the ESM Cambrian (ESM-C) 6B model; a sparse
autoencoder trained on the model's layer-60 representations then decomposed
each variant's embedding into 16,384 sparse feature activations. The wild-type
sequence was processed the same way, giving a wild-type feature vector.

The core quantity throughout this report is the **feature drop** — the
activation a mutation *removes* from a feature:

```
drop(feature f) = max(0, activation_wild-type(f) − activation_mutant(f))
```

A drop is zero if the mutation leaves a feature unchanged or increases it; only
lost activation is counted. Feature *gains* are a separate signal and are not
analysed here.

The per-variant feature vectors are assembled into one tensor of shape
(20 amino acids × 187 positions × 16,384 features), aligned cell-for-cell with
the DMS effect matrices. Each of the 16,384 features carries a category and a
one-sentence description, derived from the sequence patterns it responds to
across many proteins; these labels are interpretive hints, not curated
annotations of KRAS specifically.

---

## 3. Methods and statistics

### 3.1 Validation 1 — does SAE disruption track functional effect

**Metric.** For each substitution, the feature drops across all 16,384
features are summarised into one number, the **global score** = the mean of
the K largest drops. SAE features are sparse, so K is small; K = 1, 3, 10 are
all reported.

**Groups.** Within each assay, substitutions are split into:
- **disruptive** — ΔΔG in the most-destabilising 5% of the assay;
- **neutral** — |ΔΔG| ≤ 0.1 (a tight band around no effect).

Tight bands are used deliberately: a wider neutral band admits weakly damaging
substitutions and blurs the contrast.

**Test.** A one-sided **Mann–Whitney U test** asks whether the global score is
larger for disruptive than for neutral substitutions, per assay. As an
effect-size measure we also report the **AUC** (area under the ROC curve) — the
probability that a randomly chosen disruptive substitution has a higher global
score than a randomly chosen neutral one (0.5 = no separation, 1.0 = perfect).
A parameter sweep over the aggregation K and the neutral-band width checks
that the result is not an artefact of those choices.

### 3.2 Validation 2 — which SAE features respond at mutational hotspots

**Metric.** For each position, `max_drop(position, feature)` is the largest
feature drop caused by any of the substitutions at that position — "the
biggest change any mutation here makes to this feature".

**Hotspot clusters.** Within each assay, the 20 positions with the highest
ΔΔG are taken; positions within 2 residues of each other are chained into
clusters (so a cluster is a short stretch of strongly destabilising residues);
the 6 clusters with the highest ΔΔG are kept.

**Statistic and test.** For each cluster and each feature, the observed value
is the **mean of `max_drop` over the cluster's positions**. Its significance
is assessed by a **permutation test**: the same statistic is computed on
200,000 randomly chosen position-sets of the same size, drawn from all
DMS-covered positions of the assay (the cluster's own positions stay in the
pool — excluding them would bias the test toward significance). The empirical
p-value is the fraction of random sets that match or exceed the observed
value. P-values are corrected for multiple testing within each cluster using
the Benjamini–Hochberg false-discovery-rate procedure (significant at q < 0.05).

This test is not circular: clusters are defined by the DMS effect (a
laboratory measurement), while the statistic is the SAE feature drop (a model
measurement) — two independent quantities. The cluster statistic is the *mean*
of `max_drop` over the cluster's positions, not the maximum; under this
permutation scheme a maximum-based statistic returns almost no significant
features.

**Output.** Per cluster, the significant features are ranked and the top 5 are
reported, each with its category and description. These are drawn as "callout"
figures: the DMS effect heatmap with, above each hotspot, a box listing its
most-disrupted SAE features.

---

## 4. Results

### 4.1 Validation 1 — SAE disruption tracks functional effect in every assay

In all seven assays, disruptive substitutions have a significantly larger
global score than neutral ones (one-sided Mann–Whitney U, global score = mean
of the top-3 feature drops):

| Assay | p-value | Assay | p-value |
|---|---:|---|---:|
| folding | 9 × 10⁻¹⁴ | SOS1 | 2 × 10⁻¹¹ |
| RAF1 | 5 × 10⁻¹² | DARPin K27 | 3 × 10⁻⁸ |
| PIK3CG | 4 × 10⁻¹⁸ | DARPin K55 | 4 × 10⁻³² |
| RALGDS | 4 × 10⁻²³ | | |

The separation is moderate in size. Effect sizes (AUC, at the neutral band
|ΔΔG| ≤ 0.10) range from about 0.64 to 0.78 depending on the assay and the
aggregation K:

| Assay | K = 1 (max) | K = 3 | K = 5 | K = 10 |
|---|---:|---:|---:|---:|
| folding | 0.64 | 0.69 | 0.72 | **0.77** |
| RAF1 | **0.68** | 0.66 | 0.64 | 0.63 |
| PIK3CG | **0.74** | 0.71 | 0.68 | 0.66 |
| RALGDS | **0.75** | 0.75 | 0.72 | 0.69 |
| SOS1 | **0.69** | 0.67 | 0.63 | 0.61 |
| DARPin K27 | **0.66** | 0.63 | 0.61 | 0.62 |
| DARPin K55 | 0.76 | **0.78** | 0.76 | 0.74 |

At the headline aggregation (K = 3) every assay clears p < 10⁻⁷; across the
full sweep of K and neutral-band width every assay still clears p < 10⁻⁵
(`results/validation_1_sweep/`). The table reports only which K gives the
cleanest separation, not whether the effect is present.

Interpretation: the SAE responds to mutational disruptiveness — a necessary
foundation. It is a coarse result on its own, since any larger functional
perturbation moves the embedding more; it does not by itself prove the SAE has
captured KRAS-specific biology. That is what Validation 2 addresses.

### 4.2 Validation 2 — the disrupted features match the biology

Significant-feature counts per cluster span 0–153 (q < 0.05): the largest,
most coherent hotspot clusters yield the most, while single-position clusters
yield none — an expected consequence of the statistic collapsing to one value.
The top-ranked features recover canonical KRAS biology at the correct
locations:

- **folding, P-loop cluster (positions 15–17, 19)** — the shortlist includes
  feature 6724, "conserved lysine of the Walker A (P-loop) phosphate-binding
  motif that binds ATP/GTP", at q ≈ 1 × 10⁻³. This is KRAS K16, the invariant
  lysine of the phosphate-binding loop.
- **folding, β-strand 1 / P-loop edge (positions 8–10)** — the shortlist
  includes features describing the β-strand-to-loop junction N-terminal to the
  Walker A motif (q ≈ 1 × 10⁻³).
- **RAF1, P-loop tail (positions 20–21, 23)** — the shortlist includes
  feature 3715, "Motif I (Walker A / P-loop) of P-loop NTPases" (q ≈ 2 × 10⁻²).

In other words, mutating the phosphate-binding loop most disrupts SAE features
that the model associates with phosphate-binding loops. The per-cluster tables
are in `results/validation_2/` and the callout figures in
`results/validation_2/callouts_permutation_<assay>.png` (significant features)
and `callouts_descriptive_<assay>.png` (most-disrupted features, no test).

---

## 5. Limitations and notes

**The result is a foundation, not a mechanism.** Validation 1 shows the SAE
responds to disruptiveness; Validation 2 shows the disrupted features are the
biologically right ones. Neither shows that the SAE distinguishes *mechanisms*
(for example, a fold-destabilising mutation from a binding-interface mutation)
— that requires a direct comparison between assays' disruptive sets and is not
attempted here.

**Single-position clusters are uninformative in Validation 2.** When a hotspot
cluster contains one position the permutation statistic collapses to a single
value and no feature can reach significance. This is a sample-size limit, not
a negative result.

**Structural annotation track in the figures.** The callout figures
(Validation 2) carry a sequence/structure annotation track (binding interface,
nucleotide pocket, buried core, secondary structure) computed from PDB 6VJJ by
`scripts/04_compute_structural_annotations.py`. This track does not pixel-match
the equivalent panel in the source paper, for three understood reasons: (i)
that published panel is drawn with a 2-residue registration offset relative to
its own sequence, verified against the paper's text, and we use correct
numbering; (ii) interface and pocket residues are defined by side-chain
heavy-atom distance, matching the paper's actual code rather than its caption
wording; (iii) the buried-core call depends on the solvent-accessibility tool,
and residues near the cutoff legitimately differ between tools. This affects
only the figure's annotation track, not any SAE or DMS analysis.

---

## 6. Output files

| Path | Contents |
|---|---|
| `results/validation_1/aggregation_<K>.png` | per-assay disruptive-vs-neutral comparison, one per K |
| `results/validation_1/category_stats.csv` | per-assay Mann–Whitney U statistics |
| `results/validation_1/global_drop.parquet` | per-substitution global score |
| `results/validation_1_sweep/` | aggregation × neutral-band robustness sweep |
| `results/validation_2/callouts_permutation_<assay>.png` | DMS heatmap with significant-feature callouts per hotspot |
| `results/validation_2/callouts_descriptive_<assay>.png` | same layout, most-disrupted features (no test) |
| `results/validation_2/feature_pvalues.parquet` | per (assay, cluster, feature) p-value and q-value |
| `results/validation_2/clusters.parquet`, `cluster_features.parquet` | cluster definitions and top features |
