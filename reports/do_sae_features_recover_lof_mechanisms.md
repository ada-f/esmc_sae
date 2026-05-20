# Variant Curation and SAE Disruption Scoring — ESM-Atlas LoF Proof-of-Concept

## Table of Contents
1. [Overview](#overview)
2. [Positive Set Curation](#positive-set-curation)
3. [Negative Control Curation](#negative-control-curation)
4. [SAE Activation Scoring](#sae-activation-scoring)
5. [Results](#results)
6. [Output Files](#output-files)
7. [Pipeline Summary](#pipeline-summary)

---

## Overview

This document describes an end-to-end pipeline that curates a set of experimentally validated loss-of-function (LoF) protein variants and a matched set of benign controls, runs both sets through the ESM Cambrian (ESMC) sparse autoencoder (SAE), and scores how much each single amino-acid substitution disrupts the learned feature representation. The pipeline is implemented in nine numbered scripts under `variants_sae/scripts/`.

**Defining LoF and mechanism assignment.** A variant is designated LoF if its UniProt mutagenesis evidence text contains a curated LoF phrase and no exclusion phrase. The functional mechanism (e.g., catalytic loss, binding loss) is then assigned by keyword-matching the same evidence text. This grounds the `mechanism_tag` directly in curator-written experimental descriptions rather than in positional overlap with functional site annotations.

---

## Positive Set Curation

### Step 1 — UniProt Data Acquisition
**Script:** `01_download_uniprot.py`

All reviewed human Swiss-Prot entries that carry at least one `Mutagenesis` annotation are downloaded from the UniProt REST streaming endpoint in JSON format. The query `reviewed:true AND organism_id:9606 AND ft_mutagen:*` returns approximately 5,600 entries, stored as a gzip-compressed JSONL file together with a plain-text accession list.

### Step 2 — Parsing LoF Mutagenesis Records and Mechanism Assignment
**Script:** `02_parse_uniprot_mutagenesis.py`

Each UniProt entry is scanned for `Mutagenesis` features. A record passes the initial filter only if it describes a **single** amino-acid substitution, the reference residue matches the canonical sequence at the stated position, and the evidence text satisfies both of the following conditions simultaneously:

1. **Positive LoF phrase found:** the evidence text (case-insensitive) contains at least one of 25 curated LoF phrases (e.g., "abolishes activity", "no detectable activity", "loss of function", "catalytically inactive", "completely inactive").
2. **No exclusion phrase:** none of 17 curated exclusion phrases appear in the same text (e.g., "no effect", "retains activity", "gain of function", "partial", "reduces but does not abolish").

**Mechanism is assigned from the evidence text**, using six keyword patterns applied in priority order. The first matching pattern determines the `mechanism_tag`:

| Priority | `mechanism_tag` | Example evidence-text keywords |
|---|---|---|
| 1 | `catalytic_LoF` | "catalytic", "active site", "enzymatic activity", "catalytic triad" |
| 2 | `metal_binding_LoF` | "metal binding", "zinc", "iron-sulfur", "coordinates metal" |
| 3 | `binding_LoF` | "binding", "interaction", "affinity", "recognition" |
| 4 | `structural_stability_LoF` | "folding", "stability", "unfolding", "misfolding" |
| 5 | `regulatory_LoF` | "phosphorylation", "glycosylation", "acetylation", "ubiquitination", "regulatory" |
| 6 | `domain_or_motif_LoF` | "domain", "motif", "signal peptide", "dimerization" |

Variants whose evidence text matches none of the six patterns are excluded. Each passing variant receives:
- a unique `variant_id` of the form `<accession>_<ref><position><alt>`
- a `label_confidence` of `high` for four mechanism types (catalytic, binding, metal binding, structural stability) and `medium` for two (regulatory, domain/motif)

Quality filters applied at this stage: protein length ≤ 2700 residues (ESMC practical input limit), both reference and alternate residues are standard amino acids, and duplicates (same accession, position, alt AA) are removed.

Outputs:
- `data/raw/uniprot/parsed_mutagenesis_lof_evidence.tsv` — full filtered and annotated set
- `data/variants_with_evidence_text/positive_tier1.tsv` — Tier 1 positive set

---

## Negative Control Curation

Two complementary negative control sets are constructed to bracket the LoF signal.

### Negative Set A — ClinVar Benign Missense
**Scripts:** `03_download_clinvar.py`, `05_build_controls.py`

The ClinVar `variant_summary.txt.gz` file is downloaded from NCBI and filtered to: GRCh38 human single-nucleotide variants; classified Benign, Likely benign, or Benign/Likely benign without conflicting interpretations; parseable as a missense substitution from the HGVS `p.` notation.

Variants are mapped to UniProt canonical sequences via gene symbol. A ClinVar variant is admitted to Negative Set A only if: (i) the reference residue matches the UniProt sequence, (ii) the position does not overlap any UniProt functional site annotation, (iii) it does not duplicate any positive variant, and (iv) the protein is ≤ 2700 residues. Negative Set A is downsampled to at most 3× the number of positive variants (random seed 42).

### Negative Set B — AlphaMissense Benign Nearby Controls
**Scripts:** `04_download_alphamissense.py`, `05_build_controls.py`

AlphaMissense amino-acid substitution pathogenicity predictions (Cheng et al., *Science* 2023; Zenodo record 8208688) are downloaded for all ~216 million possible human missense variants. Only variants classified `am_class = benign` in the same proteins as the positive set are kept.

Candidate benign variants are further filtered per protein: (i) reference residue must match the UniProt sequence, (ii) position must not overlap any functional site, (iii) position must be at least 10 residues from any functional site, and (iv) position must be within ± 50 residues of at least one positive variant in the same protein, or within the same annotated domain. This "nearby" criterion ensures negative controls share local sequence context with positive variants, making the comparison a fair test of the SAE disruption signal. Up to 10 controls per protein are selected, prioritising the lowest AlphaMissense pathogenicity scores.

### Dataset QC
**Script:** `06_dataset_qc.py`

Six automated QC checks are run before SAE scoring:

1. Reference amino acid validated against the UniProt canonical sequence at the stated position.
2. Each record is a single amino-acid substitution with valid standard AAs.
3. No duplicated `variant_id`.
4. Positive and negative sets do not share any (accession, position, alt AA) triple.
5. All proteins are within the 2700-residue length limit.
6. No row has `ref_aa == alt_aa`.

---

## SAE Activation Scoring

### Model
**Script:** `07_compute_sae_activations.py`

All variants are scored with the ESMC 6B model (`esmc-6b-2024-12`) paired with its layer-60 SAE (`esmc-6b-2024-12_k64_codebook16384_layer60`). The SAE has a codebook of 16,384 features with top-*k* = 64 sparsity.

For each unique protein the reference (wild-type) sequence is run through the model once and the resulting sparse activation tensor (sequence length × 16,384) is cached to disk at `results/activations/ref_cache/`. For each variant the single amino-acid substitution is applied to the canonical sequence, the mutant tensor is computed, and per-variant feature deltas are computed over four window sizes (± 1, ± 4, ± 8, ± 16 residues centred on the mutation position). The primary window is ± 8.

### Disruption Score Computation
**Script:** `08_score_variant_disruptions.py`

Per-feature deltas are joined to a pre-computed feature annotation table (`uniref90_feature_table.parquet`) that assigns each of the 16,384 SAE features to a biological category and provides a natural-language summary.

**Step 1 — window-max activations.** For a mutation at position *p*, the ± 8 residue window spans positions [*p* − 8, *p* + 8]. For each SAE feature *f*, the window-max activation is the maximum activation value that feature takes at any position within the window:

```
wmax_f^ref = max{ a_f^ref(i)  :  i ∈ [p−8, p+8] }
wmax_f^mut = max{ a_f^mut(i)  :  i ∈ [p−8, p+8] }
```

where `a_f(i)` is the SAE activation of feature *f* at residue *i* (zero for features not in the sparse top-*k* set).

**Step 2 — per-feature delta, loss, and gain.**

```
Δf     = wmax_f^mut − wmax_f^ref      (signed change)
loss_f = max(0,  −Δf)                 (activation lost by mutation)
gain_f = max(0,  +Δf)                 (new activation gained by mutation)
```

Loss captures features that were active in the reference but suppressed in the mutant; gain captures features newly activated by the substitution.

**Step 3 — functional weighting.** Each feature is assigned a binary weight based on its annotated biological category. Seven categories are designated "functional" (weight = 1): Catalytic function, Ligand-binding site, Interaction site, Post-translational modification, Structural motif, Domain, Sequence motif. All other categories (Disorder, Membrane-associated, Compositional bias, etc.) receive weight = 0.

**Step 4 — primary score.** The **`LoF_functional_feature_loss_score`** is the weighted sum of feature losses:

```
LoF_score = Σ_f  w_f · loss_f
```

where the sum is over all features *f* in the union of active features (reference ∪ mutant) within the ± 8 window, and `w_f ∈ {0, 1}` is the functional weight. This score accumulates evidence that the mutation erases activations of biologically interpretable SAE features.

**Auxiliary scores:**

| Score | Definition |
|---|---|
| `max_functional_feature_loss` | max{ loss_f : f ∈ F_func } |
| `mean_top5_functional_feature_loss` | mean of 5 largest loss_f values among functional features |
| `functional_loss_fraction` | Σ(functional loss_f) / Σ(all loss_f) |
| `n_functional_features_active` | count of functional features with loss_f > 0 |
| `max_<mechanism>_feature_loss` | max loss_f restricted to SAE categories matching the variant's mechanism tag |

**`max_functional_feature_loss` — formula and interpretation:**

```
max_functional_feature_loss = max{ loss_f  :  f ∈ F_func }
```

where F_func is the set of SAE features belonging to any of the seven functional categories listed above, and loss_f = max(0, wmax_f^ref − wmax_f^mut) is the activation lost by the mutation for feature *f* within the ± 8-residue window.

This is the **single largest activation loss** among all functional features for a given variant — it captures whether any one specific functional SAE feature is sharply ablated, regardless of how many features are affected overall. Contrast with `LoF_functional_feature_loss_score`, which is a *sum* over all functional features and can score high by disrupting many features a little. A high `max_functional_feature_loss` means the mutation essentially silences at least one particular functional feature cleanly, which is a more targeted mechanistic signal.

---

## Results

### Dataset Composition

The final scored dataset contains **8,205 variants across 2,181 unique proteins**.

| Label | Variants | Fraction |
|---|---|---|
| LoF\_like (positive) | 1,307 | 15.9% |
| benign\_control (ClinVar) | 3,921 | 47.8% |
| nearby\_control (AlphaMissense) | 2,977 | 36.3% |
| **Total** | **8,205** | |

The 1,307 LoF variants span six mechanism classes assigned from evidence text:

| Mechanism tag | Count | % of positives |
|---|---|---|
| binding\_LoF | — | — |
| catalytic\_LoF | — | — |
| domain\_or\_motif\_LoF | — | — |
| regulatory\_LoF | — | — |
| structural\_stability\_LoF | — | — |
| metal\_binding\_LoF | — | — |

*Exact per-mechanism counts can be read from `data/variants_with_evidence_text/poc_lof_variants.tsv`. The metal\_binding\_LoF group is very small (n ≈ 1 in the scored set), so its statistics should be interpreted with caution.*

### SAE Disruption Scores

LoF variants score substantially higher than both negative control classes on all primary metrics:

| Label | LoF score (mean) | LoF score (median) | max functional loss (med.) | functional loss fraction (med.) |
|---|---|---|---|---|
| LoF\_like | 0.745 | 0.530 | 0.106 | 0.683 |
| benign\_control | 0.240 | 0.187 | 0.072 | 0.517 |
| nearby\_control | 0.205 | 0.160 | 0.068 | 0.588 |

Key observations:

- **`LoF_functional_feature_loss_score`:** LoF variants score ~2.8× higher (median) than ClinVar benign controls and ~3.3× higher than nearby controls.
- **`max_functional_feature_loss`:** LoF variants (0.106) show that at least one functional SAE feature is sharply silenced per variant, versus 0.072 and 0.068 for the two control classes.
- **Functional loss fraction:** LoF variants (0.683) show the majority of their total feature disruption is concentrated in functionally annotated SAE features, compared to 0.517 for ClinVar benign variants. Nearby controls fall in between (0.588), consistent with their placement within the same domain neighbourhood as positive sites.

### Mechanism-Specific Disruption Patterns

Within the LoF positive class, mechanism-specific maximum feature losses (medians):

| Mechanism | max feature loss | max functional feature loss |
|---|---|---|
| metal\_binding\_LoF | 0.193 | 0.193 |
| structural\_stability\_LoF | 0.156 | 0.156 |
| binding\_LoF | 0.122 | 0.105 |
| domain\_or\_motif\_LoF | 0.121 | 0.091 |
| catalytic\_LoF | 0.118 | 0.108 |
| regulatory\_LoF | 0.116 | 0.100 |

### Heatmap — Fold Enrichment over Nearby-Control Baseline

Median max feature loss per mechanism × SAE category, expressed as fold enrichment over the nearby-control median for the same category:

| Mechanism | Catalytic function | Ligand-binding site | Interaction site | Structural motif | Domain | PTM | Sequence motif |
|---|---|---|---|---|---|---|---|
| catalytic\_LoF | **16.4×** | **15.9×** | 2.2× | 4.0× | 7.3× | 2.4× | 7.5× |
| binding\_LoF | 4.8× | **13.7×** | 4.5× | 4.0× | **8.9×** | 3.5× | 8.2× |
| metal\_binding\_LoF | 8.1× | 3.2× | **67.5×** | 3.1× | 1.6× | 0.8× | **150.9×** |
| structural\_stability\_LoF | **11.9×** | 9.2× | **13.3×** | 6.0× | **18.1×** | **52.9×** | **132.6×** |
| regulatory\_LoF | 1.9× | 5.2× | 2.5× | 3.2× | 2.8× | **8.8×** | 3.1× |
| domain\_or\_motif\_LoF | 4.3× | **12.9×** | 4.6× | 3.1× | 2.4× | 4.2× | 7.9× |

Nearby-control baseline (median per category): Catalytic function 0.0013, Ligand-binding site 0.0035, Interaction site 0.0029, Structural motif 0.0125, Domain 0.0044, PTM 0.0022, Sequence motif 0.0004.

Notable patterns:
- **Catalytic LoF** shows the strongest enrichment in Catalytic function (16.4×) and Ligand-binding site (15.9×), consistent with active-site variants disrupting both catalytic and substrate-binding SAE features.
- **Binding LoF** shows the expected enrichment in Ligand-binding site (13.7×) and Domain (8.9×).
- **Regulatory LoF** shows the expected enrichment in PTM (8.8×).
- **Metal binding LoF** shows extreme enrichment in Interaction site (67.5×) and Sequence motif (150.9×); however, this group contains only n ≈ 1 variant and these values should not be interpreted without a larger sample.
- **Structural stability LoF** shows the most striking off-diagonal enrichments: PTM (52.9×) and Sequence motif (132.6×), suggesting that stability-disrupting variants can incidentally ablate features associated with regulatory and motif-level representations.

---

## Output Files

All results are written under the base data directory `/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ESM-ATLAS/variants_sae/`.

| File | Description |
|---|---|
| `data/variants_with_evidence_text/positive_tier1.tsv` | Tier 1 LoF positive set with evidence-text mechanism tags |
| `data/variants_with_evidence_text/poc_lof_variants.tsv` | Full dataset (positives + both control sets) ready for scoring |
| `results_evidence_text/tables/variant_scores_raw.tsv` | Per-variant aggregate SAE statistics (8,205 rows): active feature counts, max/sum feature loss and gain, total absolute delta, per-window (± 1/4/8/16) totals |
| `results_evidence_text/tables/variant_scores.tsv` | Enriched per-variant table with functional disruption scores (`LoF_functional_feature_loss_score`, `max_functional_feature_loss`, `mean_top5_functional_feature_loss`, `functional_loss_fraction`, mechanism-specific component scores) |
| `results_evidence_text/tables/variant_feature_deltas.tsv.gz` | Per-(variant, feature) delta records for all features active in the ± 8 window (2,351,685 rows): `feature_id`, `delta_window_max`, `feature_loss`, `feature_gain`, per-window values |
| `results_evidence_text/tables/variant_top_disrupted_features.tsv` | Top-10 features per variant by loss, gain, and absolute delta, annotated with SAE feature category and natural-language summary (164,100 rows) |
| `results_evidence_text/figures/fig_mechanism_category_heatmap.pdf` | Heatmap of mechanism × SAE feature category fold enrichment |
| `results_evidence_text/figures/fig_mechanism_category_heatmap.png` | Same heatmap as PNG |

---

## Pipeline Summary

| Step | Script | Description |
|---|---|---|
| 1 | `01_download_uniprot.py` | Download ~5,600 reviewed human UniProt entries with mutagenesis annotations |
| 2 | `02_parse_uniprot_mutagenesis.py` | Extract single-AA LoF substitutions; assign mechanism from evidence-text keywords; apply quality filters; produce `positive_tier1.tsv` |
| 3 | `03_download_clinvar.py` | Download and filter ClinVar for human benign missense SNVs |
| 4 | `04_download_alphamissense.py` | Download AlphaMissense; extract benign predictions for positive-set proteins |
| 5 | `05_build_controls.py` | Build Negative Set A (ClinVar) and Negative Set B (AlphaMissense nearby); assemble `poc_lof_variants.tsv` |
| 6 | `06_dataset_qc.py` | Run six automated QC checks; write `qc_passed.tsv` |
| 7 | `07_compute_sae_activations.py` | Run ESMC-6B + SAE on all variants; cache ref activations; write `variant_feature_deltas.tsv.gz` and `variant_scores_raw.tsv` |
| 8 | `08_score_variant_disruptions.py` | Join deltas with feature annotations; compute functional disruption scores; write `variant_scores.tsv` and `variant_top_disrupted_features.tsv` |
| 9 | `09_make_figures.py` | Generate heatmap of mechanism × SAE feature category, normalised by nearby-control baseline |

### Orchestration

CPU steps (3–6) are submitted as a SLURM batch job (requires `positive_tier1.tsv` from step 2 to already exist):
```bash
sbatch scripts/run_pipeline.sh
```

The GPU step (7) must be submitted separately with the ESM API key exported:
```bash
sbatch --export=ALL scripts/run_sae_activations.sh
```

After the GPU step completes, resume the remaining CPU steps:
```bash
sbatch scripts/run_pipeline.sh --start 08
```
