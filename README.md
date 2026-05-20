# ESM C SAE Feature Disruption Scoring for LoF Variants

This repository contains a pipeline that curates experimentally validated human loss-of-function (LoF) missense variants from UniProt, builds matched benign controls, runs every variant through the ESM Cambrian 6B (ESMC-6B) Sparse Autoencoder (SAE), and scores how much each substitution disrupts the learned feature representation. The central question: **do SAE features learned from protein sequences recover the functional mechanism of LoF mutations?**

---

## Dataset

8,205 human missense variants across 2,181 proteins, split into three classes:

| Class | Source | Variants |
|---|---|---|
| LoF positive (`LoF_like`) | UniProt mutagenesis — LoF phrase in evidence text | 1,307 |
| Benign control A | ClinVar benign/likely benign missense | 3,921 |
| Benign control B | AlphaMissense benign, nearby in sequence | 2,977 |
| **Total** | | **8,205** |

LoF variants are labelled with a mechanism tag derived from keyword-matching the UniProt evidence text:

| Mechanism | Variants |
|---|---|
| `binding_LoF` | 575 |
| `catalytic_LoF` | 446 |
| `regulatory_LoF` | 232 |
| `domain_or_motif_LoF` | 50 |
| `structural_stability_LoF` | 3 |
| `metal_binding_LoF` | 1 |

The curated variant tables are committed to this repo under `data/variants_with_evidence_text/` — see [Data](#data) below.

---

## Key Results

**LoF variants score 2.8–3.3× higher than benign controls** on the primary SAE disruption score:

| Class | LoF score (median) | Max functional loss (med.) | Functional loss fraction (med.) |
|---|---|---|---|
| LoF\_like | 0.530 | 0.106 | 0.683 |
| ClinVar benign | 0.187 | 0.072 | 0.517 |
| AlphaMissense nearby | 0.160 | 0.068 | 0.588 |

The primary score (`LoF_functional_feature_loss_score`) sums the activation loss across SAE features categorised as functionally relevant (catalytic, binding, PTM, structural motif, domain, etc.). The functional loss fraction (0.683 for LoF vs. 0.517 for benign) shows that the disruption is preferentially concentrated in functional features, not diffuse across the representation.

**Mechanism-specific enrichment is interpretable.** When broken down by mechanism tag × SAE feature category, fold enrichments over the nearby-control baseline show clear diagonal structure:

| Mechanism | Strongest SAE category enrichment |
|---|---|
| `catalytic_LoF` | Catalytic function (16.4×), Ligand-binding site (15.9×) |
| `binding_LoF` | Ligand-binding site (13.7×), Domain (8.9×) |
| `regulatory_LoF` | Post-translational modification (8.8×) |
| `domain_or_motif_LoF` | Ligand-binding site (12.9×) |

Full heatmap: `reports/do_sae_features_recover_lof_mechanisms.md`  
Figure: `results_evidence_text/figures/fig_mechanism_category_heatmap.pdf` (on cluster)

---

## Agent Skills

Two agent skills are defined in `skills/` for interactively interpreting individual variants using an LLM agent paired with protein informatics tools.

### Available Skills

| Skill file | Description |
|---|---|
| `ESMC_SAE_variant_interpreter.md` | Core skill: fetches canonical sequence from UniProt, runs ref and mutant through ESMC-6B SAE via Forge API, returns ranked lists of gained/lost SAE features with biological category labels |
| `variant_lof_mechanism.md` | Full agent prompt: reason about LoF mechanism using standard tools (AlphaMissense, ThermoMPNN, AlphaFold, ESMC logit scores) **plus** SAE feature activations |
| `variant_lof_mechanism_no_ESMC_SAE.md` | Ablation prompt: same tools but no SAE feature access — used as the no-SAE baseline |

### What the SAE Interpreter Skill Does

Given a protein (UniProt accession or gene symbol) and a variant (e.g. `R175H`), the skill:
1. Fetches the canonical sequence and validates the reference residue
2. Runs both reference and mutant sequences through the ESMC-6B SAE (Forge API)
3. Computes per-feature activation deltas over a ±8 residue window at the mutation site
4. Returns ranked top-10 lost and gained SAE features, each annotated with a biological category and natural-language summary

The Forge API costs 2 credits per variant (ref + mut). If the reference protein is already cached locally in `results/activations/ref_cache/{accession}.npz`, only 1 credit is consumed.

### Proof-of-Concept Evaluation

We ran a blinded experiment on 10 LoF variants from `positive_tier1.tsv`, comparing agents with and without SAE feature access. Ground-truth UniProt evidence text was withheld from the agent at all times.

**Mean mechanism-identification scores (1–5 scale):** With SAE: **4.5**, Without SAE: **4.2**  
**Blind pairwise judge wins:** Without SAE: 6/10, With SAE: 4/10

Preliminary evidence suggests SAE features are most useful for PTM and subcellular localisation mechanisms, where standard structure-based tools provide weak evidence and the SAE has learned dedicated feature detectors (e.g., Feature 16076 for N-glycosylation sequons, Feature 15254 for phospholipid-binding domains). For well-characterised catalytic mechanisms, sequence-based reasoning alone is often sufficient or superior.

Full write-up with per-variant scores and case analysis: `reports/agent_variant_effects.md`

---

## Data

The processed variant tables are in `data/variants_with_evidence_text/`:

| File | Rows | Description |
|---|---|---|
| `positive_tier1.tsv` | 1,307 | LoF positive set: UniProt variants with LoF evidence text and assigned mechanism tag |
| `poc_lof_variants.tsv` | 8,205 | Full dataset: positives + both benign control sets, QC-passed, ready for scoring |
| `qc_report.txt` | — | Automated QC summary: label counts, mechanism breakdown, protein length stats |

**Key columns** (shared by both TSVs):

| Column | Description |
|---|---|
| `variant_id` | `<accession>_<ref><position><alt>`, e.g. `P04637_R175H` |
| `label` | `LoF_like`, `benign_control`, or `nearby_control` |
| `mechanism_tag` | LoF mechanism (positives) or control class label |
| `label_confidence` | `high` or `medium` |
| `uniprot_accession` | UniProt accession |
| `gene` | HGNC gene symbol |
| `position_1idx` | 1-indexed mutation position in canonical sequence |
| `ref_aa` / `alt_aa` | Reference and alternate amino acids (single-letter) |
| `sequence_context_21aa` | 21-residue window centred on the mutation site |
| `evidence_text` | UniProt curator evidence text (LoF positives only) |

---

## Repo Structure

```
variants_sae/
├── README.md
├── data/
│   └── variants_with_evidence_text/   — curated variant tables (committed)
├── scripts/                           — numbered pipeline scripts
│   ├── config.env.template            — copy to config.env and fill in paths
│   ├── config.py                      — shared config loader (imports config.env)
│   ├── 01_download_uniprot.py
│   ├── 02_parse_uniprot_mutagenesis.py
│   ├── 03_download_clinvar.py
│   ├── 04_download_alphamissense.py
│   ├── 05_build_controls.py
│   ├── 06_dataset_qc.py
│   ├── 07_compute_sae_activations.py  — GPU step, requires ESM Forge API key
│   ├── 08_score_variant_disruptions.py
│   ├── 09_make_figures.py
│   ├── run_pipeline.sh                — SLURM orchestration (CPU steps)
│   └── run_sae_activations.sh         — SLURM job for GPU step
├── reports/                           — human-readable result write-ups
└── skills/                            — agent skill definitions
    └── ESMC_SAE_variant_interpreter.md
```

---

## Running the Pipeline

**Prerequisites:** access to the [EvolutionaryScale Forge API](https://forge.evolutionaryscale.ai) (for ESMC-6B) and an ESM installation with the `esm` Python package.

**1. Configure paths**

```bash
cp scripts/config.env.template scripts/config.env
# Edit config.env: set SAE_BASE_DATA, SAE_ESM_DIR, SAE_PYTHON
# Also update the #SBATCH --account and --output lines in the two .sh files
```

**2. Download UniProt and parse LoF variants** (step 02 produces `positive_tier1.tsv`)

```bash
python scripts/01_download_uniprot.py
python scripts/02_parse_uniprot_mutagenesis.py
```

**3. Build controls and run QC** (steps 03–06, CPU)

```bash
sbatch scripts/run_pipeline.sh
```

**4. Compute SAE activations** (step 07, GPU — requires `ESM_API_KEY` in environment)

```bash
export ESM_API_KEY=your_key
sbatch --export=ALL scripts/run_sae_activations.sh
```

Step 07 is resumable: resubmit after preemption and it picks up from where it left off.

**5. Score and plot** (steps 08–09, CPU)

```bash
sbatch scripts/run_pipeline.sh --start 08
```

---

## Dependencies

- Python ≥ 3.10
- `pandas`, `numpy`, `torch`, `matplotlib`
- [`esm`](https://github.com/evolutionaryscale/esm) — EvolutionaryScale ESM package
- EvolutionaryScale Forge API key (for ESMC-6B inference)
- SLURM cluster with GPU node (for step 07; CPU steps run locally too)
