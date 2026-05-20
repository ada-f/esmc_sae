## Agent Variant-Effect Experiments

To assess whether SAE feature information improves an LLM agent's ability to reason about molecular mechanisms of variant LoF, we ran a small proof-of-concept experiment using the first 10 variants in the Tier 1 positive set (`positive_tier1_with_evidence_text.tsv`). Each variant was presented to an agent with only the protein metadata and the 21-residue sequence context — the UniProt evidence text (ground truth) was withheld from the agent at all times.

### The ESMC SAE Variant Interpreter Skill

The **With ESM SAE** condition relies on the skill defined in `variants_sae/skills/ESMC_SAE_variant_interpreter.md`. This skill gives the agent the ability to query the ESMC 6B Sparse Autoencoder for any missense variant and receive a ranked list of gained and lost SAE features at the mutation site.

**What it does, step by step:**

1. **Fetch the canonical protein sequence** from the UniProt REST API, given a UniProt accession or gene symbol. Validates that the reference amino acid in the variant notation matches the canonical sequence at the stated position.
2. **Build the mutant sequence** by applying the single amino-acid substitution.
3. **Run both sequences through the ESMC 6B SAE** via the EvolutionaryScale Forge API (`esmc-6b-2024-12` model, `esmc-6b-2024-12_k64_codebook16384_layer60` SAE, layer 60, top-*k* = 64 sparsity, codebook of 16,384 features). Returns a sparse activation tensor of shape (sequence length × 16,384) for each sequence.
4. **Compute per-feature deltas** over a ±8 residue window centred on the mutation position. For each SAE feature *f*:
   - `window_max_ref` / `window_max_mut` — maximum activation within the window in the reference/mutant sequence
   - `feature_loss = max(0, window_max_ref − window_max_mut)` — activation silenced by the mutation
   - `feature_gain = max(0, window_max_mut − window_max_ref)` — new activation introduced by the mutation
5. **Load feature labels** from `uniref90_feature_table.parquet` (16,384 rows; columns: `feature_id`, `category`, `summary`, `description`, `threshold`) and join them to the delta results.
6. **Return ranked tables** of the top-10 lost and top-10 gained SAE features, annotated with biological category and natural-language summary, plus the summed `LoF_functional_feature_loss_score` over seven functional categories (Catalytic function, Ligand-binding site, Interaction site, Structural motif, Domain, Post-translational modification, Sequence motif).

**Key practical notes:**
- Each call to the Forge API for a single variant requires 2 calls (ref + mut). If the reference protein is already cached in `results/activations/ref_cache/{accession}.npz`, the skill loads it directly and costs only 1 credit.
- Sequences longer than ~2,700 AA are not supported by ESMC 6B.
- The analysis is **local**: only the ±8 residue window around the mutation is examined. Long-range allosteric effects are not captured.
- SAE feature labels are derived from UniRef90 sequence patterns, not from per-protein expert annotation. A feature labelled "Catalytic function" activates broadly on residues that tend to be catalytic across many proteins — it is not a protein-specific active-site annotation.

### Experimental Setup

Two conditions were tested:

| Condition | Agent prompt | Tools / context available |
|---|---|---|
| **With ESM SAE** | `variants_sae/skills/variant_lof_mechanism.md` | Standard protein informatics tools (AlphaMissense, ThermoMPNN, AlphaFold, ESMC logit scores) **plus** ESM SAE per-position feature activations and the annotated feature table |
| **Without ESM SAE** | `variants_sae/skills/variant_lof_mechanism_no_ESMC_SAE.md` | Same standard tools, but **no** SAE feature access |

For each condition and each variant, a subagent was spawned with the variant metadata (accession, gene, protein name, ref/alt AA, position, 21-aa sequence context) and asked to explain the molecular mechanism of LoF. The orchestrating agent then evaluated the final explanation against the withheld UniProt evidence text on a scale of 1–5:

- **5** — fully captured the same mechanism of LoF
- **4** — correct mechanism, but missing one specific detail (e.g., a binding partner, downstream pathway)
- **3** — correct mechanistic class but wrong specific kinase/interactor
- **2** — partially correct structural reasoning, wrong functional consequence
- **1** — did not capture the mechanism

Results were logged per-variant and a blind pairwise judge (also an LLM) compared the two explanations side-by-side for each variant, without knowing which was which.

### Per-Variant Scores

| Variant | Mechanism | With SAE score | Without SAE score | Judge winner | Judge confidence |
|---|---|---|---|---|---|
| A0AV96\_F115A | RNA-binding LoF (RRM1 RNP1) | 4 | 4 | without\_ESM\_SAE | high |
| A1L167\_C88A | Catalytic LoF (UBC active-site Cys) | 5 | 4 | without\_ESM\_SAE | medium |
| A2RU30\_F186A | Binding LoF (IP3R-interacting domain) | 5 | 4 | without\_ESM\_SAE | low |
| A6ND36\_S610A | Regulatory LoF (phosphorylation in IDR) | 4 | 3 | with\_ESM\_SAE | medium |
| A7KAX9\_Y173A | Binding LoF (phospholipid binding) | 2 | 2 | with\_ESM\_SAE | medium |
| O00115\_N86Q | Regulatory LoF (N-glycosylation) | 5 | 5 | with\_ESM\_SAE | medium |
| O00115\_N212Q | Regulatory LoF (N-glycosylation) | 5 | 5 | without\_ESM\_SAE | medium |
| O00115\_N266Q | Regulatory LoF (N-glycosylation) | 5 | 5 | without\_ESM\_SAE | medium |
| O00115\_N290Q | Regulatory LoF (N-glycosylation) | 5 | 5 | with\_ESM\_SAE | low |
| O00115\_H295A | Catalytic LoF (DNase II active-site His) | 5 | 5 | without\_ESM\_SAE | medium |
| **Mean** | | **4.5** | **4.2** | | |

### Pairwise Judge Results

The blind judge assigned wins to: **without\_ESM\_SAE: 6/10**, **with\_ESM\_SAE: 4/10**.

> **Caveat:** Only 10 variants were evaluated and the judge confidence is often medium or low. These counts are insufficient to draw statistically meaningful conclusions; the results should be interpreted as exploratory signal only.

That said, some interpretable patterns emerge:

**Cases where SAE features helped:**
- **A6ND36\_S610A** (phosphorylation in a disordered tail): the without-SAE agent scored 3 vs. 4 for the with-SAE agent. PTM-related SAE features (Features 905, 10137, 12328, 11676 — all phosphorylation/Ser–Thr disordered IDR features) provided direct, quantitative evidence that the functional disruption was PTM-based, allowing the agent to reason more confidently about the mechanism despite an ALK3/BMP pathway context it could not fully identify from sequence alone.
- **A7KAX9\_Y173A** (phospholipid binding / cytoplasmic mislocalization): both agents scored only 2, but the with-SAE agent was preferred by the judge because it surfaced Feature 15254 (C2/phospholipid-binding beta-sandwich domains) as the top *gained* feature — a signal that the mutant is losing a phospholipid-binding module. The without-SAE agent never mentioned phospholipid binding at all, focusing exclusively on thermodynamic destabilization. This illustrates SAE features providing a mechanistic clue not available from sequence or structure predictors alone, even when the agent did not fully interpret it.

**Cases where SAE features made less difference:**
- **DNASE2 N-glycosylation variants (N86Q, N212Q, N266Q, N290Q):** both agents scored 5 on all four. SAE Feature 16076 (N-glycosylation sequon detector) was consistently and completely ablated in the with-SAE condition and gave a clean signal, but the without-SAE agent could identify N-X-S/T sequons from the sequence context alone using textbook biochemistry knowledge. The judge split 2–2 between conditions on these four variants with low-to-medium confidence.
- **Catalytic and structural mechanisms (A1L167\_C88A, A2RU30\_F186A, O00115\_H295A):** the without-SAE agent was preferred 3/3 times on these, often because it provided more specific structural detail (exact atomic distances, residue identities of interaction partners, precise sequon context) that the with-SAE agent sometimes substituted with SAE feature descriptions. This suggests SAE features are most complementary for PTM and localization mechanisms where structure-based reasoning alone is insufficient.

### Interpretation

The overall mean scores (4.5 with SAE, 4.2 without) and judge counts (4 vs. 6) are consistent with SAE features providing marginal average benefit at the mechanism-identification task, but with substantial case-to-case variation. The strongest signal is that SAE features appear most useful for **PTM and subcellular localization mechanisms** — cases where standard tools (structural stability predictors, AlphaMissense) provide weak or indirect evidence and the SAE has learned dedicated feature detectors (e.g., Feature 16076 for glycosylation sequons, Feature 15254 for phospholipid-binding domains). For well-understood catalytic mechanisms, expert biochemical reasoning from sequence context alone appears sufficient or even superior. Given the small sample size, these findings motivate a larger-scale evaluation across more diverse mechanism classes.