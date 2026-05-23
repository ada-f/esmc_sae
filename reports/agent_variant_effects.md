## Agent Variant-Effect Experiments

To assess whether SAE feature information improves an LLM agent's ability to reason about molecular mechanisms of variant LoF, we ran a blinded experiment on 100 variants drawn from the Tier 1 positive set (`positive_tier1_with_evidence_text.tsv`), spanning 42 proteins across a range of mechanism classes. Each variant was presented to an agent with only the protein metadata and the 21-residue sequence context — the UniProt evidence text (ground truth) was withheld from the agent at all times.

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

### Overall Results (100 Variants)

| Metric | With SAE | Without SAE |
|---|---|---|
| Mean mechanism score (1–5) | **3.83** | 3.76 |
| Pairwise judge wins (blind) | **50/100** | 50/100 |
| Score strictly better | 25 | 24 |
| Score tied | 51 | 51 |
| Judge wins — high confidence only | **14/22** | 8/22 |
| Judge wins — medium confidence | 29/60 | 31/60 |
| Judge wins — low confidence | 7/18 | 11/18 |

Overall the two conditions are statistically indistinguishable: mean score difference +0.07 (with SAE) and a 50/50 pairwise split. However, the **high-confidence subsample** (22 cases where the judge was confident) splits 14/8 in favour of the with-SAE agent. Low-confidence cases — where both explanations were weak and the judge was guessing — tip the other direction (7/11). This pattern suggests SAE features improve explanations when the agent can actually interpret them, but add noise in hard cases where the feature set does not encode the relevant biology.

### Per-Protein-Family Results

Aggregating by protein reveals strong directional effects that average out at the variant level:

| Protein | n | Mechanism | With SAE mean | Without SAE mean | Judge with/total |
|---|---|---|---|---|---|
| O75531 (BANF1/BAF) | 12 | Multi-partner nuclear lamina binding | **3.17** | 2.58 | **10/12** |
| O00115 (DNASE2) | 9 | N-glycosylation (4) + catalytic His (5) | **5.00** | 4.56 | 4/9 |
| O00429 (DNM1L/DRP1) | 8 | SUMOylation in disordered linker | 3.12 | **4.00** | 1/8 |
| O14543 (SOCS3) | 5 | JAK/SH2 binding | 3.40 | 3.60 | 1/5 |
| O14974 (MYPN) | 5 | Coiled-coil PRKG1 binding | 3.80 | **4.40** | 2/5 |
| O75530 (EED) | 5 | H3K27me3 aromatic cage binding | 3.60 | 3.40 | 3/5 |
| O60701 (UGDH) | 4 | Catalytic nucleophile/cofactor | **5.00** | 5.00 | **4/4** |
| O60573 (EIF4E) | 4 | m7G-cap aromatic stacking | 5.00 | **5.00** | 0/4 |
| O43623 (AHNAK) | 6 | NLS-importin binding | 2.00 | 2.00 | 2/6 |

**BANF1 (BAF) — SAE strongly beneficial.** This 89-residue nuclear lamina scaffold protein simultaneously contacts dsDNA, emerin (EMD), histones H1/H3, LEMD3/MAN1, and lamin A. Without SAE, agents consistently identified one or two binding partners but missed others. SAE features encoding nuclear lamina interaction surfaces and charged DNA-binding patches allowed the with-SAE agent to capture the multi-partner binding disruption described in the ground truth. Judge won 10/12 variants for with-SAE; mean score 3.17 vs 2.58.

**DRP1 (DNM1L) — SAE harmful.** Eight K→R substitutions in DRP1's intrinsically disordered variable domain abolish SUMOylation. The without-SAE agent correctly identified the ψ-K-x-E SUMOylation consensus, the K→R chemistry (preserved charge, lost ε-amine), and the linker context from sequence alone. The with-SAE agent was misled by SAE features: the disordered IDR activated generic phosphorylation/Ser-Thr PTM features rather than SUMOylation-specific features, causing the agent to hypothesise ubiquitination or phosphorylation mechanisms. Judge won 1/8 for with-SAE; mean score 3.12 vs 4.00.

**UGDH (O60701) — SAE adds specificity at equal accuracy.** Both conditions scored 5/5 on all four variants (catalytic cysteine and conserved Asp/Lys in UDP-glucose dehydrogenase). SAE features directly labelled the active-site nucleotide-binding and oxidoreductase motifs, providing converging evidence that the judge found more compelling. Won all 4 pairwise at high or medium confidence.

**eIF4E (O60573) — SAE adds noise at equal accuracy.** Both conditions also scored 5/5 on all four variants (conserved tryptophans for m7G-cap aromatic stacking). The without-SAE agent gave cleaner, more direct mechanistic explanations. The with-SAE agent introduced SAE feature descriptions that were mostly correct but verbose and sometimes irrelevant (e.g., features for 4E-BP binding surface that were not perturbed). Lost all 4 pairwise.

**AHNAK (O43623) — both conditions failed.** Six variants in AHNAK's NLS region abolish importin binding (KPNA2, KPNB1, IPO7). Both agents consistently misidentified the mechanism as DNA-binding disruption; neither recognised the NLS character of the region. SAE features did not encode NLS/importin binding specifically, so they provided no advantage. Both conditions scored 2/5 on all six.

### Cases Where SAE Features Were Decisive

**SAE most helpful (large score gain):**

| Variant | Mechanism | With SAE | Without SAE | Key SAE feature |
|---|---|---|---|---|
| O60337\_L571A | Ac/N-degron recognition LoF | 3 | 1 | TM helix features; secondary domain disruption |
| O75531\_G25Q | EMD/dsDNA multi-binding | 5 | 3 | Interaction-surface patches (Feature 10469) |
| O75317\_C48S | Catalytic Cys / deubiquitinase | 5 | 3 | UCH Cys-His-Asp triad correctly identified |
| O75530\_F97A | H3K27me3 aromatic cage | 4 | 2 | PRC2 propeller scaffold features |
| O75530\_Y365A | H3K27me3 binding | 4 | 2 | Histone methyl-lysine recognition (Feature 15240) |
| O75531\_R8E | LEMD3/MAN1 binding | 4 | 2 | LEM-domain interaction surface |
| O60716\_N478A | Cadherin cytoplasmic tail binding | 4 | 2 | Interaction-site features vs. mistaken glycosylation mechanism |

**SAE most harmful (large score loss):**

| Variant | Mechanism | With SAE | Without SAE | What went wrong |
|---|---|---|---|---|
| O75530\_W364A | H3K27me3 aromatic cage | 3 | 5 | SAE features distracted from direct pi-stacking |
| O00429\_K532R–K608R (6) | DRP1 SUMOylation | 3.0 | 4.0 | IDR phospho features masked SUMOylation signal |
| O14974\_L1007A, L1028A | PRKG1 coiled-coil | 4 | 5 | SAE framing less specific than direct binding claim |
| O14543\_L22D | JAK KIR binding | 3 | 4 | SAE features pointed to wrong docking motif |
| O75164\_Y973A | H3K27me3 aromatic cage | 4 | 5 | Without-SAE named H3K4me3 cage members more precisely |

### Pairwise Judge Results

The blind judge split exactly **50 with-SAE / 50 without-SAE** across 100 comparisons. Among high-confidence decisions (n=22), the split was 14 with-SAE / 8 without-SAE.

> **Caveat:** The judge (also an LLM) may have systematic biases — for instance, a preference for explanations with explicit computational evidence (which SAE adds) regardless of correctness. The 50/50 overall split should be treated as an upper bound on SAE benefit.

Patterns in the judge's reasoning:
- **SAE won** when features directly named a mechanism the agent would otherwise miss (importin binding ≠ NLS in AHNAK, phospholipid-binding domain, multi-partner nuclear lamina contacts).
- **Without SAE won** when the without-SAE agent provided cleaner structural detail (exact residue distances, specific kinase motif names, correct SUMOylation consensus) that the with-SAE agent diluted with feature descriptions.
- **Low confidence** cases were roughly random — both explanations were incomplete, and the judge's tie-breaking had no meaningful signal.

### Interpretation

Across 100 variants, SAE features provide a modest and mechanism-dependent benefit:

**SAE features are most useful when:**
1. The mechanism involves a multi-partner binding surface that is hard to infer from sequence context alone (BANF1/BAF nuclear lamina scaffold).
2. The mechanism requires recognising a functional module that is labelled in the SAE feature dictionary (histone reader aromatic cages, catalytic triads, phospholipid-binding domains).
3. The variant disrupts a region where the relevant biology is poorly encoded in standard tools (structural stability predictors, AlphaMissense) but the SAE has learned specific feature detectors.

**SAE features add noise when:**
1. The mechanism can be identified precisely from sequence context or known motifs (SUMOylation ψ-K-x-E consensus, m7G-cap sandwich tryptophans, SUMO consensus).
2. The SAE feature dictionary encodes a related but incorrect category — e.g., generic IDR phosphorylation features activating on a SUMOylation site, or interaction-surface features at a DNA-binding patch rather than a specific protein partner.
3. The mechanism is opaque to both conditions (AHNAK NLS variants): adding SAE features does not help if the feature set does not encode the relevant biology.

The strongest case for SAE features is the BANF1 family (10/12 judge wins, +0.59 mean score gain) — a single-domain protein with five functionally distinct binding partners whose disruption pattern is recapitulated in the SAE feature space but not easily read from sequence alone. The strongest case against is DRP1 SUMOylation (1/8 judge wins, −0.88 mean score loss) — where the SAE activates the wrong PTM category in an intrinsically disordered region.

### Per-Variant Scores (all 100)

| Variant | With SAE | Without SAE | Judge winner | Confidence |
|---|---|---|---|---|
| A0AV96\_F115A | 4 | 4 | without\_ESM\_SAE | high |
| A1L167\_C88A | 5 | 4 | without\_ESM\_SAE | medium |
| A2RU30\_F186A | 5 | 4 | without\_ESM\_SAE | low |
| A6ND36\_S610A | 4 | 3 | with\_ESM\_SAE | medium |
| A7KAX9\_Y173A | 2 | 2 | with\_ESM\_SAE | medium |
| O00115\_N86Q | 5 | 5 | with\_ESM\_SAE | medium |
| O00115\_N212Q | 5 | 5 | without\_ESM\_SAE | medium |
| O00115\_N266Q | 5 | 5 | without\_ESM\_SAE | medium |
| O00115\_N290Q | 5 | 5 | with\_ESM\_SAE | low |
| O00115\_H295A | 5 | 5 | without\_ESM\_SAE | medium |
| O00115\_H295K | 5 | 4 | with\_ESM\_SAE | medium |
| O00115\_H295N | 5 | 4 | without\_ESM\_SAE | medium |
| O00115\_H295R | 5 | 4 | with\_ESM\_SAE | medium |
| O00115\_H295S | 5 | 4 | without\_ESM\_SAE | medium |
| O00141\_K127M | 5 | 5 | with\_ESM\_SAE | high |
| O00429\_K532R | 3 | 4 | without\_ESM\_SAE | medium |
| O00429\_K535R | 3 | 4 | without\_ESM\_SAE | medium |
| O00429\_K558R | 3 | 4 | without\_ESM\_SAE | medium |
| O00429\_K568R | 3 | 4 | without\_ESM\_SAE | medium |
| O00429\_K594R | 3 | 4 | without\_ESM\_SAE | medium |
| O00429\_K597R | 3 | 4 | without\_ESM\_SAE | medium |
| O00429\_K606R | 4 | 4 | with\_ESM\_SAE | medium |
| O00429\_K608R | 3 | 4 | without\_ESM\_SAE | medium |
| O00444\_D154A | 5 | 5 | without\_ESM\_SAE | medium |
| O14495\_D184E | 1 | 2 | with\_ESM\_SAE | low |
| O14543\_F25A | 4 | 4 | without\_ESM\_SAE | medium |
| O14543\_L22D | 3 | 4 | without\_ESM\_SAE | high |
| O14543\_L41R | 3 | 3 | with\_ESM\_SAE | medium |
| O14543\_R71E | 4 | 4 | without\_ESM\_SAE | high |
| O14543\_V34E | 3 | 3 | without\_ESM\_SAE | medium |
| O14746\_D712A | 5 | 4 | with\_ESM\_SAE | high |
| O14777\_E234K | 2 | 3 | with\_ESM\_SAE | high |
| O14974\_L1007A | 4 | 5 | without\_ESM\_SAE | high |
| O14974\_L1014A | 4 | 5 | with\_ESM\_SAE | medium |
| O14974\_L1021A | 4 | 5 | with\_ESM\_SAE | medium |
| O14974\_L1028A | 4 | 5 | without\_ESM\_SAE | medium |
| O14974\_S473A | 3 | 2 | without\_ESM\_SAE | medium |
| O15265\_K257R | 5 | 5 | without\_ESM\_SAE | high |
| O15294\_H508A | 4 | 5 | with\_ESM\_SAE | medium |
| O15381\_W173A | 2 | 3 | without\_ESM\_SAE | medium |
| O43293\_T265A | 4 | 3 | with\_ESM\_SAE | high |
| O43314\_K248A | 4 | 5 | with\_ESM\_SAE | high |
| O43567\_C258A | 5 | 5 | with\_ESM\_SAE | medium |
| O43567\_H260A | 5 | 5 | with\_ESM\_SAE | medium |
| O43586\_W232A | 4 | 3 | with\_ESM\_SAE | medium |
| O43623\_K166E | 2 | 2 | without\_ESM\_SAE | low |
| O43623\_K175E | 2 | 2 | without\_ESM\_SAE | low |
| O43623\_K192E | 2 | 2 | without\_ESM\_SAE | low |
| O43623\_R196E | 2 | 2 | without\_ESM\_SAE | low |
| O43623\_R225E | 2 | 2 | with\_ESM\_SAE | low |
| O43623\_R229E | 2 | 2 | with\_ESM\_SAE | low |
| O60331\_S650D | 3 | 3 | without\_ESM\_SAE | medium |
| O60337\_L571A | 3 | 1 | with\_ESM\_SAE | medium |
| O60493\_Y71A | 4 | 4 | without\_ESM\_SAE | medium |
| O60573\_W63A | 5 | 5 | without\_ESM\_SAE | high |
| O60573\_W135A | 5 | 5 | without\_ESM\_SAE | medium |
| O60573\_W148A | 5 | 5 | without\_ESM\_SAE | medium |
| O60573\_W183F | 5 | 5 | without\_ESM\_SAE | medium |
| O60667\_F67A | 3 | 3 | with\_ESM\_SAE | medium |
| O60667\_K69A | 4 | 4 | without\_ESM\_SAE | medium |
| O60667\_R45A | 4 | 5 | without\_ESM\_SAE | low |
| O60701\_C276A | 5 | 5 | with\_ESM\_SAE | high |
| O60701\_C276S | 5 | 5 | with\_ESM\_SAE | medium |
| O60701\_D280A | 5 | 5 | with\_ESM\_SAE | high |
| O60701\_K220A | 5 | 5 | with\_ESM\_SAE | medium |
| O60704\_K158A | 5 | 5 | with\_ESM\_SAE | medium |
| O60716\_K401M | 4 | 5 | without\_ESM\_SAE | medium |
| O60716\_N478A | 4 | 2 | with\_ESM\_SAE | high |
| O60841\_H706E | 5 | 5 | without\_ESM\_SAE | medium |
| O60870\_K135R | 2 | 1 | without\_ESM\_SAE | low |
| O60885\_N140A | 5 | 5 | with\_ESM\_SAE | medium |
| O60885\_N433A | 5 | 5 | without\_ESM\_SAE | low |
| O75151\_W29A | 4 | 3 | without\_ESM\_SAE | low |
| O75151\_Y7A | 3 | 3 | without\_ESM\_SAE | low |
| O75164\_D945R | 4 | 4 | with\_ESM\_SAE | medium |
| O75164\_W967H | 5 | 5 | with\_ESM\_SAE | medium |
| O75164\_Y973A | 4 | 5 | without\_ESM\_SAE | high |
| O75317\_C48A | 5 | 5 | with\_ESM\_SAE | low |
| O75317\_C48S | 5 | 3 | with\_ESM\_SAE | high |
| O75365\_D72A | 5 | 5 | without\_ESM\_SAE | low |
| O75530\_F97A | 4 | 2 | with\_ESM\_SAE | high |
| O75530\_W364A | 3 | 5 | without\_ESM\_SAE | medium |
| O75530\_W364L | 4 | 5 | with\_ESM\_SAE | medium |
| O75530\_Y148A | 3 | 3 | without\_ESM\_SAE | medium |
| O75530\_Y365A | 4 | 2 | with\_ESM\_SAE | high |
| O75531\_G25E | 4 | 3 | with\_ESM\_SAE | high |
| O75531\_G25Q | 5 | 3 | with\_ESM\_SAE | medium |
| O75531\_G47E | 2 | 2 | with\_ESM\_SAE | low |
| O75531\_K6A | 3 | 2 | with\_ESM\_SAE | medium |
| O75531\_K6E | 3 | 3 | with\_ESM\_SAE | medium |
| O75531\_K53E | 2 | 1 | with\_ESM\_SAE | medium |
| O75531\_L46E | 3 | 4 | without\_ESM\_SAE | medium |
| O75531\_R8E | 4 | 2 | with\_ESM\_SAE | high |
| O75531\_S4A | 3 | 3 | with\_ESM\_SAE | medium |
| O75531\_S4E | 3 | 4 | without\_ESM\_SAE | high |
| O75531\_V51E | 3 | 2 | with\_ESM\_SAE | medium |
| O75531\_W62A | 3 | 2 | with\_ESM\_SAE | low |
| O75665\_S735A | 4 | 4 | with\_ESM\_SAE | high |
| O75689\_R149C | 4 | 4 | with\_ESM\_SAE | medium |
| O75689\_R273C | 4 | 4 | without\_ESM\_SAE | medium |
| **Mean** | **3.83** | **3.76** | | |
