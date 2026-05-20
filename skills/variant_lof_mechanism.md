# Variant LoF Mechanism Interpreter

Given a variant, propose the mechanism by which it causes loss of function using AI structural tools — without reading any curated experimental annotation.

---

## Inputs — fill these in before running

```
VARIANT_ID   = ""   # {uniprot_accession}_{ref_aa}{position}{alt_aa}
ACCESSION    = ""   # UniProt accession
GENE         = ""   # gene symbol
PROTEIN_NAME = ""   # full protein name
REF_AA       = ""   # reference amino acid (single letter)
POSITION     = 0    # mutation position (1-indexed)
ALT_AA       = ""   # alternate amino acid (single letter)
SEQ_CONTEXT  = ""   # 21-aa window centred on the mutated residue
```

Parse the mutation string if needed:
```python
import re
accession, mutation_str = VARIANT_ID.rsplit("_", 1)
m = re.match(r"([A-Z])(\d+)([A-Z])", mutation_str)
ref_aa, position, alt_aa = m.group(1), int(m.group(2)), m.group(3)
```

---

## Constraints — what you must NOT use

* Web search or literature (Google, PubMed, bioRxiv)
* UniProt annotation / mutagenesis / variant pages — fetching the FASTA sequence for model input is fine, reading curated functional annotations is not
* PDB entries annotated with this variant's effect
* The dataset file `/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ESM-ATLAS/variants_sae/data/variants_with_evidence_text/positive_tier1_with_evidence_text.tsv` — it contains ground truth

---

## Step 1 — Domain context with ToolUniverse (required)

Use PfamTool to find which Pfam domain covers the mutation position.

**Use `/usr/bin/python3.11` and `sys.path.append` (not `insert`) to avoid venv stdlib conflicts:**

```python
import sys
sys.path.append('/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/lib/python3.11/site-packages')
from tooluniverse.pfam_tool import PfamTool

tool    = PfamTool(tool_config={"fields": {"endpoint": "get_protein_pfam"}})
result  = tool.run({"accession": ACCESSION})
domains = result.get("data", {}).get("domains", [])

covering = [d for d in domains if d["start"] <= POSITION <= d["end"]]
print("Domains covering position", POSITION, ":", covering)
```

---

## Step 2 — AlphaMissense pathogenicity + AlphaFold structure confidence (required)

### 2a — AlphaMissense pathogenicity score

Use `/usr/bin/python3.11` and `sys.path.append`:

```python
import sys
sys.path.append('/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ai_scientists/.venv/lib/python3.11/site-packages')
from tooluniverse.alphamissense_tool import AlphaMissenseTool

tool   = AlphaMissenseTool(tool_config={"fields": {"operation": "get_variant_score"}})
result = tool.run({"uniprot_id": ACCESSION, "variant": f"p.{REF_AA}{POSITION}{ALT_AA}"})
raw    = result["data"]["raw_response"]
print("mean_all (position intolerance):", raw["mean_all"])
print("pathogenic_all:", raw["pathogenic_all"])
print("benign_all:",     raw["benign_all"])
```

Key fields in `raw_response`:
- `mean_all` — mean pathogenicity across all 19 substitutions at this position (0–1; >0.564 = pathogenic)
- `pathogenic_all` / `benign_all` / `ambiguous_all` — count:AA-list for ALL substitutions
- `pathogenic` / `benign` / `ambiguous` — classification of this specific ALT_AA only

### 2b — AlphaFold per-residue pLDDT confidence

Use `/n/holylabs/LABS/mzitnik_lab/Users/afang/ESM-atlas/esm/.pixi/envs/default/bin/python`:

```python
import requests

r    = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{ACCESSION}", timeout=30)
pred = r.json()[0]
print("Global pLDDT:", pred["globalMetricValue"])

r2   = requests.get(pred["plddtDocUrl"], timeout=30)
conf = r2.json()
plddt_at_site = conf["confidenceScore"][POSITION - 1]  # 1-indexed → 0-indexed
print(f"pLDDT at position {POSITION} ({REF_AA}→{ALT_AA}): {plddt_at_site:.1f}")
# >90 = very high confidence (well-structured); 70–90 = confident; 50–70 = low; <50 = disordered
```

---

## Step 3 — ESMC zero-shot logit score (required)

The masked-marginals protocol used in ProteinGym: mask the mutation position, run ESMC forward, compute `log P(mut | context) − log P(wt | context)`. Positive = mutation is more evolutionarily plausible; negative = less plausible / predicted LoF.

Use `/n/holylabs/LABS/mzitnik_lab/Users/afang/ESM-atlas/esm/.pixi/envs/default/bin/python`:

```python
import torch, attr
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

model = ESMC.from_pretrained("esmc_300m").to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Build amino-acid → token index mapping
aa_to_token = {}
for aa in "ACDEFGHIKLMNPQRSTVWY":
    tok = model.encode(ESMProtein(sequence=aa))
    aa_to_token[aa] = tok.sequence[1].item()  # skip BOS

# Mask position and score
def esmc_logit_score(model, sequence, ref_aa, alt_aa, position_1idx):
    protein        = ESMProtein(sequence=sequence)
    protein_tensor = model.encode(protein)
    token_pos      = position_1idx  # +1 for BOS, but encode already gives 1-indexed tokens
    masked         = protein_tensor.sequence.clone()
    mask_id        = getattr(model, "mask_token_id", 32)
    masked[token_pos] = mask_id
    masked_tensor  = attr.evolve(protein_tensor, sequence=masked)
    logits         = model.logits(masked_tensor, LogitsConfig(sequence=True)).logits.sequence
    log_probs      = torch.log_softmax(logits[0, token_pos], dim=-1)
    score          = (log_probs[aa_to_token[alt_aa]] - log_probs[aa_to_token[ref_aa]]).item()
    return score

score = esmc_logit_score(model, ref_sequence, REF_AA, ALT_AA, POSITION)
print(f"ESMC logit score ({REF_AA}{POSITION}{ALT_AA}): {score:.4f}")
# score < -1.0: strong evolutionary signal against this substitution (predicted LoF)
# score ≈  0.0: neutral
# score > 0.0: mutation more plausible than wt at this position
```

Reference implementation (ProteinGym):
`/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ProteinGym/proteingym/baselines/evoscale/compute_fitness.py`
(function `_score_mutations_common`)

---

## Step 4 — ProteinMPNN / SoluMPNN / ThermoMPNN (required)

All three tools require a protein structure. **Download the AlphaFold structure** using the `pdbUrl` from Step 2b:

```python
import requests, tempfile, os

pdb_path = f"/tmp/{ACCESSION}_AF.pdb"
if not os.path.exists(pdb_path):
    r = requests.get(pred["pdbUrl"], timeout=60)
    with open(pdb_path, "wb") as f:
        f.write(r.content)
print("AlphaFold PDB saved to:", pdb_path)
```

### 4a — ProteinMPNN / SoluMPNN

Scores sequence likelihood given backbone structure. Delta NLL = NLL(mut) − NLL(wt): positive = mutation structurally incompatible = destabilising.

Use `/n/holylabs/LABS/mzitnik_lab/Users/afang/ESM-atlas/esm/.pixi/envs/default/bin/python`:

```python
import sys, copy, torch
import numpy as np
sys.path.append('/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ProteinMPNN')
from protein_mpnn_utils import parse_PDB, tied_featurize, _scores, ProteinMPNN

ALPHABET      = 'ACDEFGHIKLMNPQRSTVWYX'
alphabet_dict = {aa: i for i, aa in enumerate(ALPHABET)}
device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_pmpnn(weights_path):
    ckpt  = torch.load(weights_path, map_location="cpu")
    model = ProteinMPNN(ca_only=False, num_letters=21, node_features=128, edge_features=128,
                        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
                        k_neighbors=ckpt["num_edges"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.eval().to(device)

def score_sequence(model, pdb_path, sequence):
    pdb_dict = parse_PDB(pdb_path, ca_only=False)
    all_chains = [k[-1:] for k in pdb_dict[0] if k[:9] == "seq_chain"]
    chain_id_dict = {pdb_dict[0]["name"]: (all_chains, [])}
    batch = [copy.deepcopy(pdb_dict[0])]
    X, S, mask, *_, chain_M, chain_encoding_all, _, _, _, _, chain_M_pos, _, residue_idx, *_ = \
        tied_featurize(batch, device, chain_id_dict, None, None, None, None, None, ca_only=False)
    S_in = torch.tensor([alphabet_dict[aa] for aa in sequence], device=device)[None]
    S[:, :S_in.shape[1]] = S_in
    with torch.no_grad():
        log_probs = model(X, S, mask, chain_M * chain_M_pos, residue_idx, chain_encoding_all,
                          torch.zeros_like(chain_M))
    return _scores(S, log_probs, mask * chain_M * chain_M_pos).mean().item()

# --- Vanilla ProteinMPNN ---
vanilla_weights = '/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ProteinMPNN/vanilla_model_weights/v_48_020.pt'
model_vanilla   = load_pmpnn(vanilla_weights)
nll_ref_vanilla = score_sequence(model_vanilla, pdb_path, ref_sequence)
nll_mut_vanilla = score_sequence(model_vanilla, pdb_path, mut_sequence)
print(f"ProteinMPNN  ΔlogP (mut−wt): {nll_mut_vanilla - nll_ref_vanilla:+.4f}")

# --- SoluMPNN (trained on soluble proteins) ---
solu_weights    = '/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ProteinMPNN/soluble_model_weights/v_48_020.pt'
model_solu      = load_pmpnn(solu_weights)
nll_ref_solu    = score_sequence(model_solu, pdb_path, ref_sequence)
nll_mut_solu    = score_sequence(model_solu, pdb_path, mut_sequence)
print(f"SoluMPNN     ΔlogP (mut−wt): {nll_mut_solu - nll_ref_solu:+.4f}")
# Positive delta = mutation less likely given structure = structurally destabilising
```

### 4b — ThermoMPNN

Directly predicts ΔΔG (kcal/mol) for every single-point mutation at once. Positive ddG_pred = destabilising; negative = stabilising.

Before running, ensure `local.yaml` in the ThermoMPNN directory has `platform.thermompnn_dir` pointing to the correct path:

```bash
# Check / patch local.yaml
python -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ThermoMPNN/local.yaml')
cfg.platform.thermompnn_dir = '/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ThermoMPNN'
OmegaConf.save(cfg, '/n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ThermoMPNN/local.yaml')
print('thermompnn_dir set to:', cfg.platform.thermompnn_dir)
"
```

Then run inference (produces all single-point variants at once; filter to your mutation afterward):

```bash
python /n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ThermoMPNN/analysis/custom_inference.py \
  --pdb    /tmp/${ACCESSION}_AF.pdb \
  --chain  A \
  --model_path /n/holylabs/LABS/mzitnik_lab/Users/afang/clawmind/ClawInstitute/ThermoMPNN/models/thermoMPNN_default.pt \
  --out_dir /tmp/thermompnn_${ACCESSION}/
```

Filter result to the specific variant:

```python
import pandas as pd
df = pd.read_csv(f"/tmp/thermompnn_{ACCESSION}/ThermoMPNN_inference_{ACCESSION}_AF.csv")
row = df[(df["wildtype"] == REF_AA) & (df["position"] == POSITION - 1) & (df["mutation"] == ALT_AA)]
ddg = row["ddG_pred"].values[0]
print(f"ThermoMPNN ddG_pred ({REF_AA}{POSITION}{ALT_AA}): {ddg:+.3f} kcal/mol")
# > +1.0: significantly destabilising (structural stability LoF)
# ≈  0.0: neutral — LoF likely functional, not structural
# < -1.0: stabilising — LoF mechanism is direct functional disruption
```

Note: ThermoMPNN uses 0-indexed positions internally. The output `position` column is 0-indexed (subtract 1 from your 1-indexed POSITION when filtering).

---

## Step 5 — ESM SAE analysis (required)

Read the skill file and follow every step:
`/n/holylabs/LABS/mzitnik_lab/Users/afang/ESM-atlas/variants_sae/skills/ESMC_SAE_variant_interpreter.md`

Python environment: `/n/holylabs/LABS/mzitnik_lab/Users/afang/ESM-atlas/esm/.pixi/envs/default/bin/python`

`ESM_API_KEY` is already set in the environment.

Check for a cached ref activation before calling the API (saves 1 credit):
```python
import os
cache_path = f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ESM-ATLAS/variants_sae/results/activations/ref_cache/{ACCESSION}.npz"
# If the file exists, load it with load_npz_cache(); otherwise call run_sae() for the ref
```

---

## Expected output

1. **Domain annotation** — which Pfam domain covers the mutation site, residue range, domain function
2. **AlphaMissense score** — `mean_all` position intolerance, classification of this specific substitution, fraction pathogenic/benign across all substitutions
3. **AlphaFold pLDDT** — per-residue confidence at the mutation site, structural context (well-folded vs. disordered loop)
4. **ESMC logit score** — masked-marginals log P(mut) − log P(wt); interpret as evolutionary fitness change
5. **ProteinMPNN / SoluMPNN ΔlogP** — structural compatibility delta; positive = destabilising
6. **ThermoMPNN ddG_pred** — predicted stability change (kcal/mol); positive = destabilising
7. **SAE top lost features** — feature ID, category, loss score, full summary for top 10
8. **SAE top gained features** — feature ID, category, gain score, full summary for top 5
9. **Mechanistic conclusion** — grounded explicitly in:
   - the domain the residue is in (from Step 1)
   - the AlphaMissense pathogenicity and structural confidence (from Step 2)
   - the ESMC logit score (evolutionary plausibility, from Step 3)
   - the ProteinMPNN/ThermoMPNN structural scores (from Step 4): if ddG_pred > +1 and ΔlogP > 0 → structural stability LoF; if scores near zero → direct functional disruption
   - the specific SAE features lost (cite feature IDs and their summaries)
   - the specific SAE features gained (disorder, alternative scaffold, etc.)
   - classify the mechanism: RNA-binding LoF / catalytic LoF / metal-binding LoF / structural stability LoF / regulatory (PTM) LoF / other
