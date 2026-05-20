# Skill: ESMC SAE Variant Interpreter

## Purpose

Given a protein identifier and a missense variant, this skill retrieves the canonical protein sequence, runs both the reference and mutant sequences through the ESM Cambrian (ESMC) 6B Sparse Autoencoder (SAE) via the EvolutionaryScale Forge API, and interprets which SAE features are gained or lost at the mutation site. The output is a mechanistic interpretation of the variant grounded in the SAE feature labels.

---

## Prerequisites

- **Python environment:** `/n/holylabs/mzitnik_lab/Users/afang/ESM-atlas/esm/.pixi/envs/default/bin/python`
- **ESM_API_KEY** environment variable set to a valid EvolutionaryScale Forge token
- **Feature label table:** `/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ESM-ATLAS/variants_sae/data/uniref90_feature_table.parquet`
  - 16,384 rows, one per SAE feature
  - Key columns: `feature_id`, `category`, `summary`, `description`, `threshold`
- **Daily credit limit:** The Forge API has a limit of 10 credits/day. Each API call (ref or mut sequence) consumes credits. Check remaining credits before running.

---

## Inputs

| Input | Format | Example |
|---|---|---|
| Protein identifier | UniProt accession or gene symbol | `P04637` or `TP53` |
| Variant | Single-letter AA codes: `{ref_aa}{position_1idx}{alt_aa}` | `R175H` |

---

## Step 1: Retrieve the canonical protein sequence

Use the UniProt REST API to fetch the canonical sequence for the protein.

```python
import requests

def get_uniprot_sequence(accession: str) -> str:
    """Fetch canonical sequence for a UniProt accession."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["sequence"]["value"]

def resolve_gene_to_accession(gene_symbol: str) -> str:
    """Resolve a human gene symbol to the best reviewed UniProt accession."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
        "fields": "accession,gene_names,protein_name",
        "format": "json",
        "size": 1,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])
    if not results:
        raise ValueError(f"No reviewed human UniProt entry found for gene: {gene_symbol}")
    return results[0]["primaryAccession"]
```

**Validate the variant:** confirm that `sequence[position_1idx - 1] == ref_aa`. If it does not match, the variant coordinates may be on a different isoform or the accession is wrong.

```python
def validate_variant(sequence: str, ref_aa: str, position_1idx: int) -> bool:
    pos_0idx = position_1idx - 1
    if pos_0idx < 0 or pos_0idx >= len(sequence):
        return False
    return sequence[pos_0idx] == ref_aa
```

---

## Step 2: Build the mutant sequence

```python
def apply_variant(sequence: str, ref_aa: str, alt_aa: str, position_1idx: int) -> str:
    pos_0idx = position_1idx - 1
    assert sequence[pos_0idx] == ref_aa, "ref_aa mismatch"
    return sequence[:pos_0idx] + alt_aa + sequence[pos_0idx + 1:]
```

---

## Step 3: Run both sequences through the ESMC SAE

Use `ESMCForgeInferenceClient` with `LogitsConfig(sae_config=SAEConfig(...))`. The correct model names are:
- ESM model: `esmc-6b-2024-12`
- SAE model: `esmc-6b-2024-12_k64_codebook16384_layer60`

```python
import os
import torch
from esm.sdk.api import ESMProtein, SAEConfig, LogitsConfig
from esm.sdk.forge import ESMCForgeInferenceClient

def run_sae(client, sae_config, logits_config, sequence: str) -> torch.Tensor:
    """
    Run a sequence through the SAE. Returns a sparse COO tensor of shape
    (seq_len, 16384) with BOS/EOS tokens removed.
    """
    protein = ESMProtein(sequence=sequence)
    protein_tensor = client.encode(protein)
    output = client.logits(protein_tensor, config=logits_config, return_bytes=False)
    t = output.sae_outputs[sae_config.model]
    # Remove BOS (index 0) and EOS (index -1) tokens
    indices = t.coalesce().indices()
    values  = t.coalesce().values()
    mask = (indices[0] > 0) & (indices[0] < t.shape[0] - 1)
    new_indices = indices[:, mask].clone()
    new_indices[0] -= 1  # shift to 0-indexed residue positions
    return torch.sparse_coo_tensor(
        new_indices, values[mask],
        size=(t.shape[0] - 2, t.shape[1])
    ).coalesce()

# Initialise client
client = ESMCForgeInferenceClient(
    model="esmc-6b-2024-12",
    url="https://forge.evolutionaryscale.ai",
    token=os.environ["ESM_API_KEY"],
)
sae_config    = SAEConfig(model="esmc-6b-2024-12_k64_codebook16384_layer60", normalize_features=True)
logits_config = LogitsConfig(sae_config=sae_config)

ref_t = run_sae(client, sae_config, logits_config, ref_sequence)
mut_t = run_sae(client, sae_config, logits_config, mut_sequence)
```

---

## Step 4: Compute per-feature deltas at the mutation site and local window

Use a ±8 residue window around the mutation site (the primary window from the POC analysis). Also compute the site-only value for precision.

```python
import numpy as np

def dense_row(sparse_t: torch.Tensor, row: int) -> np.ndarray:
    """Extract a single residue's activations as a dense array."""
    t = sparse_t.coalesce()
    idx = t.indices()
    val = t.values()
    mask = idx[0] == row
    out = np.zeros(sparse_t.shape[1], dtype=np.float32)
    out[idx[1, mask].numpy()] = val[mask].numpy()
    return out

def window_max(sparse_t: torch.Tensor, center: int, half_width: int) -> np.ndarray:
    """Max activation across a residue window for each feature."""
    L = sparse_t.shape[0]
    lo = max(0, center - half_width)
    hi = min(L, center + half_width + 1)
    t = sparse_t.coalesce()
    idx = t.indices()
    val = t.values()
    mask = (idx[0] >= lo) & (idx[0] < hi)
    out = np.zeros(sparse_t.shape[1], dtype=np.float32)
    for f, v in zip(idx[1, mask].numpy(), val[mask].numpy()):
        if v > out[f]:
            out[f] = v
    return out

pos_0idx = position_1idx - 1
WINDOW   = 8

ref_site       = dense_row(ref_t, pos_0idx)
mut_site       = dense_row(mut_t, pos_0idx)
ref_window_max = window_max(ref_t, pos_0idx, WINDOW)
mut_window_max = window_max(mut_t, pos_0idx, WINDOW)

delta_site       = mut_site - ref_site
delta_window_max = mut_window_max - ref_window_max
feature_loss     = np.maximum(0, ref_window_max - mut_window_max)
feature_gain     = np.maximum(0, mut_window_max - ref_window_max)
```

---

## Step 5: Load feature labels and rank disrupted features

```python
import pandas as pd

FEATURE_TABLE = "/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ESM-ATLAS/variants_sae/data/uniref90_feature_table.parquet"
features = pd.read_parquet(FEATURE_TABLE)

# Build result dataframe
n_features = ref_window_max.shape[0]
results = pd.DataFrame({
    "feature_id":      np.arange(n_features),
    "ref_site":        ref_site,
    "mut_site":        mut_site,
    "delta_site":      delta_site,
    "ref_window_max":  ref_window_max,
    "mut_window_max":  mut_window_max,
    "delta_window_max":delta_window_max,
    "feature_loss":    feature_loss,
    "feature_gain":    feature_gain,
})
results = results.merge(
    features[["feature_id", "category", "summary", "threshold"]],
    on="feature_id", how="left"
)

# Top lost and gained features
top_lost  = results.nlargest(10, "feature_loss")[
    ["feature_id", "category", "summary", "ref_window_max", "mut_window_max", "feature_loss"]
]
top_gained = results.nlargest(10, "feature_gain")[
    ["feature_id", "category", "summary", "ref_window_max", "mut_window_max", "feature_gain"]
]

# Functional categories only
FUNCTIONAL = {
    "Catalytic function", "Ligand-binding site", "Interaction site",
    "Structural motif", "Domain", "Post-translational modification", "Sequence motif",
}
functional_loss = results[results["category"].isin(FUNCTIONAL)].nlargest(10, "feature_loss")
```

---

## Step 6: Interpret the results

For each top lost feature, report:
- `feature_id` — the SAE feature index (0–16383)
- `category` — high-level functional class (e.g. `Catalytic function`, `Ligand-binding site`)
- `summary` — one-sentence natural language description of what the feature encodes
- `ref_window_max` → `mut_window_max` — how much the feature activation changed
- `feature_loss` — the magnitude of loss (positive = ref was higher)

**Interpretation template:**

```
Variant {ref_aa}{position_1idx}{alt_aa} in protein {accession} ({gene}):

The mutation site is at position {position_1idx} (0-indexed: {pos_0idx}).
Reference AA: {ref_aa} → Mutant AA: {alt_aa}

Top lost SAE features (±8 residue window):
  Feature {id} [{category}]: {summary}
    ref={ref_window_max:.3f} → mut={mut_window_max:.3f}, loss={feature_loss:.3f}
  ...

Top gained SAE features:
  Feature {id} [{category}]: {summary}
    ref={ref_window_max:.3f} → mut={mut_window_max:.3f}, gain={feature_gain:.3f}
  ...

Functional feature loss score (sum over functional categories):
  {sum of feature_loss for features in FUNCTIONAL categories:.4f}

Mechanistic interpretation:
  If the top lost features are labeled Catalytic function or Ligand-binding site,
  the variant likely disrupts a functionally critical residue.
  If losses are concentrated in Structural motif or Domain, the effect may be
  structural rather than directly catalytic.
  Gain of Disorder features may indicate local unfolding.
  If no functional features are strongly lost, the variant may be tolerated or
  act through a non-local mechanism not captured by this local window analysis.
```

---

## Notes and limitations

- **Protein length limit:** ESMC 6B handles sequences up to ~2,700 AA in practice. Longer sequences will fail or be truncated.
- **Local window only:** This analysis uses a ±8 residue window. Long-range allosteric effects will not be captured.
- **Feature labels are descriptive, not curated:** The `uniref90_feature_table.parquet` labels are derived from UniRef90 patterns, not from human expert annotation of specific residues. A feature labeled `Catalytic function` activates on residues that tend to be catalytic across many proteins — it is not a per-protein active-site annotation.
- **Credit limit:** Each call to the Forge API costs credits. A single variant analysis requires exactly 2 API calls (ref + mut). If the protein's ref has already been cached in `results/activations/ref_cache/{accession}.npz`, load it directly to save 1 credit.
- **Sparse output:** The SAE produces sparse activations (k=64 active features per residue). Most of the 16,384 features will be zero at any given position. Only features above the per-feature `threshold` are considered reliably active.

---

## Loading a cached ref sequence (optional, saves 1 API call)

```python
import torch
import numpy as np

def load_npz_cache(path: str) -> torch.Tensor:
    # Cache format: positions (nnz,), features (nnz,), values (nnz,), shape (2,)
    d = np.load(path)
    indices = torch.from_numpy(np.stack([d["positions"], d["features"]]))  # shape (2, nnz)
    values  = torch.from_numpy(d["values"])   # shape (nnz,)
    size    = tuple(d["shape"])
    return torch.sparse_coo_tensor(indices, values, size=size).coalesce()

cache_path = f"/n/holylfs06/LABS/mzitnik_lab/Lab/afang/ESM-ATLAS/variants_sae/results/activations/ref_cache/{accession}.npz"
if os.path.exists(cache_path):
    ref_t = load_npz_cache(cache_path)
else:
    ref_t = run_sae(client, sae_config, logits_config, ref_sequence)
```
