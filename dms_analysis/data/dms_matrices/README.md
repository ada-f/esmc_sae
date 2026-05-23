# KRAS DMS matrices (re-exported from Weng et al. 2024, Nature)

Source: supplementary Excel files of [Weng et al. *Nature* 2024](https://www.nature.com/articles/s41586-023-06954-0).

- MOESM5 → per-variant **fitness** (raw measurement-level)
- MOESM6 → per-substitution **inferred ΔΔG** (from a three-state MoCHI thermodynamic model)

All matrices share the same heatmap layout:
- one row per amino acid (20 rows, alphabetical: A, C, D, …, Y)
- one column per KRAS canonical position (positions 2..188 — the construct skips Met1)
- the first row (`WT_aa`) is the wild-type amino acid letter at each position
- cells are NaN if not measured / not confident; WT cells are NaN

The KRAS WT sequence (188 aa, UniProt P01116-2, KRAS 4B isoform) is in `kras_wt.fasta`.

## Files

### ΔΔG matrices (kcal/mol; positive = LOF)
| File | MOESM6 assay |
|---|---|
| `kras_ddG_folding.csv` | folding |
| `kras_ddG_RAF1.csv` | RAF1 (RBD) |
| `kras_ddG_PIK3CG.csv` | PIK3CG (RBD) |
| `kras_ddG_RALGDS.csv` | RALGDS (RBD) |
| `kras_ddG_SOS1.csv` | SOS1 (GEF) |
| `kras_ddG_DARPin_K27.csv` | DARPin K27 |
| `kras_ddG_DARPin_K55.csv` | DARPin K55 |
| `kras_ddG_full_length_RAF1.csv` | full-length RAF1 (positions 2–64 only — partial-coverage validation) |

### Fitness matrices (paper's `fitness` column; positive = better growth = no LOF)
| File | MOESM5 assay |
|---|---|
| `kras_fitness_folding.csv` | AbundancePCA |
| `kras_fitness_RAF1.csv` | BindingPCA RAF1RBD |
| `kras_fitness_PIK3CG.csv` | BindingPCA PIK3CGRBD |
| `kras_fitness_RALGDS.csv` | BindingPCA RALGDSRBD |
| `kras_fitness_SOS1.csv` | BindingPCA SOS1 ⚠ |
| `kras_fitness_DARPin_K27.csv` | BindingPCA DARPin K27 ⚠ |
| `kras_fitness_DARPin_K55.csv` | BindingPCA DARPin K55 ⚠ |
| `kras_fitness_full_length_RAF1.csv` | BindingPCA full length RAF1 (positions 2–64 only) |
| `kras_fitness_RAF1RBD_coex_GAP.csv` | BindingPCA RAF1RBD coexpression GAP (block 1 only) |

## ⚠ Known data issue in MOESM5 — SOS1 / K27 / K55 fitness are duplicates

In MOESM5 (the published Excel), the `fitness` column for the three assays `BindingPCA SOS1`, `BindingPCA DARPin K27`, and `BindingPCA DARPin K55` are **byte-identical** (same `aa_seq`, same `fitness`, n=2855 single mutants, mean=−0.4061, min=−1.7312, max=0.7400 for all three). This is a source-file artefact, not a re-export bug.

**For these three partners, use the ΔΔG matrices** (`kras_ddG_SOS1.csv`, `kras_ddG_DARPin_K27.csv`, `kras_ddG_DARPin_K55.csv`) — the MoCHI ΔΔG inferences are genuinely different per partner and form the basis of the paper's Fig 1g / 2 / 3 analyses.

## Reading example (Python)

```python
import pandas as pd
m = pd.read_csv('kras_ddG_RAF1.csv', index_col=0)
print(m.loc['A', '38'])     # ΔΔG of D38A on RAF1 binding (kcal/mol)
```

## Reading example (R)

```r
m <- read.csv('kras_ddG_RAF1.csv', row.names = 1, check.names = FALSE)
m['A', '38']                 # ΔΔG of D38A on RAF1 binding
```

## SAE feature tensors (one cell ↔ one mutant variant)

In addition to the ΔΔG / fitness CSVs, the same heatmap layout is materialised
as 3-D tensors of ESM-C SAE pooled features:

| File | Shape | Description |
|---|---|---|
| `kras_sae_tensor.pt` | (20, 187, 16384) | `sae[a, p, f]` = pooled SAE activation of feature `f` on the KRAS_{p→a} mutant (one per DMS cell) |
| `kras_sae_tensor_diff.pt` | (20, 187, 16384) | `sae[a, p, f] − sae_WT[f]` — WT-subtracted version of the same tensor |
| `kras_wt_sae_vector.pt` | (16384,) | pooled SAE feature vector of WT KRAS (the reference subtracted above) |
| `kras_sae_tensor_meta.json` | — | axis / index documentation, WT sequence, indexing keys |

Axes (same axis-0 / axis-1 as the DMS CSVs):
- axis 0 — mutant amino acid; 20 entries, alphabetical (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)
- axis 1 — KRAS canonical position 2..188 (187 entries; column `c` ↔ position `c + 2`)
- axis 2 — SAE feature 0..16383

WT cells (`mutant_aa == WT_aa(position)`) are `NaN` in both tensors — consistent
with the CSVs. Stop-codon substitutions (`*`) are not represented (the 20-AA
grid only).

### Reading example (Python / PyTorch)

```python
import torch, json
T = torch.load('kras_sae_tensor.pt', map_location='cpu')        # (20, 187, 16384)
D = torch.load('kras_sae_tensor_diff.pt', map_location='cpu')   # (20, 187, 16384)
WT = torch.load('kras_wt_sae_vector.pt', map_location='cpu')    # (16384,)
meta = json.load(open('kras_sae_tensor_meta.json'))
aa_idx  = meta['aa_rows'].index('A')        # 0
pos_idx = 38 - 2                            # canonical pos 38
print(T[aa_idx, pos_idx])                   # D38A's SAE vector
print(D[aa_idx, pos_idx].abs().sum())       # L1 distance to WT
```
