"""Step 05 / Validation 1 — global SAE perturbation vs DMS category, per assay.

Per single amino-acid substitution (a, p):
    drop[a,p,f]      = max(0, sae_WT[f] − sae[a,p,f])    # lost activation only
    global_drop[a,p] = mean(top-K of drop[a,p,:])        # default K = 3

For each assay, every confident substitution is binned into:
    disruptive : ΔΔG in the top DISRUPT_FRAC of the assay
    neutral    : |ΔΔG| ≤ NEUTRAL_ABS  (tight band around zero)

A one-sided Mann–Whitney U test reports whether disruptive substitutions have
larger global drops than neutral ones.

This script computes the variant table and per-assay statistics; the figures
are drawn by 06_validation1_aggregations.py.

Outputs:
    results/validation_1/global_drop.parquet         (variant-level table)
    results/validation_1/category_stats.csv          (per-assay U statistics)
"""

import json
import numpy as np
import pandas as pd
import torch
from scipy.stats import mannwhitneyu

from config import DMS_MATRICES, RESULTS

DMS = DMS_MATRICES
OUT = RESULTS / 'validation_1'
OUT.mkdir(parents=True, exist_ok=True)

ASSAYS = ['folding', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'DARPin_K27', 'DARPin_K55']
TOP_K_DROPS  = 3           # how many top-feature drops we average per variant
DISRUPT_FRAC = 0.05        # disruptive = top-5 % of ΔΔG within the assay
NEUTRAL_ABS  = 0.1         # neutral = |ΔΔG| ≤ 0.1 (tight band around zero)


def load_tensors():
    T  = torch.load(DMS / 'kras_sae_tensor.pt',      map_location='cpu')      # (20, 187, 16384)
    WT = torch.load(DMS / 'kras_wt_sae_vector.pt',   map_location='cpu')      # (16384,)
    meta = json.loads((DMS / 'kras_sae_tensor_meta.json').read_text())
    aa_rows   = meta['aa_rows']
    positions = meta['positions']
    return T, WT, aa_rows, positions


def compute_drop(T, WT):
    """drop[a,p,f] = max(0, WT[f] − sae[a,p,f]); NaN at WT cells preserved."""
    nan_mask = torch.isnan(T).any(dim=-1, keepdim=True)               # (20, 187, 1)
    raw_diff = WT.view(1, 1, -1) - T
    drop = torch.clamp(raw_diff, min=0.0)
    drop = torch.where(nan_mask.expand_as(drop), torch.full_like(drop, float('nan')), drop)
    return drop                                                       # (20, 187, 16384)


def variant_global_drop(drop, k=TOP_K_DROPS):
    """For each (a, p) variant return mean of its top-K feature drops.
    Returns a (20, 187) tensor of float; NaN at WT cells.
    """
    n_aa, n_pos, n_feat = drop.shape
    out = torch.full((n_aa, n_pos), float('nan'), dtype=torch.float32)
    for a in range(n_aa):
        for p in range(n_pos):
            v = drop[a, p]
            if torch.isnan(v).any():
                continue
            topk = torch.topk(v, k=k).values
            out[a, p] = topk.mean()
    return out                                                        # (20, 187)


def assay_ddg(assay):
    """Return long-form DataFrame (mutant_aa, position, ddG) for an assay."""
    path = DMS / f'kras_ddG_{assay}.csv'
    m = pd.read_csv(path, index_col=0)
    m = m.iloc[1:]                                                    # drop the WT_aa row
    m.index.name = 'mutant_aa'
    long = m.reset_index().melt(id_vars='mutant_aa', var_name='position', value_name='ddG')
    long['position'] = long['position'].astype(int)
    long = long.dropna(subset=['ddG'])
    long['ddG'] = pd.to_numeric(long['ddG'], errors='coerce')
    long = long.dropna(subset=['ddG'])
    return long


def main():
    T, WT, aa_rows, positions = load_tensors()
    print(f'[load] T={tuple(T.shape)}  WT={tuple(WT.shape)}')
    drop = compute_drop(T, WT)
    g_drop = variant_global_drop(drop)
    print(f'[stat] global_drop  observed={(~torch.isnan(g_drop)).sum().item()} variants')

    # ---------- variant-level table for export ----------
    rows = []
    for a_i, aa in enumerate(aa_rows):
        for p_i, p in enumerate(positions):
            v = g_drop[a_i, p_i].item()
            if not np.isnan(v):
                rows.append({'mutant_aa': aa, 'position': p, 'global_drop': v})
    g_df = pd.DataFrame(rows)
    g_df.to_parquet(OUT / 'global_drop.parquet')
    print(f'[save] global_drop.parquet  rows={len(g_df)}')

    # ---------- per-assay disruptive-vs-neutral statistics ----------
    summary = []
    for assay in ASSAYS:
        ddg = assay_ddg(assay)
        m = ddg.merge(g_df, on=['mutant_aa', 'position'])
        if m.empty:
            print(f'[skip] {assay}: no overlap'); continue
        q_disrupt  = m['ddG'].quantile(1 - DISRUPT_FRAC)
        neutral    = m[m['ddG'].abs() <= NEUTRAL_ABS]['global_drop']
        disruptive = m[m['ddG'] >= q_disrupt]['global_drop']
        try:
            U, p_val = mannwhitneyu(disruptive, neutral, alternative='greater')
        except ValueError:
            U, p_val = float('nan'), float('nan')
        summary.append({
            'assay':           assay,
            'n_neutral':       len(neutral),
            'n_disruptive':    len(disruptive),
            'median_neutral':  float(neutral.median()),
            'median_disrupt':  float(disruptive.median()),
            'U':               float(U),
            'p_mwu_greater':   float(p_val),
            'neutral_band':    f'|ΔΔG| ≤ {NEUTRAL_ABS}',
            'disrupt_cutoff':  float(q_disrupt),
            'disrupt_fraction':DISRUPT_FRAC,
        })
        print(f'[stat] {assay:>11s}  '
              f'p={p_val:.2e}  median(neutral)={neutral.median():.3f}  '
              f'median(disrupt)={disruptive.median():.3f}')

    pd.DataFrame(summary).to_csv(OUT / 'category_stats.csv', index=False)
    print(f'[save] category_stats.csv  rows={len(summary)}')
    print('Done.')


if __name__ == '__main__':
    main()
