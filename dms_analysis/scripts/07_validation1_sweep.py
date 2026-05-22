"""Step 07 / Validation 1 — grid sweep over Validation 1 settings.

Sweeps:
  - feature aggregation: max (K=1), top-3 mean, top-5 mean, top-10 mean
  - neutral band:        |ΔΔG| ≤ {0.05, 0.10, 0.20, 0.50}
  - disrupt cutoff:      fixed at top-5 % of ΔΔG (per assay)

For each (assay, agg, neutral) cell we report:
  n_neutral, n_disrupt
  median(neutral), median(disrupt)
  median_shift              = median(disrupt) − median(neutral)
  relative_shift            = median_shift / median(neutral)
  AUC                       = area under the ROC curve of disrupt-vs-neutral
  Mann-Whitney U p-value (one-sided)

Writes:
  results/validation_1_sweep/sweep.csv           every combination, every assay
  results/validation_1_sweep/sweep_summary.csv   per (agg, neutral) averaged over assays
  results/validation_1_sweep/heatmap_<assay>.png small heatmap per assay
"""

import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import mannwhitneyu

from config import DMS_MATRICES, ARIAL, RESULTS

DMS = DMS_MATRICES
OUT = RESULTS / 'validation_1_sweep'
OUT.mkdir(parents=True, exist_ok=True)

if ARIAL.exists():
    font_manager.fontManager.addfont(str(ARIAL))
    plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

ASSAYS = ['folding', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'DARPin_K27', 'DARPin_K55']
AGG_OPTS   = [('max', 1), ('top3', 3), ('top5', 5), ('top10', 10)]
NEUTRAL_OPTS = [0.05, 0.10, 0.20, 0.50]
DISRUPT_FRAC = 0.05


def load_tensors():
    T  = torch.load(DMS / 'kras_sae_tensor.pt',    map_location='cpu')
    WT = torch.load(DMS / 'kras_wt_sae_vector.pt', map_location='cpu')
    meta = json.loads((DMS / 'kras_sae_tensor_meta.json').read_text())
    return T, WT, meta


def compute_drop(T, WT):
    nan_mask = torch.isnan(T).any(dim=-1, keepdim=True)
    raw_diff = WT.view(1, 1, -1) - T
    drop = torch.clamp(raw_diff, min=0.0)
    drop = torch.where(nan_mask.expand_as(drop), torch.full_like(drop, float('nan')), drop)
    return drop


def variant_aggregate(drop, k):
    """Per (a,p) variant: mean of top-k drops (k=1 ≡ max). Returns (20,187)."""
    out = torch.full(drop.shape[:2], float('nan'), dtype=torch.float32)
    for a in range(drop.shape[0]):
        for p in range(drop.shape[1]):
            v = drop[a, p]
            if torch.isnan(v).any():
                continue
            topk = torch.topk(v, k=k).values
            out[a, p] = topk.mean() if k > 1 else topk[0]
    return out


def assay_ddg(assay):
    m = pd.read_csv(DMS / f'kras_ddG_{assay}.csv', index_col=0)
    m = m.iloc[1:]
    m.index.name = 'mutant_aa'
    long = m.reset_index().melt(id_vars='mutant_aa', var_name='position', value_name='ddG')
    long['position'] = long['position'].astype(int)
    long['ddG']      = pd.to_numeric(long['ddG'], errors='coerce')
    return long.dropna(subset=['ddG'])


def auc(disrupt, neutral):
    """AUC of disrupt > neutral; equivalent to MWU U / (n1*n2)."""
    if len(disrupt) == 0 or len(neutral) == 0:
        return float('nan')
    try:
        U, _ = mannwhitneyu(disrupt, neutral, alternative='greater')
        return float(U / (len(disrupt) * len(neutral)))
    except ValueError:
        return float('nan')


def main():
    T, WT, meta = load_tensors()
    drop = compute_drop(T, WT)
    AA_ROWS = meta['aa_rows']
    POSITIONS = meta['positions']

    rows = []
    for agg_name, k in AGG_OPTS:
        print(f'[agg] {agg_name} (k={k}) — computing per-variant statistic')
        g = variant_aggregate(drop, k).numpy()                 # (20, 187)
        # flatten to long format
        long_g = []
        for a_i, aa in enumerate(AA_ROWS):
            for p_i, p in enumerate(POSITIONS):
                v = float(g[a_i, p_i])
                if not np.isnan(v):
                    long_g.append((aa, p, v))
        g_df = pd.DataFrame(long_g, columns=['mutant_aa', 'position', 'agg_score'])

        for assay in ASSAYS:
            ddg = assay_ddg(assay)
            m = ddg.merge(g_df, on=['mutant_aa', 'position'])
            q_disrupt = m['ddG'].quantile(1 - DISRUPT_FRAC)
            for n_abs in NEUTRAL_OPTS:
                neu  = m[m['ddG'].abs() <= n_abs]['agg_score']
                dis  = m[m['ddG'] >= q_disrupt]['agg_score']
                if len(neu) < 5 or len(dis) < 5:
                    continue
                try:
                    _, p_val = mannwhitneyu(dis, neu, alternative='greater')
                except ValueError:
                    p_val = float('nan')
                med_n = float(neu.median()); med_d = float(dis.median())
                shift = med_d - med_n
                rel_shift = shift / med_n if med_n else float('nan')
                rows.append({
                    'assay':         assay,
                    'aggregation':   agg_name,
                    'k':             k,
                    'neutral_abs':   n_abs,
                    'q_disrupt':     float(q_disrupt),
                    'n_neutral':     int(len(neu)),
                    'n_disrupt':     int(len(dis)),
                    'median_neutral':med_n,
                    'median_disrupt':med_d,
                    'median_shift':  shift,
                    'rel_shift':     rel_shift,
                    'auc':           auc(dis, neu),
                    'p_mwu_greater': float(p_val),
                })

    sw = pd.DataFrame(rows)
    sw.to_csv(OUT / 'sweep.csv', index=False)
    print(f'[save] sweep.csv  rows={len(sw)}')

    # Per (agg, neutral) summary across assays
    summary = sw.groupby(['aggregation', 'k', 'neutral_abs']).agg(
        mean_median_shift =('median_shift', 'mean'),
        mean_rel_shift    =('rel_shift',    'mean'),
        mean_auc          =('auc',          'mean'),
        worst_auc         =('auc',          'min'),
        max_log10_p       =('p_mwu_greater',
                             lambda s: float(np.max(-np.log10(s.replace(0, 1e-300)))))).reset_index()
    summary.to_csv(OUT / 'sweep_summary.csv', index=False)
    print(f'[save] sweep_summary.csv  rows={len(summary)}')
    print('\nPer (agg, neutral) summary across assays:')
    print(summary.to_string(index=False))

    # heatmaps: rows = agg, cols = neutral; value = median_shift (per assay)
    for assay in ASSAYS:
        sub = sw[sw['assay'] == assay].copy()
        pivot_shift = sub.pivot(index='aggregation', columns='neutral_abs', values='median_shift')
        pivot_auc   = sub.pivot(index='aggregation', columns='neutral_abs', values='auc')
        # reorder rows by k
        order = [a for a, _ in AGG_OPTS]
        pivot_shift = pivot_shift.reindex(order)
        pivot_auc   = pivot_auc.reindex(order)
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, mat, label, fmt in [
                (axs[0], pivot_shift, 'median(disrupt) − median(neutral)', '.3f'),
                (axs[1], pivot_auc,   'AUC (disrupt > neutral)',           '.2f'),
        ]:
            im = ax.imshow(mat.values, aspect='auto', cmap='magma', origin='upper')
            ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns)
            ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index)
            ax.set_xlabel('neutral |ΔΔG| ≤')
            ax.set_ylabel('aggregation')
            ax.set_title(label, fontsize=10)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat.iloc[i, j]
                    if pd.notna(val):
                        ax.text(j, i, f'{val:{fmt}}', ha='center', va='center',
                                color='white' if val < (mat.values[~pd.isna(mat.values)].mean()) else 'black',
                                fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        fig.suptitle(f'{assay}', fontsize=12, weight='bold')
        plt.tight_layout()
        plt.savefig(OUT / f'heatmap_{assay}.png', dpi=160, bbox_inches='tight')
        plt.close(fig)
    print('[save] one heatmap PNG per assay')
    print('Done.')


if __name__ == '__main__':
    main()
