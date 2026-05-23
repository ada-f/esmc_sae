"""Step 06 / Validation 1 — figures: SAE drop for disruptive vs neutral substitutions.

For each substitution the global score is the mean of its K largest feature
drops (K = 1 is the single max). One 2×4 panel grid is produced per K in
{1, 3, 10}: each panel compares the score of disruptive substitutions
(ΔΔG in the top 5% of the assay) against neutral ones (|ΔΔG| ≤ 0.10), with a
one-sided Mann–Whitney U p-value. Positive ΔΔG destabilises, so disruptive is
the top tail.

Output: results/validation_1/aggregation_<K>.png
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
OUT = RESULTS / 'validation_1'
OUT.mkdir(parents=True, exist_ok=True)

if ARIAL.exists():
    font_manager.fontManager.addfont(str(ARIAL))
    plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

ASSAYS       = ['folding', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'DARPin_K27', 'DARPin_K55']
# aggregations: mean of the top-K feature drops (K = 1 is the single max)
AGG_OPTS     = [('max', 1, 'max feature drop'),
                ('top3', 3, 'mean of top-3 feature drops'),
                ('top10', 10, 'mean of top-10 feature drops')]
NEUTRAL_ABS  = 0.10
DISRUPT_FRAC = 0.05
YLABEL       = 'SAE feature drop'


def load_inputs():
    T  = torch.load(DMS / 'kras_sae_tensor.pt',    map_location='cpu')
    WT = torch.load(DMS / 'kras_wt_sae_vector.pt', map_location='cpu')
    meta = json.loads((DMS / 'kras_sae_tensor_meta.json').read_text())
    return T, WT, meta


def compute_drop(T, WT):
    """drop_j = max(0, activation_WT,j − activation_mutant,j); NaN at WT cells."""
    nan_mask = torch.isnan(T).any(dim=-1, keepdim=True)
    drop = torch.clamp(WT.view(1, 1, -1) - T, min=0.0)
    drop = torch.where(nan_mask.expand_as(drop), torch.full_like(drop, float('nan')), drop)
    return drop


def variant_aggregate(drop, k):
    """topK_drop per variant = mean of the K largest feature drops."""
    out = torch.full(drop.shape[:2], float('nan'), dtype=torch.float32)
    for a in range(drop.shape[0]):
        for p in range(drop.shape[1]):
            v = drop[a, p]
            if torch.isnan(v).any():
                continue
            out[a, p] = torch.topk(v, k=k).values.mean()
    return out


def assay_ddg(assay):
    m = pd.read_csv(DMS / f'kras_ddG_{assay}.csv', index_col=0)
    m = m.iloc[1:]
    m.index.name = 'mutant_aa'
    long = m.reset_index().melt(id_vars='mutant_aa', var_name='position', value_name='ddG')
    long['position'] = long['position'].astype(int)
    long['ddG']      = pd.to_numeric(long['ddG'], errors='coerce')
    return long.dropna(subset=['ddG'])


def long_g(g_arr, aa_rows, positions):
    rows = []
    for a_i, aa in enumerate(aa_rows):
        for p_i, p in enumerate(positions):
            v = float(g_arr[a_i, p_i])
            if not np.isnan(v):
                rows.append((aa, p, v))
    return pd.DataFrame(rows, columns=['mutant_aa', 'position', 'agg_score'])


def plot_one_aggregation(g_df, out_path):
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.8))
    axs = axs.flatten()
    rng = np.random.default_rng(0)

    for idx, assay in enumerate(ASSAYS):
        ax  = axs[idx]
        ddg = assay_ddg(assay)
        m   = ddg.merge(g_df, on=['mutant_aa', 'position'])
        # disruptive = most-disruptive 5% = top 5% of ΔΔG (positive = destabilising)
        q_disrupt = m['ddG'].quantile(1 - DISRUPT_FRAC)
        neu = m[m['ddG'].abs() <= NEUTRAL_ABS]['agg_score'].to_numpy()
        dis = m[m['ddG'] >= q_disrupt]['agg_score'].to_numpy()
        data = [neu, dis]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color='black', linewidth=1.4))
        for box, fc in zip(bp['boxes'], ['#dfe5ee', '#f7c7c0']):
            box.set_facecolor(fc); box.set_edgecolor('#333')
        for i, (arr, color) in enumerate(zip(data, ['#5a6f8a', '#a31300']), start=1):
            x = i + (rng.random(len(arr)) - 0.5) * 0.30
            ax.scatter(x, arr, s=4, alpha=0.45, color=color, linewidths=0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels([f'Neutral\n(n = {len(neu)})',
                            f'Disruptive\n(n = {len(dis)})'], fontsize=13)
        # panel title = assay name + one-sided MWU p (disruptive > neutral);
        # the description of the analysis itself lives in the report.
        try:
            _, p_val = mannwhitneyu(dis, neu, alternative='greater')
        except ValueError:
            p_val = float('nan')
        p_str = f'$p$ = {p_val:.1e}' if np.isfinite(p_val) else '$p$ = n.a.'
        ax.set_title(f'{assay}  ({p_str})', fontsize=13, weight='bold', pad=6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=12)

    axs[-1].axis('off')                       # 8th cell blank (7 assays)
    # ONE shared y-axis label; no figure title — descriptions live in the report.
    fig.tight_layout(rect=[0.055, 0, 1, 1])
    fig.supylabel(YLABEL, fontsize=15, weight='bold', x=0.035)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    T, WT, meta = load_inputs()
    drop = compute_drop(T, WT)
    aa_rows, positions = meta['aa_rows'], meta['positions']
    print(f'[load] T={tuple(T.shape)}  drop ready')

    for agg_name, k, desc in AGG_OPTS:
        print(f'[agg] {agg_name} (k={k})')
        g_df = long_g(variant_aggregate(drop, k).numpy(), aa_rows, positions)
        out  = OUT / f'aggregation_{agg_name}.png'
        plot_one_aggregation(g_df, out)
        print(f'  [save] {out.name}')
    print('Done.')


if __name__ == '__main__':
    main()
