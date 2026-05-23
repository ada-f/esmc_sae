"""Step 11 / Validation 3 — within-position SAE drop at LOF hotspots (demonstration).

At each KRAS loss-of-function hotspot (a single high-ΔΔG position) we hold the
residue fixed and vary the substituted amino acid, then ask: do DMS-disruptive
substitutions perturb the SAE more than DMS-neutral substitutions at the SAME
position?

Score per substitution = mean of its top-3 feature drops over ALL 16,384 SAE
features — a global score. No feature selection from the test data, so the
comparison is NOT circular. One-sided Mann-Whitney U per hotspot and pooled.

  disruptive = ΔΔG in the top 10 % of the assay
  neutral    = |ΔΔG| ≤ 0.2

This is a demonstration: a finer-grained, position-controlled echo of
Validation 1.
For interpretation we also record, per hotspot, the 3 SAE features most
perturbed by mutations there (position_features.parquet) — descriptive only.

Outputs (results/validation_3/):
  <assay>_positions.png          per-assay within-position strip plot
  within_position_stats.parquet  per (assay, hotspot) MWU on the global score
  position_features.parquet      per (assay, hotspot) the 3 most-perturbed features
  summary.csv                    per-assay pooled MWU
"""

import json
import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.stats import mannwhitneyu

from config import DMS_MATRICES, FEATURE_TABLE, ARIAL, RESULTS

DMS = DMS_MATRICES
UNIREF = FEATURE_TABLE
OUT = RESULTS / 'validation_3'
OUT.mkdir(parents=True, exist_ok=True)

if ARIAL.exists():
    font_manager.fontManager.addfont(str(ARIAL))
    plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

DISRUPT_FRAC  = 0.10     # disruptive = top 10 % of ΔΔG within the assay
NEUTRAL_ABS   = 0.2      # neutral    = |ΔΔG| ≤ 0.2
MIN_PER_GROUP = 2        # min substitutions per group at a position
K_FEATURES    = 3        # # most-perturbed features recorded per hotspot
HOTSPOT_TOP_K = 20       # candidate hotspot positions per assay (by max ΔΔG)
SCORE_TOP_N   = 3        # global score = mean of top-3 drops over all features

ASSAYS = ['folding', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'DARPin_K27', 'DARPin_K55']


def load_inputs():
    T  = torch.load(DMS / 'kras_sae_tensor.pt',    map_location='cpu')
    WT = torch.load(DMS / 'kras_wt_sae_vector.pt', map_location='cpu')
    meta = json.loads((DMS / 'kras_sae_tensor_meta.json').read_text())
    feat_desc = pq.read_table(UNIREF).to_pandas().set_index('feature_id')
    return T, WT, meta, feat_desc


def compute_drop(T, WT):
    nan_mask = torch.isnan(T).any(dim=-1, keepdim=True)
    drop = torch.clamp(WT.view(1, 1, -1) - T, min=0.0)
    return torch.where(nan_mask.expand_as(drop), torch.full_like(drop, float('nan')), drop)


def assay_ddg_long(assay):
    m = pd.read_csv(DMS / f'kras_ddG_{assay}.csv', index_col=0).iloc[1:]
    m.index.name = 'mutant_aa'
    long = m.reset_index().melt(id_vars='mutant_aa', var_name='position',
                                value_name='ddG')
    long['position'] = long['position'].astype(int)
    long['ddG']      = pd.to_numeric(long['ddG'], errors='coerce')
    return long.dropna(subset=['ddG'])


def analyse_hotspot(drop, p, ddg_long, aa_rows, q_disrupt):
    """Per-hotspot result, or None if too few substitutions."""
    p_col = p - 2
    if not (0 <= p_col < drop.shape[1]):
        return None
    pos = ddg_long[ddg_long['position'] == p].set_index('mutant_aa')['ddG']
    disrupt_aa = set(pos[pos >= q_disrupt].index)
    neutral_aa = set(pos[pos.abs() <= NEUTRAL_ABS].index)
    if len(disrupt_aa) < MIN_PER_GROUP or len(neutral_aa) < MIN_PER_GROUP:
        return None

    D = drop[:, p_col, :]
    valid = ~torch.isnan(D).any(dim=1)
    if valid.sum().item() < 5:
        return None
    vi  = torch.nonzero(valid).squeeze(1).tolist()
    vaa = [aa_rows[i] for i in vi]
    Dv  = D[vi]                                          # (n_sub, 16384)

    # global score: mean of each substitution's top-3 drops over ALL features
    g = torch.topk(Dv, k=SCORE_TOP_N, dim=1).values.mean(dim=1).numpy()
    d = np.array([g[vaa.index(a)] for a in disrupt_aa if a in vaa])
    n = np.array([g[vaa.index(a)] for a in neutral_aa if a in vaa])
    if len(d) < MIN_PER_GROUP or len(n) < MIN_PER_GROUP:
        return None
    try:
        p_mwu = float(mannwhitneyu(d, n, alternative='greater').pvalue)
    except ValueError:
        p_mwu = float('nan')

    # descriptive: the 3 features most perturbed at this position
    R = torch.topk(Dv, k=min(3, Dv.shape[0]), dim=0).values.mean(dim=0)
    S = [int(x) for x in torch.topk(R, k=K_FEATURES).indices.numpy()]

    return {'position': int(p), 'disrupt': d, 'neutral': n,
            'p_mwu': p_mwu, 'features': S}


def plot_positions(assay, results, out):
    """Within-position strip plot: one column per hotspot, neutral vs
    disruptive, wide spacing; legend placed below the panel."""
    results = sorted(results, key=lambda r: r['position'])
    n = len(results)
    fig, ax = plt.subplots(figsize=(max(7, 1.05 * n), 5.2))
    rng = np.random.default_rng(0)
    for i, r in enumerate(results):
        for arr, dx, col in [(r['neutral'], -0.20, '#5a6f8a'),
                             (r['disrupt'], +0.20, '#a31300')]:
            xs = i + dx + (rng.random(len(arr)) - 0.5) * 0.22
            ax.scatter(xs, arr, s=26, alpha=0.8, color=col,
                       edgecolor='black', linewidths=0.3, zorder=3)
            ax.hlines(np.median(arr), i + dx - 0.16, i + dx + 0.16,
                      color='black', lw=2.0, zorder=4)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(r['position']) for r in results], fontsize=11)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xlabel('KRAS hotspot position', fontsize=13, weight='bold')
    ax.set_ylabel('SAE feature drop', fontsize=13)
    ax.set_title(assay, fontsize=14, weight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=11)
    # legend BELOW the panel so it never overlaps the points
    ax.scatter([], [], s=44, color='#5a6f8a', edgecolor='black', linewidths=0.3,
               label=f'Neutral substitution  (|ΔΔG| ≤ {NEUTRAL_ABS})')
    ax.scatter([], [], s=44, color='#a31300', edgecolor='black', linewidths=0.3,
               label=f'Disruptive substitution  (top {int(DISRUPT_FRAC*100)}% ΔΔG)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=11, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)


def main():
    T, WT, meta, feat_desc = load_inputs()
    aa_rows = meta['aa_rows']
    drop = compute_drop(T, WT)
    print(f'[load] T={tuple(T.shape)}  drop computed')

    stat_rows, feat_rows, summary = [], [], []
    for assay in ASSAYS:
        ddg = assay_ddg_long(assay)
        q_disrupt   = ddg['ddG'].quantile(1 - DISRUPT_FRAC)
        max_per_pos = ddg.groupby('position')['ddG'].max()
        hotspots    = max_per_pos.sort_values(ascending=False).head(HOTSPOT_TOP_K).index

        results = []
        for p in hotspots:
            r = analyse_hotspot(drop, int(p), ddg, aa_rows, q_disrupt)
            if r is not None:
                results.append(r)
        if not results:
            print(f'[skip] {assay}: no hotspot passed the filter'); continue

        pooled_d = np.concatenate([r['disrupt'] for r in results])
        pooled_n = np.concatenate([r['neutral'] for r in results])
        try:
            p_pool = float(mannwhitneyu(pooled_d, pooled_n,
                                        alternative='greater').pvalue)
        except ValueError:
            p_pool = float('nan')

        plot_positions(assay, results, OUT / f'{assay}_positions.png')
        print(f'[{assay:<11s}] hotspots={len(results):2d}  '
              f'pooled within-position MWU p = {p_pool:.2e}')

        for r in results:
            stat_rows.append({
                'assay': assay, 'position': r['position'],
                'n_disrupt': len(r['disrupt']), 'n_neutral': len(r['neutral']),
                'median_disrupt': float(np.median(r['disrupt'])),
                'median_neutral': float(np.median(r['neutral'])),
                'p_mwu': r['p_mwu'],
            })
            for fid in r['features']:
                info = feat_desc.loc[fid] if fid in feat_desc.index else None
                feat_rows.append({
                    'assay': assay, 'position': r['position'], 'feature_id': fid,
                    'category': (info['category'] if info is not None else None),
                    'summary':  (info['summary']  if info is not None else None)})
        summary.append({'assay': assay, 'n_hotspots': len(results),
                        'pooled_n_disrupt': len(pooled_d),
                        'pooled_n_neutral': len(pooled_n),
                        'pooled_mwu_p': p_pool})

    pd.DataFrame(stat_rows).to_parquet(OUT / 'within_position_stats.parquet')
    pd.DataFrame(feat_rows).to_parquet(OUT / 'position_features.parquet')
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    print(f'[save] within_position_stats.parquet  rows={len(stat_rows)}')
    print(f'[save] position_features.parquet      rows={len(feat_rows)}')
    print('[save] summary.csv')
    print('Done.')


if __name__ == '__main__':
    main()
