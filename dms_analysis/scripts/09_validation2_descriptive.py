"""Step 09 / Validation 2 (descriptive) -- most-dropped SAE features per LOF hotspot cluster.

Computes the *descriptive* (no significance test) feature table that feeds the
descriptive callout figure (10_validation2_plot_callouts.py, mode='descriptive').

Pipeline (same hotspot clustering as 08_validation2_hotspot_enrichment.py):
  - top-20 ddG positions per assay -> clusters (gap <= 2) -> top-6 by max ddG.
  - per (position, feature): max_drop[p,f] = max over alleles of
        max(0, sae_WT[f] - sae[a,p,f]).
  - per cluster: rank features by mean(max_drop) across the cluster positions;
    keep the top 5.

Outputs (in results/validation_2/):
  descriptive_features.parquet    top-5 features per cluster (no test)
  descriptive.txt                 human-readable
"""

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq

from config import MOESM6, SAE_TENSOR_DIFF, FEATURE_TABLE, RESULTS
from dms_common import (V2_ASSAYS as ASSAYS, V2_SKIP as SKIP, TOP_K_CANDIDATES,
                        N_CLUSTERS, CLUSTER_GAP, cluster_positions, cluster_label,
                        lof_table, max_drop_per_position)

SAE_DIFF = SAE_TENSOR_DIFF
FEAT_TBL = FEATURE_TABLE
OUT_DIR = RESULTS / 'validation_2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_DESC = 5         # features listed per cluster (matches the tested top-5)


def main():
    print(f'[load] {MOESM6.name}')
    m6 = pd.read_excel(MOESM6, sheet_name='TableS5')
    m6 = m6[~m6['assay'].isin(SKIP)].copy()

    print(f'[load] {SAE_DIFF.name}')
    DIFF = torch.load(SAE_DIFF, map_location='cpu')
    assert DIFF.shape == (20, 187, 16384), DIFF.shape

    print(f'[load] {FEAT_TBL.name}')
    feat_desc = pq.read_table(FEAT_TBL).to_pandas().set_index('feature_id')

    lof = {a: lof_table(m6, a) for a in ASSAYS}

    print('[drop] computing per-position max drop across alleles')
    max_drop = max_drop_per_position(DIFF)                       # (188, 16384)
    F = np.nonzero((max_drop > 0).any(axis=0))[0]
    max_drop_active = max_drop[:, F]
    print(f'[feat] active features (non-zero drop on KRAS): {len(F)} / 16384')

    records = []
    for a in ASSAYS:
        s = lof[a]['max_ddG'].sort_values(ascending=False)
        top_K = s.head(TOP_K_CANDIDATES).index.tolist()
        cls = cluster_positions(top_K, gap=CLUSTER_GAP)
        cls = sorted(cls, key=lambda c: lof[a].loc[c, 'max_ddG'].max(),
                     reverse=True)[:N_CLUSTERS]
        for rank, c in enumerate(cls):
            rep_p = int(lof[a].loc[c, 'max_ddG'].idxmax())
            c_rows = np.array([p - 1 for p in c])
            mean_md = max_drop_active[c_rows].mean(axis=0)
            rep_md = max_drop_active[rep_p - 1]
            order = np.argsort(mean_md)[::-1][:TOP_N_DESC]
            for j, li in enumerate(order):
                fid = int(F[li])
                info = feat_desc.loc[fid] if fid in feat_desc.index else None
                records.append({
                    'assay': a, 'cluster_rank': rank + 1,
                    'cluster_label': cluster_label(c), 'rep_pos': rep_p,
                    'cluster_max_ddG': float(lof[a].loc[c, 'max_ddG'].max()),
                    'desc_rank': j + 1, 'feature_id': fid,
                    'mean_max_drop': float(mean_md[li]),
                    'max_drop_at_rep': float(rep_md[li]),
                    'category': (info['category'] if info is not None else None),
                    'summary':  (info['summary']  if info is not None else None),
                })

    df = pd.DataFrame(records)
    df.to_parquet(OUT_DIR / 'descriptive_features.parquet')
    print(f'[save] validation_2/descriptive_features.parquet  rows={len(df)}')

    def trunc(s, n=120):
        if not isinstance(s, str): return ''
        return s if len(s) <= n else s[:n - 1] + '…'

    with open(OUT_DIR / 'descriptive.txt', 'w') as fh:
        fh.write('# Validation 2 (descriptive) — most-dropped SAE features per hotspot cluster\n\n')
        fh.write('# No significance test. Per cluster, features are ranked by\n')
        fh.write('# mean(max_drop) across the cluster positions; max_drop is the\n')
        fh.write('# largest drop = max(0, sae_WT − sae) over the 20 alleles at a position.\n')
        fh.write(f'# Top {TOP_N_DESC} features listed per cluster.\n\n')
        for a in ASSAYS:
            fh.write(f'## {a}\n')
            sub_a = df[df['assay'] == a]
            for cr in sorted(sub_a['cluster_rank'].unique()):
                cc = sub_a[sub_a['cluster_rank'] == cr]
                r0 = cc.iloc[0]
                fh.write(f'  Cluster #{cr} ({r0.cluster_label}) — '
                         f'rep pos {r0.rep_pos}, max ΔΔG {r0.cluster_max_ddG:+.2f}\n')
                for r in cc.itertuples():
                    fh.write(f'    f{r.feature_id:<5d}  '
                             f'mean_max_drop={r.mean_max_drop:.3f}  '
                             f'(@rep {r.max_drop_at_rep:.3f})  '
                             f'[{r.category}] {trunc(r.summary)}\n')
            fh.write('\n')
    print('[save] validation_2/descriptive.txt')
    print('Done.')


if __name__ == '__main__':
    main()
