"""Step 08 / Validation 2 -- which SAE features are disrupted at LOF hotspots.

Per position, max_drop[p, f] is the largest drop any substitution at position p
causes in feature f, where drop = max(0, sae_WT - sae). The highest-ΔΔG
positions of each assay are chained into clusters; for each cluster and feature
the observed statistic is the mean of max_drop over the cluster's positions.

Significance is a permutation test: the statistic is recomputed on 200,000
random position-sets of the same size, drawn from all DMS positions of the
assay (the cluster's own positions included). The empirical p-value is
corrected per cluster with the Benjamini-Hochberg procedure (significant at
q < 0.05). Per cluster, the top-5 significant features are reported, ranked by
max_drop at the cluster's highest-ΔΔG position.

Input:  data/dms_matrices/kras_sae_tensor_diff.pt (sae - sae_WT).
Outputs (results/validation_2/):
  clusters.parquet           per-cluster info per assay
  feature_pvalues.parquet    per (assay, cluster, feature) p-value and q-value
  cluster_features.parquet   top-5 features per cluster
  summary.txt                human-readable summary
"""

import numpy as np
import pandas as pd
import torch
import pyarrow.parquet as pq
from statsmodels.stats.multitest import multipletests

from config import MOESM6, SAE_TENSOR_DIFF, FEATURE_TABLE, RESULTS
from dms_common import (V2_ASSAYS as ASSAYS, V2_SKIP as SKIP, TOP_K_CANDIDATES,
                        N_CLUSTERS, CLUSTER_GAP, cluster_positions, cluster_label,
                        lof_table, max_drop_per_position)

SAE_DIFF = SAE_TENSOR_DIFF
FEAT_TBL = FEATURE_TABLE

OUT_DIR = RESULTS / 'validation_2'
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_PERMUTATIONS = 200_000     # size-matched random subsets per cluster
PERM_BATCH = 1_000
Q_THRESH = 0.05
SEED = 1234

rng = np.random.default_rng(SEED)


def cluster_mean_perm_pvalues(obs, pool_md, n_c, n_perm, rng):
    """Empirical p-value per feature: the fraction of n_perm random size-n_c
    position-sets whose mean max_drop matches or exceeds obs. pool_md is
    max_drop over all DMS positions of the assay.
    """
    n_pool = pool_md.shape[0]
    n_F = pool_md.shape[1]
    counts = np.zeros(n_F, dtype=np.int64)
    for b0 in range(0, n_perm, PERM_BATCH):
        this_b = min(PERM_BATCH, n_perm - b0)
        # without-replacement size-n_c samples from the full pool
        samp = np.empty((this_b, n_c), dtype=int)
        for r in range(this_b):
            samp[r] = rng.choice(n_pool, n_c, replace=False)
        null_mean = pool_md[samp].mean(axis=1)                  # (this_b, n_F)
        counts += (null_mean >= obs).sum(axis=0)
    p_emp = (counts + 1.0) / (n_perm + 1.0)
    return p_emp


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
    active_feats = np.nonzero((max_drop > 0).any(axis=0))[0]
    F = active_feats
    max_drop_active = max_drop[:, F].astype(np.float32)          # (188, n_F)
    print(f'[feat] features with non-zero drop somewhere on KRAS: {len(F)} / 16384')

    clusters_per_assay = {}
    for a in ASSAYS:
        s = lof[a]['max_ddG'].sort_values(ascending=False)
        top_K = s.head(TOP_K_CANDIDATES).index.tolist()
        cls = cluster_positions(top_K, gap=CLUSTER_GAP)
        cls_sorted = sorted(cls, key=lambda c: lof[a].loc[c, 'max_ddG'].max(),
                            reverse=True)[:N_CLUSTERS]
        clusters_per_assay[a] = cls_sorted
        print(f'[cluster] {a:<12s} top-{TOP_K_CANDIDATES} -> {len(cls)} raw -> {len(cls_sorted)} kept')

    pval_records = []
    cluster_records = []
    cluster_feat_records = []
    for a in ASSAYS:
        cls = clusters_per_assay[a]
        rep_positions = [int(lof[a].loc[c, 'max_ddG'].idxmax()) for c in cls]
        # pool = all DMS positions of the assay (cluster positions included)
        dms_pos = lof[a].index.to_numpy()
        pool_rows = np.array([p - 1 for p in dms_pos])
        pool_md = max_drop_active[pool_rows]                     # (n_pool, n_F)
        n_pool = len(pool_rows)
        for rank, (c, rep_p) in enumerate(zip(cls, rep_positions)):
            n_c = len(c)
            c_rows = np.array([p - 1 for p in c])
            obs = max_drop_active[c_rows].mean(axis=0)           # (n_F,)
            print(f'[perm] {a:<12s} cluster#{rank+1} ({cluster_label(c)}) '
                  f'n_c={n_c} pool={n_pool} B={N_PERMUTATIONS}', flush=True)
            p_emp = cluster_mean_perm_pvalues(obs, pool_md, n_c,
                                              N_PERMUTATIONS, rng)
            rej, q_bh, _, _ = multipletests(p_emp, alpha=Q_THRESH, method='fdr_bh')
            sig_mask = q_bh < Q_THRESH

            for li, f in enumerate(F):
                pval_records.append((a, rank + 1, int(f),
                                     float(obs[li]),
                                     float(p_emp[li]),
                                     float(q_bh[li]),
                                     bool(rej[li])))

            # per-cluster top-5: BH-sig only, ranked by max_drop at rep
            act_at_rep = max_drop_active[rep_p - 1]
            sub = pd.DataFrame({
                'feature_id': F,
                'activation': act_at_rep,
                'p_emp':      p_emp,
                'q_bh':       q_bh,
                'sig':        sig_mask,
            })
            sub = sub[sub['sig']].sort_values(['activation', 'q_bh'],
                                              ascending=[False, True]).head(5)
            for fr in sub.itertuples():
                info = feat_desc.loc[int(fr.feature_id)] if int(fr.feature_id) in feat_desc.index else None
                cluster_feat_records.append({
                    'assay':           a,
                    'cluster_rank':    rank + 1,
                    'cluster_label':   cluster_label(c),
                    'rep_pos':         int(rep_p),
                    'cluster_max_ddG': float(lof[a].loc[c, 'max_ddG'].max()),
                    'feature_id':      int(fr.feature_id),
                    'activation':      float(fr.activation),
                    'p_emp':           float(fr.p_emp),
                    'q_bh':            float(fr.q_bh),
                    'direction':       1,
                    'category':        (info['category'] if info is not None else None),
                    'summary':         (info['summary']  if info is not None else None),
                })
            cluster_records.append({
                'assay': a, 'cluster_rank': rank + 1,
                'cluster_label': cluster_label(c), 'rep_pos': int(rep_p),
                'positions': c, 'n_positions': n_c,
                'cluster_max_ddG':  float(lof[a].loc[c, 'max_ddG'].max()),
                'cluster_mean_ddG': float(lof[a].loc[c, 'max_ddG'].mean()),
                'n_features_sig':   int(sig_mask.sum()),
                'n_pool_pos':       int(n_pool),
            })

    pv = pd.DataFrame(pval_records,
                      columns=['assay', 'cluster_rank', 'feature_id',
                               'obs_meanmax', 'p_emp', 'q_bh', 'sig'])
    pv.to_parquet(OUT_DIR / 'feature_pvalues.parquet')
    clust_df = pd.DataFrame(cluster_records)
    clust_df.to_parquet(OUT_DIR / 'clusters.parquet')
    cf_df = pd.DataFrame(cluster_feat_records)
    cf_df.to_parquet(OUT_DIR / 'cluster_features.parquet')
    print(f'[save] validation_2/feature_pvalues.parquet  rows={len(pv)}')
    print(f'[save] validation_2/clusters.parquet         rows={len(clust_df)}')
    print(f'[save] validation_2/cluster_features.parquet rows={len(cf_df)}')

    def trunc(s, n=120):
        if not isinstance(s, str): return ''
        return s if len(s) <= n else s[:n - 1] + '…'

    with open(OUT_DIR / 'summary.txt', 'w') as fh:
        fh.write('# Validation 2 — per-cluster mean-of-max-drop permutation test\n\n')
        fh.write('# Per (allele, pos, feature): drop = max(0, sae_WT − sae).\n')
        fh.write('# Per (pos, feature):         max_drop = max over alleles.\n')
        fh.write('# Per (cluster, feature):     obs = mean over cluster positions of max_drop.\n')
        fh.write('# Null: same statistic on |c| random DMS positions drawn from ALL\n'
                 f'#       DMS positions (cluster included; proper permutation pool),\n'
                 f'#       B = {N_PERMUTATIONS}, without replacement.\n')
        fh.write(f'# BH-FDR INDEPENDENTLY per cluster over {len(active_feats)} active features\n'
                 f'#   (q < {Q_THRESH}).\n')
        fh.write('# Top-5: BH-sig features ranked by max_drop at the cluster rep position.\n\n')
        for a in ASSAYS:
            fh.write(f'## {a}\n')
            cls = clust_df[clust_df['assay'] == a].sort_values('cluster_rank')
            for r in cls.itertuples():
                fh.write(f'  Cluster #{r.cluster_rank} ({r.cluster_label}) — '
                         f'rep pos {r.rep_pos}, max ΔΔG {r.cluster_max_ddG:+.2f}, '
                         f'{r.n_positions} pos, pool={r.n_pool_pos}, '
                         f'{r.n_features_sig} sig features (q<{Q_THRESH})\n')
                if cf_df.empty:
                    continue
                sub = cf_df[(cf_df['assay'] == a) & (cf_df['cluster_rank'] == r.cluster_rank)]
                for f in sub.itertuples():
                    fh.write(f'    f{f.feature_id:<5d}  q={f.q_bh:.2e}  '
                             f'max_drop={f.activation:.3f}  [{f.category}] {trunc(f.summary)}\n')
            fh.write('\n')
    print('[save] validation_2/summary.txt')
    print('Done.')


if __name__ == '__main__':
    main()
