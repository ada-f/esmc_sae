"""Shared helpers for the Validation 2 hotspot-clustering scripts.

Steps 08 (permutation test) and 09 (descriptive ranking) both cluster the same
loss-of-function hotspots, so the clustering logic lives here and is imported
by both. Keeping it in one module avoids the two scripts drifting apart.
"""

import numpy as np
import pandas as pd
import torch

# --- Validation 2 assay set --------------------------------------------------
# Raw MOESM6 assay names (spaces, not underscores) — steps 08/09 read MOESM6
# directly, unlike steps 05-07 and 11 which read the exported kras_ddG_<assay>.csv files.
V2_ASSAYS = ['folding', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'DARPin K27', 'DARPin K55']
V2_SKIP = {'full length RAF1'}          # partial-coverage assay, excluded

TOP_K_CANDIDATES = 20                   # candidate hotspot positions per assay
N_CLUSTERS = 6                          # clusters reported per assay
CLUSTER_GAP = 2                         # positions within this gap are chained


def cluster_positions(positions_sorted_by_score, gap=CLUSTER_GAP):
    """Greedily chain positions (given in DESCENDING score order) into clusters;
    a position joins a cluster if it is within ``gap`` of any of its members."""
    clusters = []
    for p in positions_sorted_by_score:
        attached = False
        for c in clusters:
            if any(abs(p - q) <= gap for q in c):
                c.append(p)
                attached = True
                break
        if not attached:
            clusters.append([p])
    return [sorted(c) for c in clusters]


def cluster_label(positions):
    """Compact human label for a cluster, e.g. [12,13,14,17] -> '12-14, 17'."""
    if not positions:
        return ''
    runs = []
    start = positions[0]; prev = start
    for q in positions[1:]:
        if q - prev <= 1:
            prev = q
        else:
            runs.append((start, prev)); start = q; prev = q
    runs.append((start, prev))
    return ', '.join(f'{a}-{b}' if a != b else f'{a}' for a, b in runs)


def lof_table(m6, assay):
    """Per-position loss-of-function summary for one assay: max ddG across
    substitutions and the number of distinct substitutions measured."""
    sub = m6[(m6['assay'] == assay) & m6['Pos_real'].notna()
             & (m6['wt_codon'] != m6['mt_codon']) & (m6['ddG_conf'] == 1)].copy()
    sub['Pos_real'] = sub['Pos_real'].astype(int)
    by_pos = sub.groupby('Pos_real')
    return pd.DataFrame({
        'max_ddG': by_pos['mean_kcal/mol'].max(),
        'n_subs':  by_pos['mt_codon'].nunique(),
    })


def max_drop_per_position(DIFF):
    """From the (20, n_pos, F) diff tensor build a (n_pos+1, F) array of max_drop
    where row p-1 == canonical position p; row 0 (the skipped N-terminal Met)
    is zeros. drop = max(0, -diff); NaN cells (WT diagonal) are zeroed first."""
    drop = torch.clamp(-DIFF, min=0.0)                          # (20, n_pos, F)
    drop = torch.where(torch.isnan(drop), torch.zeros_like(drop), drop)
    md = drop.max(dim=0).values                                 # (n_pos, F)
    pad = torch.zeros(1, md.shape[1], dtype=md.dtype)
    return torch.cat([pad, md], dim=0).numpy()                  # (n_pos+1, F)
