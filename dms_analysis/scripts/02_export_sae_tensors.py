"""Step 02 — reshape the raw SAE features into DMS-heatmap-layout tensors.

Two tensors are written, both of shape (20, 187, 16384) and identical axis
ordering to the DMS CSVs in data/dms_matrices/:

  kras_sae_tensor.pt        sae[a, p, f] = pooled SAE activation of feature f
                             on the KRAS_{p->a} variant (the mutant where
                             position p+1 is mutated to amino acid AA_ROWS[a]).
                             WT cells (where AA_ROWS[a] == WT_aa at p+1) are NaN.

  kras_sae_tensor_diff.pt   d[a, p, f] = sae[a, p, f] - sae_WT[f]; same NaN
                             pattern for WT cells.

Indexing
  axis 0 -- mutant amino acid; 20 entries alphabetical: ACDEFGHIKLMNPQRSTVWY
  axis 1 -- KRAS canonical position 2..188 (187 entries); column c == position c+2.
  axis 2 -- SAE feature dimension 0..16383.

Also writes:
  kras_wt_sae_vector.pt     (16384,) the WT KRAS pooled SAE vector (the reference
                            subtracted from every mutant in the diff tensor).

The mutant feature rows are read from data/kras_sae_features.pt (step 01), whose
row ordering matches MOESM6.drop_duplicates(['Pos_real','mt_codon']) (row 0 =
the (NaN, NaN) WT placeholder; rows 1..3740 = the 187x20 single-mutant cells;
rows where mt_codon equals the WT residue at that position are 'diagonal'
placeholders that map to the WT cells and are written as NaN here).
"""

import json

import numpy as np
import pandas as pd
import torch

from config import MOESM6, SAE_FEATURES, DMS_MATRICES

DMS_MATRICES.mkdir(parents=True, exist_ok=True)

AA_ROWS = list('ACDEFGHIKLMNPQRSTVWY')                   # alphabetical (matches DMS CSVs)
POSITIONS = list(range(2, 189))                          # 2..188 inclusive
KRAS_WT = ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY"
           "SAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTK"
           "QAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM")
assert len(KRAS_WT) == 188


def main():
    print(f'[load] {MOESM6.name}')
    m6 = pd.read_excel(MOESM6, sheet_name='TableS5')

    # Step 01 builds pos_mts from MOESM6 in exactly this way and the SAE tensor
    # row ordering follows it (1 WT placeholder + 187x20 mutants).
    pos_mts = m6[['Pos_real', 'mt_codon']].drop_duplicates().reset_index(drop=True)
    assert len(pos_mts) == 3741, f'pos_mts has {len(pos_mts)} rows, expected 3741'

    print(f'[load] {SAE_FEATURES.name}')
    feats = torch.load(SAE_FEATURES, weights_only=False, map_location='cpu').to(torch.float32)
    assert feats.shape == (3741, 16384), feats.shape
    n_feat = feats.shape[1]

    wt_rows = pos_mts.index[pos_mts['Pos_real'].isna()].tolist()
    assert len(wt_rows) == 1, f'expected one WT placeholder row, found {wt_rows}'
    wt_idx = wt_rows[0]
    wt_vec = feats[wt_idx].clone()                        # (16384,)
    print(f'[align] WT placeholder at row {wt_idx}; WT vector norm = {wt_vec.norm().item():.4f}')

    # (20, 187, 16384) float32, ~245 MB
    n_aa, n_pos = len(AA_ROWS), len(POSITIONS)
    raw = torch.full((n_aa, n_pos, n_feat), float('nan'), dtype=torch.float32)

    wt_lookup = {p: KRAS_WT[p - 1] for p in POSITIONS}    # WT aa at each position
    aa_to_row = {aa: i for i, aa in enumerate(AA_ROWS)}

    n_filled = 0
    n_diag = 0
    n_stop = 0
    n_skip = 0
    for tensor_row, (p_real, mt) in enumerate(zip(pos_mts['Pos_real'], pos_mts['mt_codon'])):
        if pd.isna(p_real):
            continue                                     # WT placeholder row
        p = int(p_real)
        if p not in wt_lookup:
            n_skip += 1; continue
        if not isinstance(mt, str):
            n_skip += 1; continue
        # '*' (stop codon) and other non-canonical are excluded -- DMS layout is 20 AAs
        if mt == '*':
            n_stop += 1; continue
        if mt not in aa_to_row:
            n_skip += 1; continue
        # Skip the diagonal -- mutant_aa == WT_aa is the no-mutation placeholder.
        # We want those cells NaN to mirror the DMS CSV layout.
        if mt == wt_lookup[p]:
            n_diag += 1; continue
        raw[aa_to_row[mt], p - 2, :] = feats[tensor_row]
        n_filled += 1
    print(f'[fill] mutation cells filled : {n_filled}')
    print(f'[fill] diagonal WT cells     : {n_diag}  (left as NaN)')
    print(f'[fill] stop-codon cells (*)  : {n_stop}  (not represented in 20-AA grid)')
    print(f'[fill] otherwise-skipped     : {n_skip}')

    # mutant - wild-type; NaN propagates to WT cells
    diff = raw - wt_vec.view(1, 1, n_feat)

    raw_path = DMS_MATRICES / 'kras_sae_tensor.pt'
    diff_path = DMS_MATRICES / 'kras_sae_tensor_diff.pt'
    wt_path = DMS_MATRICES / 'kras_wt_sae_vector.pt'
    torch.save(raw, raw_path)
    torch.save(diff, diff_path)
    torch.save(wt_vec, wt_path)
    print(f'[save] {raw_path.name}        shape={tuple(raw.shape)}    '
          f'~{raw.element_size()*raw.numel()/2**20:.0f} MB')
    print(f'[save] {diff_path.name}   shape={tuple(diff.shape)}   '
          f'~{diff.element_size()*diff.numel()/2**20:.0f} MB')
    print(f'[save] {wt_path.name}     shape={tuple(wt_vec.shape)}')

    # metadata for decoding the tensor indices
    meta = {
        'shape':                list(raw.shape),
        'axes':                 ['mutant_aa', 'position', 'sae_feature'],
        'aa_rows':              AA_ROWS,
        'positions':            POSITIONS,
        'n_features':           n_feat,
        'wt_sequence_188aa':    KRAS_WT,
        'wt_aa_at_position':    {p: KRAS_WT[p - 1] for p in POSITIONS},
        'wt_cell_convention':   'NaN at (mutant_aa == WT_aa at that position) cells',
        'stop_codon_excluded':  True,
        'source':               'kras_sae_features.pt (3741, 16384) reordered into DMS-heatmap layout',
        'reference_subtracted_in_diff':
                                 'kras_wt_sae_vector.pt -- the WT pooled SAE feature vector',
    }
    (DMS_MATRICES / 'kras_sae_tensor_meta.json').write_text(json.dumps(meta, indent=2))
    print('[save] kras_sae_tensor_meta.json')

    # sanity check on a few cells
    for (aa, p) in [('A', 12), ('G', 12), ('A', 38), ('R', 41)]:
        a_row = AA_ROWS.index(aa); p_col = p - 2
        v_raw = raw[a_row, p_col]; v_diff = diff[a_row, p_col]
        if torch.isnan(v_raw).any():
            kind = 'NaN (WT cell)'
        else:
            kind = (f'norm_raw={v_raw.norm():.4f}  '
                    f'|diff|_inf={v_diff.abs().max():.4f}  '
                    f'|diff|_L1={v_diff.abs().sum():.4f}')
        print(f'[check] {aa}{p}  ({kind})')


if __name__ == '__main__':
    main()
