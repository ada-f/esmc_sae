"""Step 03 — export per-assay DMS matrices in heatmap layout, plus the WT FASTA.

For each assay we produce TWO CSVs:
  kras_ddG_<assay>.csv       inferred ddG from MOESM6 (kcal/mol; the values
                             plotted in the Validation 2 callout figures)
  kras_fitness_<assay>.csv   single-mutant fitness from MOESM5 (`fitness`
                             column), averaged across blocks for duplicates.

CSV layout
  rows:    20 amino acids in alphabetical order (A, C, D, E, ...).
  columns: KRAS canonical positions 2..188 (one-letter WT residue is in row 1).
  cells:   the value (NaN if not measured / not confident); the WT cell at each
           position is left NaN (so collaborators can detect it).

KRAS WT FASTA is saved as kras_wt.fasta.
"""

import pandas as pd
import numpy as np

from config import MOESM5, MOESM6, DMS_MATRICES

DMS_MATRICES.mkdir(parents=True, exist_ok=True)

KRAS_WT = ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY"
           "SAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTK"
           "QAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM")
assert len(KRAS_WT) == 188

AA_ROWS = list('ACDEFGHIKLMNPQRSTVWY')                  # alphabetical
POSITIONS = list(range(2, 189))                          # 2..188 inclusive

# MOESM6 assay name -> output file stem
ASSAYS_DDG = {
    'folding':           'folding',
    'RAF1':              'RAF1',
    'PIK3CG':            'PIK3CG',
    'RALGDS':            'RALGDS',
    'SOS1':              'SOS1',
    'DARPin K27':        'DARPin_K27',
    'DARPin K55':        'DARPin_K55',
    'full length RAF1':  'full_length_RAF1',
}
# MOESM5 assay name -> output file stem
ASSAYS_FIT = {
    'AbundancePCA':                          'folding',          # AbundancePCA ~ folding
    'BindingPCA RAF1RBD':                    'RAF1',
    'BindingPCA PIK3CGRBD':                  'PIK3CG',
    'BindingPCA RALGDSRBD':                  'RALGDS',
    'BindingPCA SOS1':                       'SOS1',
    'BindingPCA DARPin K27':                 'DARPin_K27',
    'BindingPCA DARPin K55':                 'DARPin_K55',
    'BindingPCA full length RAF1':           'full_length_RAF1',
    'BindingPCA RAF1RBD coexpression GAP':   'RAF1RBD_coex_GAP',
}


def empty_matrix():
    """Return a 20 x 187 DataFrame of NaN, rows=AA, cols=positions."""
    m = pd.DataFrame(np.nan, index=AA_ROWS, columns=POSITIONS, dtype=float)
    m.index.name = 'mutant_aa'
    m.columns.name = 'position'
    return m


def wt_row():
    """One-row DataFrame with the WT amino acid at each position (label row)."""
    return pd.DataFrame(
        {p: [KRAS_WT[p - 1]] for p in POSITIONS},
        index=['WT_aa']
    )


def export_ddg():
    df = pd.read_excel(MOESM6, sheet_name='TableS5')
    sub = df[df['Pos_real'].notna() & (df['wt_codon'] != df['mt_codon'])].copy()
    sub['Pos_real'] = sub['Pos_real'].astype(int)
    sub = sub.rename(columns={'mean_kcal/mol': 'ddG'})
    for raw_name, stem in ASSAYS_DDG.items():
        chunk = sub[sub['assay'] == raw_name]
        if chunk.empty:
            print(f'[skip] MOESM6 {raw_name}: no rows'); continue
        mat = empty_matrix()
        n = 0
        for _, r in chunk.iterrows():
            aa, p = r['mt_codon'], int(r['Pos_real'])
            if aa in mat.index and p in mat.columns:
                mat.at[aa, p] = r['ddG']; n += 1
        out = DMS_MATRICES / f'kras_ddG_{stem}.csv'
        pd.concat([wt_row(), mat]).to_csv(out)
        print(f'[save] {out.name}  cells={n}  range=[{chunk["ddG"].min():.2f}, {chunk["ddG"].max():.2f}]')


def export_fitness():
    df = pd.read_excel(MOESM5, sheet_name='TableS4')
    # In MOESM5, aa_seq is 187 chars (positions 2..188 in canonical KRAS).
    df = df[df['Nham_aa'].isin([0, 1])].copy()
    wt_seq187 = KRAS_WT[1:]
    assert len(wt_seq187) == 187

    def decode(seq):
        diffs = [(i, seq[i]) for i in range(len(wt_seq187)) if seq[i] != wt_seq187[i]]
        return diffs[0] if len(diffs) == 1 else (None, None)

    for raw_name, stem in ASSAYS_FIT.items():
        chunk = df[df['assay'] == raw_name].copy()
        chunk_single = chunk[chunk['Nham_aa'] == 1]
        if chunk_single.empty:
            print(f'[skip] MOESM5 {raw_name}: no single mutants'); continue
        # this assay used a truncated KRAS (full length RAF1 = aa 2..64 only);
        # restrict the decoder to the same length and remap positions later.
        if (chunk_single['aa_seq'].str.len() != 187).any():
            n_seq = int(chunk_single['aa_seq'].str.len().mode().iloc[0])
            wt_local = KRAS_WT[1:1 + n_seq]
        else:
            n_seq = 187
            wt_local = wt_seq187

        def decode_local(seq, ref=wt_local):
            diffs = [(i, seq[i]) for i in range(min(len(seq), len(ref))) if seq[i] != ref[i]]
            return diffs[0] if len(diffs) == 1 else (None, None)

        decoded = chunk_single['aa_seq'].apply(decode_local)
        chunk_single['Pos_real'] = [d[0] + 2 if d[0] is not None else None for d in decoded]
        chunk_single['mt_codon'] = [d[1] for d in decoded]
        chunk_single = chunk_single.dropna(subset=['Pos_real', 'mt_codon']).copy()
        chunk_single['Pos_real'] = chunk_single['Pos_real'].astype(int)
        # average across blocks for duplicate (Pos_real, mt_codon)
        avg = chunk_single.groupby(['Pos_real', 'mt_codon'])['fitness'].mean().reset_index()
        mat = empty_matrix()
        n = 0
        for _, r in avg.iterrows():
            aa, p = r['mt_codon'], r['Pos_real']
            if aa in mat.index and p in mat.columns:
                mat.at[aa, p] = r['fitness']; n += 1
        out = DMS_MATRICES / f'kras_fitness_{stem}.csv'
        pd.concat([wt_row(), mat]).to_csv(out)
        print(f'[save] {out.name}  cells={n}  range=[{avg["fitness"].min():.2f}, {avg["fitness"].max():.2f}]')


def export_fasta():
    fa = DMS_MATRICES / 'kras_wt.fasta'
    fa.write_text(
        '>KRAS_WT|UniProt:P01116-2|KRAS_4B|188aa\n'
        + '\n'.join(KRAS_WT[i:i + 60] for i in range(0, len(KRAS_WT), 60))
        + '\n'
    )
    print(f'[save] {fa.name}  ({len(KRAS_WT)} residues)')


if __name__ == '__main__':
    export_fasta()
    export_ddg()
    export_fitness()
