"""Step 04 — per-residue KRAS structural annotations from PDB 6VJJ.

For each KRAS residue, classify it against the partner and ligand of the 6VJJ
complex (KRAS chain A + RAF1-RBD chain B + GNP + MG):

  - binding interface : side-chain heavy-atom distance to RAF1 < 5 A
  - GTP pocket        : side-chain heavy-atom distance to GNP/MG < 5 A
  - core / surface    : relative solvent accessibility (RSA) < 0.25 is core

Distances use side-chain heavy atoms (glycine -> Ca), matching the paper's
analysis code. RSA is computed with freesasa on the isolated KRAS chain.
Residue numbering is canonical KRAS (6VJJ chain A residue R = KRAS position R).

This annotation track does not pixel-match the paper's Fig 1i panel; see
RESULTS.md section 5 for the reasons.

Output: data/kras_anno.csv  (Pos, aa, dist_RAF1, dist_GTP, rsa, region).
"""

import csv
import tempfile

import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.SeqUtils import seq1
import freesasa

from config import PDB, KRAS_ANNO

KRAS_CHAIN = 'A'
RAF1_CHAIN = 'B'
LIGANDS = {'GNP', 'MG'}
DIST_CUT = 5.0
RSA_CORE = 0.25
BACKBONE = {'N', 'CA', 'C', 'O', 'OXT'}


def sidechain_heavy(res):
    """Side-chain heavy-atom coords; glycine (no side chain) -> Ca. Matches
    the paper's scHA atom selection."""
    sc = [a for a in res if a.element != 'H' and a.get_name() not in BACKBONE]
    if not sc:
        sc = [a for a in res if a.get_name() == 'CA']
    return np.array([a.coord for a in sc]) if sc else np.empty((0, 3))


def main():
    parser = PDBParser(QUIET=True)
    s = parser.get_structure('6vjj', str(PDB))
    model = next(iter(s))
    chain_K, chain_R = model[KRAS_CHAIN], model[RAF1_CHAIN]

    # KRAS: side-chain heavy atoms per residue
    kras_res = {}
    for res in chain_K:
        if res.id[0] != ' ':
            continue
        try:
            aa1 = seq1(res.resname)
        except Exception:
            aa1 = 'X'
        kras_res[res.id[1]] = (aa1, sidechain_heavy(res))

    # RAF1: side-chain heavy atoms, pooled (scHA on both sides of the interface)
    raf1_scha = np.vstack([sidechain_heavy(res) for res in chain_R
                           if res.id[0] == ' '])
    # ligand: all heavy atoms (a small molecule has no side chain)
    lig_atoms = np.array([a.coord for ch in model for res in ch
                          if res.resname.strip() in LIGANDS
                          for a in res if a.element != 'H'])
    print(f'[atoms] RAF1 scHA={len(raf1_scha)}  ligand heavy={len(lig_atoms)}')

    # ---- RSA via freesasa relativeTotal on the isolated KRAS chain ----
    class KRASOnly(Select):
        def accept_chain(self, c):   return c.id == KRAS_CHAIN
        def accept_residue(self, r): return r.id[0] == ' '
    tmp = tempfile.NamedTemporaryFile(suffix='.pdb', delete=False); tmp.close()
    io = PDBIO(); io.set_structure(s); io.save(tmp.name, KRASOnly())
    fs_res = freesasa.calc(freesasa.Structure(tmp.name))
    areas = fs_res.residueAreas().get(KRAS_CHAIN, {})
    rsa = {int(k): v.relativeTotal for k, v in areas.items()}

    rows = []
    for resno in sorted(kras_res):
        aa, sc = kras_res[resno]
        d_raf = (float(np.min(np.linalg.norm(
                    sc[:, None, :] - raf1_scha[None, :, :], axis=-1)))
                 if len(sc) and len(raf1_scha) else float('inf'))
        d_gtp = (float(np.min(np.linalg.norm(
                    sc[:, None, :] - lig_atoms[None, :, :], axis=-1)))
                 if len(sc) and len(lig_atoms) else float('inf'))
        is_int, is_gtp = d_raf < DIST_CUT, d_gtp < DIST_CUT
        region = ('both' if (is_int and is_gtp) else
                  'interface' if is_int else
                  'gtp' if is_gtp else 'other')
        rows.append({'Pos': resno, 'aa': aa,
                     'dist_RAF1': round(d_raf, 3), 'dist_GTP': round(d_gtp, 3),
                     'rsa': round(rsa.get(resno, float('nan')), 3),
                     'region': region})

    with open(KRAS_ANNO, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['Pos', 'aa', 'dist_RAF1',
                                           'dist_GTP', 'rsa', 'region'])
        w.writeheader(); w.writerows(rows)
    print(f'[save] {KRAS_ANNO}  ({len(rows)} residues)')

    from collections import Counter
    print('[regions]', dict(Counter(r['region'] for r in rows)))
    gtp = sorted(r['Pos'] for r in rows if r['region'] in ('gtp', 'both'))
    itf = sorted(r['Pos'] for r in rows if r['region'] in ('interface', 'both'))
    print(f'[gtp pocket] {gtp}')
    print(f'[interface]  {itf}')
    print(f'[core] rsa<{RSA_CORE}: '
          f"{sum(1 for r in rows if r['rsa']==r['rsa'] and r['rsa']<RSA_CORE)}")


if __name__ == '__main__':
    main()
