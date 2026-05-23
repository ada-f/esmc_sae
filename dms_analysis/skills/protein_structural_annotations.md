# Skill: Protein structural annotations from a PDB

## Purpose

Given a protein structure, produce a per-residue annotation table: which
residues sit at a binding interface, which line a ligand pocket, which are
buried (core) vs solvent-exposed (surface), and the secondary-structure element
each belongs to. This is the annotation track drawn under a DMS heatmap (see
`annotated_dms_heatmap`) and the structural prior the SAE results are read
against.

Reference implementation: `dms_analysis/scripts/04_compute_structural_annotations.py`
(annotates KRAS from PDB 6VJJ — KRAS chain + RAF1-RBD partner + GTP-analogue
ligand). The skill is structure-agnostic: swap the PDB ID, chains and ligand
resnames for any complex. Could be formalised as a ToolUniverse `StructuralAnnotation`
tool taking a PDB ID + chain ids + ligand resnames.

## Inputs

| Input | KRAS example |
|---|---|
| PDB file / ID | `6VJJ` |
| Target chain (the protein being annotated) | `A` |
| Partner chain(s) for the interface | `B` (RAF1-RBD) |
| Ligand resnames for the pocket | `GNP`, `MG` |
| Distance cutoff | `5.0` Å |
| Core RSA cutoff | `0.25` |

## Method

1. **Pin the numbering.** Read the PDB; confirm the target chain's residue
   numbers map to the canonical sequence by an identity check (residue 1 is the
   expected amino acid, etc.). Crystal constructs often carry extra N-terminal
   cloning residues — record the offset, never assume 1:1.

2. **Binding interface** — a residue is "interface" if its minimum distance to
   any partner-chain atom is below the cutoff. *Metric choice matters:*
   all-heavy-atom (`HA`) vs side-chain-heavy-atom (`scHA`, glycine → Cα). `HA`
   is the literal residue distance; `scHA` excludes backbone reach and runs
   ≈ 2–3 residues tighter at loop edges. Pick one explicitly and report it.

3. **Ligand pocket** — the same distance rule, target residue vs the ligand's
   heavy atoms (a small molecule has no side chain — use all heavy atoms).

4. **Core vs surface** — relative solvent accessibility (RSA).
   *Critical:* RSA must be self-consistent — the residue SASA and the
   fully-exposed reference must come from the **same** tool/algorithm.
   `freesasa`'s `residueAreas().relativeTotal` does this internally; dividing
   one tool's SASA by another's max-ASA table biases RSA and over-calls core.
   Compute SASA on the **isolated** target chain — solvent accessibility is an
   intrinsic property of the fold, and partner chains would bury interface
   residues.

5. **Secondary structure** — from the structure (e.g. DSSP) or a curated
   source; record element ranges (β-strands, α-helices) on the same numbering.

## Output

A table keyed by canonical residue position: `aa`, `dist_partner`,
`dist_ligand`, `rsa`, `region ∈ {interface, ligand, both, other}`, `is_core`,
`ss_element`.

## Pitfalls

- **Numbering offsets are silent.** Verify residue identity before any join. A
  published annotation figure can itself carry a registration offset between
  its track and its sequence — when reproducing one, cross-check against the
  paper's *text*, which states residue numbers explicitly. (In the KRAS case
  the published Fig 1i panel is shifted +2 relative to its own sequence; the
  scHA annotation on canonical numbering is correct, the panel is not.)
- **RSA tool mismatch** (SASA from tool X ÷ max-ASA from tool Y) systematically
  shifts the core/surface call. Use one self-consistent RSA, and expect
  residues near the cutoff to flip between tools — a residue-exact match to a
  published core/surface track is generally not reproducible.
- **HA vs scHA** changes the interface/pocket extent by a few residues — never
  leave the choice implicit.
