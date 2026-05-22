# Skill: Annotated DMS heatmap

## Purpose

Visualise a deep-mutational-scanning (DMS) effect matrix — substitutions ×
positions — as a heatmap aligned to the protein sequence and to a structural
annotation track (domains, core/surface, secondary structure, interface /
ligand-pocket residue colouring). This is the standard "Fig 1-style" DMS panel,
and the figure that places per-hotspot SAE-feature callouts next to the
biology (see `sae_hotspot_feature_enrichment`).

Reference implementation: `dms_analysis/scripts/10_validation2_plot_callouts.py`
(the heatmap + sequence + annotation portion); the annotation track comes from
the `protein_structural_annotations` skill. Could be formalised as a
ToolUniverse plotting tool taking a DMS matrix + an annotation table.

## Inputs

| Input | Notes |
|---|---|
| DMS matrix | rows = 20 amino acids, columns = positions; values = ΔΔG or fitness |
| Sequence | one-letter, on the same numbering as the matrix columns |
| Annotation table | from `protein_structural_annotations`, same numbering |
| Colour scale | diverging, centred at 0 for ΔΔG (e.g. blue–grey–red) |

## Method

1. **Align everything to one position axis.** Heatmap column `p`, sequence
   letter `p`, and every annotation bar covering residue `p` must share x = `p`.
   This is the single thing to get right — verify the sequence letter under a
   known landmark column is the expected residue before drawing anything.

2. **Heatmap** — `imshow` the matrix with symmetric colour limits for ΔΔG
   (the KRAS figures use ±3 kcal/mol). Distinguish three cell kinds: a real
   measurement (coloured), the wild-type cell (mark it — e.g. a short dash —
   and set it to the colour-scale centre, not "missing"), and a genuinely
   missing measurement (render it as a distinct colour, e.g. white).

3. **Sequence strip** — one monospace letter per column, optionally coloured
   by residue role (interface / pocket / both).

4. **Annotation tracks**, stacked below the sequence:
   - domains / motifs as labelled bars,
   - core (filled) vs surface,
   - secondary structure (β-strand vs α-helix).
   Put any legends *above* the sequence so they never collide with the bars.

5. **Optional callout row** above the heatmap — boxes of per-hotspot SAE
   features (from `sae_hotspot_feature_enrichment`), each linked by a leader
   line to a bracket spanning its hotspot cluster on the heatmap.

6. Keep text concise and jargon explained — this is a figure for readers, and
   the analysis description belongs in the report, not the panel.

## Output

A per-assay PNG: optional callout row · heatmap · sequence · annotation tracks.

## Pitfalls

- A 1–2 residue misalignment between the heatmap and the annotation track is a
  common error and is visually subtle — always check a landmark column.
- "Wild-type cell" and "not measured" are different states; giving them the
  same colour misleads the reader into seeing missing data as neutral.
- If reproducing a published panel, verify *its* track alignment before
  treating it as ground truth (published DMS panels do carry registration
  errors — see `protein_structural_annotations`).
