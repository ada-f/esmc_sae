"""Step 10 / Validation 2 -- publication-quality per-assay LOF heatmap with
SAE-feature callouts.

Layout (per assay PNG):

   +--------+  +--------+  +--------+  +--------+  +--------+  (5 callout boxes,
   | ddG .  |  | ddG .  |  | ddG .  |  | ddG .  |  | ddG .  |   ordered by
   | feats  |  | feats  |  | feats  |  | feats  |  | feats  |   position)
   +---+----+  +---+----+  +---+----+  +---+----+  +---+----+
       |           |           |           |           |      <- leader lines
   +---[bracket]---[bracket]---[bracket]--[bracket]--[bracket]--+
   |   signed ddG heatmap (20 AA x 187 pos)                    |
   +-----------------------------------------------------------+
   sequence strip (one letter per position)
   secondary-structure + region annotation bars
"""

import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch
from matplotlib.gridspec import GridSpec

from config import MOESM6, KRAS_ANNO, ARIAL, RESULTS

V2 = RESULTS / 'validation_2'                     # cluster tables + figure outputs
ASSAYS = ['folding', 'RAF1', 'PIK3CG', 'RALGDS', 'SOS1', 'DARPin K27', 'DARPin K55']
AA_ORDER = list('DERHKSTNQCGPAVILMFYW')

# Sequence-residue colours (paper Fig 1i)
COLOR_INTERFACE = '#d62728'   # red
COLOR_GTP       = '#1f77b4'   # blue
COLOR_BOTH      = '#7c3aed'   # purple
COLOR_OTHER     = '#111111'   # near-black
COLOR_CORE      = '#222222'   # black for core bars
COLOR_SURFACE   = '#d8dde4'   # light grey for surface bars

# Arial
if ARIAL.exists():
    font_manager.fontManager.addfont(str(ARIAL))
    plt.rcParams['font.family']      = 'Arial'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm']      = 'Arial'
    plt.rcParams['mathtext.it']      = 'Arial:italic'
    plt.rcParams['mathtext.bf']      = 'Arial:bold'
plt.rcParams['axes.unicode_minus']   = False

DDG_VMIN, DDG_VMAX = -3.0, 3.0
# diverging blue-grey-red (paper Fig 1g/h convention): centre is GREY so that
# ddG = 0 is visually distinct from missing data, which renders white.
CMAP = LinearSegmentedColormap.from_list('bgr', ['#2166ac', '#d6d6d6', '#b2182b'])
CMAP.set_bad('white')

KRAS_WT = ("MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAGQEEY"
           "SAMRDQYMRTGEGFLCVFAINNTKSFEDIHHYREQIKRVKDSEDVPMVLVGNKCDLPSRTVDTK"
           "QAQDLARSYGIPFIETSAKTRQGVDDAFYTLVREIRKHKEKMSKDGKKKKKKSKTKCVIM")

REGIONS = [
    ('P-loop',     10, 17, '#b35a5a', 'white'),  # muted brick red
    ('Switch I',   25, 40, '#d99756', 'white'),  # muted amber
    ('Switch II',  58, 76, '#6da670', 'white'),  # muted moss green
    ('HVR',       167, 188, '#9d9d62', 'white'), # muted olive
]
HELIX_FACE  = '#cccccc'   # light gray
STRAND_FACE = '#666666'   # mid-dark gray
SECONDARY = [
    ('β1',  3,   9,   'strand'),
    ('α1',  15,  24,  'helix'),
    ('β2',  38,  44,  'strand'),
    ('β3',  51,  57,  'strand'),
    ('α2',  67,  73,  'helix'),
    ('β4',  77,  84,  'strand'),
    ('α3',  87,  104, 'helix'),
    ('β5',  109, 115, 'strand'),
    ('α4',  127, 136, 'helix'),
    ('β6',  139, 143, 'strand'),
    ('α5',  148, 166, 'helix'),
]


def load_kras_anno():
    """Return (region_by_pos, core_by_pos) dicts keyed by canonical position."""
    if not KRAS_ANNO.exists():
        return {}, {}
    df = pd.read_csv(KRAS_ANNO)
    region = dict(zip(df['Pos'].astype(int), df['region']))
    core   = {int(p): bool(r < 0.25) for p, r in zip(df['Pos'], df['rsa'])}
    return region, core


def ddg_matrix(m6, assay):
    sub = m6[(m6['assay'] == assay) & m6['Pos_real'].notna()
             & (m6['wt_codon'] != m6['mt_codon'])]
    positions = np.arange(2, 189)
    mat = pd.DataFrame(np.nan, index=AA_ORDER, columns=positions, dtype=float)
    for _, r in sub.iterrows():
        m = r['mt_codon']
        if m in mat.index:
            mat.at[m, int(r['Pos_real'])] = r['mean_kcal/mol']
    # WT cells are not "missing" -- set them to 0 (renders grey) so white stays
    # reserved for genuinely missing measurements; a dash marks the WT cell.
    for p in positions:
        wt = KRAS_WT[p - 1]
        if wt in mat.index:
            mat.at[wt, int(p)] = 0.0
    return mat


def _normalize(text):
    """Replace glyphs Arial lacks (non-breaking hyphen, some narrow-no-break
    spaces, etc.) so matplotlib doesn't fall back."""
    if not isinstance(text, str):
        return ''
    return (text
            .replace('‑', '-')   # non-breaking hyphen
            .replace(' ', ' ')   # narrow no-break space
            .replace(' ', ' ')   # no-break space
            .replace(' ', ' ')   # thin space
            )


def wrap(text, width=32, max_lines=4):
    text = _normalize(text)
    if not text:
        return ''
    # break_long_words=True so a stray 25-char term doesn't push past the box.
    parts = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=True)
    if len(parts) > max_lines:
        parts = parts[:max_lines]
        parts[-1] = parts[-1].rstrip(' ,.;:—–-') + '…'
    return '\n'.join(parts)


def draw_annotations(ax_seq, ax_anno, x_min, x_max, region_by_pos, core_by_pos):
    REGION_COLOR = {
        'interface': COLOR_INTERFACE,
        'gtp':       COLOR_GTP,
        'both':      COLOR_BOTH,
        'other':     COLOR_OTHER,
    }
    ax_seq.set_xlim(x_min, x_max); ax_seq.set_ylim(0, 1); ax_seq.axis('off')
    for i, aa in enumerate(KRAS_WT):
        pos = i + 1
        c = REGION_COLOR.get(region_by_pos.get(pos, 'other'), COLOR_OTHER)
        ax_seq.text(pos, 0.5, aa, ha='center', va='center',
                    family='monospace', fontsize=11, color=c, weight='bold')

    # ax_anno y in [0, 1] partitioned into three strips:
    #   regions      y = 0.70-0.98
    #   core/surface y = 0.38-0.58
    #   sec. struct. y = 0.04-0.30
    ax_anno.set_xlim(x_min, x_max); ax_anno.set_ylim(0, 1); ax_anno.axis('off')

    # Top row: regions (P-loop, switches, HVR) as filled bars
    y_top, h_top = 0.70, 0.28
    for name, start, end, color, tcolor in REGIONS:
        ax_anno.add_patch(Rectangle((start - 0.5, y_top), end - start + 1, h_top,
                                    facecolor=color, edgecolor='none',
                                    alpha=0.95, zorder=2))
        ax_anno.text((start + end) / 2, y_top + h_top / 2, name,
                     ha='center', va='center', fontsize=17, weight='bold',
                     color=tcolor, zorder=3)

    # Middle row: Core (black bars) on a WHITE background. Non-core positions
    # are surface; we annotate them with a single 'Surface' label in the widest
    # contiguous white region.
    y_mid, h_mid = 0.38, 0.20
    if core_by_pos:
        # Core = black bars on residues with RSA < 0.25 (consecutive positions merged)
        positions = sorted(p for p, is_core in core_by_pos.items() if is_core)
        runs = []
        if positions:
            start = prev = positions[0]
            for p in positions[1:]:
                if p == prev + 1:
                    prev = p
                else:
                    runs.append((start, prev)); start = p; prev = p
            runs.append((start, prev))
            for a, b in runs:
                ax_anno.add_patch(Rectangle((a - 0.5, y_mid), b - a + 1, h_mid,
                                            facecolor=COLOR_CORE, edgecolor='none',
                                            zorder=2))
        # Find the widest surface gap (between two consecutive core runs OR
        # between the last core run and x_max -- the HVR is typically the widest).
        boundaries = [(x_min, x_min)] + runs + [(x_max, x_max)]
        widest_gap = (None, 0.0)
        for (a1, b1), (a2, b2) in zip(boundaries, boundaries[1:]):
            gap_start = b1 + 0.5
            gap_end   = a2 - 0.5
            width = gap_end - gap_start
            if width > widest_gap[1]:
                widest_gap = ((gap_start + gap_end) / 2, width)
        if widest_gap[0] is not None and widest_gap[1] > 6:
            ax_anno.text(widest_gap[0], y_mid + h_mid / 2, 'Surface',
                         ha='center', va='center', fontsize=16, weight='bold',
                         color='#444', zorder=3)
        # 'Core' label sits inside the widest black core bar, in white.
        if runs:
            wa, wb = max(runs, key=lambda r: r[1] - r[0])
            ax_anno.text((wa + wb) / 2, y_mid + h_mid / 2, 'Core',
                         ha='center', va='center', fontsize=16, weight='bold',
                         color='white', zorder=3)

    # Bottom row: secondary structure -- grayscale only, simple shapes.
    # Helix = plain rectangle; strand = elongated triangle pointing right.
    y_bot, h_bot = 0.04, 0.26
    for name, start, end, kind in SECONDARY:
        if kind == 'helix':
            ax_anno.add_patch(Rectangle((start - 0.5, y_bot), end - start + 1, h_bot,
                                        facecolor=HELIX_FACE, edgecolor='none',
                                        zorder=2))
            ax_anno.text((start + end) / 2, y_bot + h_bot / 2, name,
                         ha='center', va='center', fontsize=15, weight='bold',
                         color='#222', zorder=3)
        else:
            # b-strand = plain rectangle (same shape as helix, darker shade).
            ax_anno.add_patch(Rectangle((start - 0.5, y_bot), end - start + 1, h_bot,
                                        facecolor=STRAND_FACE, edgecolor='none',
                                        zorder=2))
            ax_anno.text((start + end) / 2, y_bot + h_bot / 2, name,
                         ha='center', va='center', fontsize=15, weight='bold',
                         color='white', zorder=3)


def plot_assay(m6, assay, clusters_df, feats_df, desc_df, mode, outpath):
    """mode = 'permutation' (callout boxes = BH-significant features, q shown)
            or 'descriptive' (callout boxes = most-dropped features, no test)."""
    mat = ddg_matrix(m6, assay)
    pos_x = np.array(mat.columns)
    x_min, x_max = pos_x[0] - 1, pos_x[-1] + 1

    fig = plt.figure(figsize=(22, 15.0))
    # 5 rows: callout / heatmap / spacer / sequence / annotations
    # (the annotation strip carries regions, core/surface, secondary structure)
    gs = GridSpec(5, 1, figure=fig,
                  height_ratios=[4.5, 3.0, 0.60, 0.30, 1.30],
                  hspace=0.04,
                  left=0.045, right=0.965, top=0.96, bottom=0.04)
    ax_call = fig.add_subplot(gs[0])
    ax_hmap = fig.add_subplot(gs[1])
    ax_seq  = fig.add_subplot(gs[3])
    ax_anno = fig.add_subplot(gs[4])

    # ---------- heatmap ----------
    im = ax_hmap.imshow(mat.values, aspect='auto', cmap=CMAP,
                        vmin=DDG_VMIN, vmax=DDG_VMAX, interpolation='none',
                        extent=[pos_x[0] - 0.5, pos_x[-1] + 0.5,
                                len(AA_ORDER), 0])
    ax_hmap.set_xlim(x_min, x_max)
    ax_hmap.set_xlabel('Position', fontsize=16, weight='bold', labelpad=5)
    ax_hmap.spines['top'].set_visible(False)
    ax_hmap.spines['right'].set_visible(False)
    ax_hmap.set_ylabel('Mutant AA', fontsize=15, weight='bold')
    ax_hmap.set_yticks(np.arange(len(AA_ORDER)) + 0.5)
    ax_hmap.set_yticklabels(AA_ORDER, fontsize=13)
    xticks = list(np.arange(10, 189, 10))
    ax_hmap.set_xticks(xticks)
    ax_hmap.set_xticklabels(xticks, fontsize=13)
    # WT cells marked with a short, bold dash
    for p in pos_x:
        wt_aa = KRAS_WT[p - 1]
        if wt_aa in AA_ORDER:
            i = AA_ORDER.index(wt_aa)
            ax_hmap.plot([p - 0.32, p + 0.32], [i + 0.5, i + 0.5],
                         color='black', lw=1.5, solid_capstyle='butt', zorder=3)

    # ---------- sequence + annotation strips ----------
    region_by_pos, core_by_pos = load_kras_anno()
    draw_annotations(ax_seq, ax_anno, x_min, x_max, region_by_pos, core_by_pos)

    # Three-colour sequence legend, placed ABOVE the sequence strip (between
    # the heatmap and the sequence).
    fig.canvas.draw()
    seq_box = ax_seq.get_position()
    leg_y = seq_box.y1 + 0.011
    for i, (label, color) in enumerate([
        ('Binding interface', COLOR_INTERFACE),
        ('GTP pocket',        COLOR_GTP),
        ('Both',              COLOR_BOTH),
    ]):
        fig.text(0.05 + i * 0.115, leg_y, label, color=color,
                 fontsize=16, weight='bold', ha='left', va='bottom')

    # ---------- colour bar (tall, right of the heatmap, outside) ----------
    # Anchor to the heatmap's bbox so it always tracks the heatmap row.
    fig.canvas.draw()                                  # realise layout for get_position()
    hbox = ax_hmap.get_position()
    cax = fig.add_axes([hbox.x1 + 0.006, hbox.y0 + 0.04,
                        0.008, hbox.height - 0.08])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('ΔΔG (kcal/mol)', fontsize=12)
    cbar.ax.tick_params(labelsize=11)

    # ---------- callouts ----------
    ax_call.set_xlim(x_min, x_max)
    ax_call.set_ylim(0, 100)
    ax_call.axis('off')

    # ORDER BY POSITION so leader lines don't cross
    clust = (clusters_df[clusters_df['assay'] == assay]
             .sort_values('rep_pos').reset_index(drop=True))
    n_box = len(clust)

    # Box geometry in ax_call data units (y range 0..100). The box is sized
    # to match its content so there is no empty space between features.
    # 5 features x per_feat_h + small top/bottom margin.
    PER_FEAT_H = 13.0
    BY_BOT     = 2.0
    BY_HEIGHT  = 5 * PER_FEAT_H + 4.0    # = 69.0
    BY_TOP     = BY_BOT + BY_HEIGHT      # = 71.0
    BOX_X_MIN = pos_x[0] - 0.5
    BOX_X_MAX = pos_x[-1] + 0.5

    # ---------- separate callboxes, EQUALLY spaced across heatmap width ----------
    box_w  = (BOX_X_MAX - BOX_X_MIN) / n_box * 0.965   # small uniform gap
    box_xs = np.linspace(BOX_X_MIN + box_w / 2,
                         BOX_X_MAX - box_w / 2,
                         n_box)

    # Assay title only -- descriptive captions belong in the report, not the figure.
    fig.suptitle(assay, fontsize=18, x=0.045, ha='left', y=0.872, weight='bold')

    for i, hrow in clust.iterrows():
        bx        = box_xs[i]
        positions = list(hrow['positions'])
        cl_min, cl_max = min(positions), max(positions)

        # --- callout box ---
        rect = FancyBboxPatch((bx - box_w / 2, BY_BOT), box_w, BY_HEIGHT,
                              boxstyle='round,pad=0.30,rounding_size=0.50',
                              linewidth=1.0, edgecolor='#333',
                              facecolor='#f8fafc', zorder=4)
        ax_call.add_patch(rect)

        # --- callout content: up to 5 features ---
        # 'permutation' : the cluster's BH-significant features (q shown).
        #                 a cluster with no significant features shows a note.
        # 'descriptive' : the cluster's most-dropped features (mean drop shown),
        #                 ranked by mean max-drop; no significance test.
        cr = hrow['cluster_rank']
        if mode == 'permutation':
            sub = (feats_df[(feats_df['assay'] == assay)
                            & (feats_df['cluster_rank'] == cr)]
                   .sort_values('activation', ascending=False).head(5))
            entries = [dict(fid=int(r.feature_id), summary=r.summary,
                            right=f'q = {r.q_bh:.1e}')
                       for r in sub.itertuples()]
        else:
            sub = (desc_df[(desc_df['assay'] == assay)
                           & (desc_df['cluster_rank'] == cr)]
                   .sort_values('mean_max_drop', ascending=False).head(5))
            entries = [dict(fid=int(r.feature_id), summary=r.summary,
                            right=f'drop {r.mean_max_drop:.2f}')
                       for r in sub.itertuples()]

        feat_top   = BY_TOP - 1.4
        text_left  = bx - box_w / 2 + 0.9
        text_right = bx + box_w / 2 - 0.9

        if not entries:                               # permutation, 0 sig
            ax_call.text(bx, BY_BOT + BY_HEIGHT / 2,
                         'no features\npass q < 0.05',
                         ha='center', va='center', fontsize=10.5,
                         style='italic', color='#999', zorder=5)

        for j, e in enumerate(entries):
            y0 = feat_top - j * PER_FEAT_H
            ax_call.text(text_left, y0 - 0.2,
                         f"f{e['fid']}",
                         ha='left', va='top', fontsize=11.0,
                         family='monospace', weight='bold',
                         color='#222', zorder=5)
            ax_call.text(text_right, y0 - 0.2,
                         e['right'],
                         ha='right', va='top', fontsize=9.5,
                         family='monospace', color='#666', zorder=5)
            # wrap to comfortably fit inside the box (with margin); break long
            # words too so 25-char terms like "phosphotransferases" can't blow
            # past the right edge.
            ax_call.text(text_left, y0 - 2.5,
                         wrap(e['summary'], width=52, max_lines=4),
                         ha='left', va='top', fontsize=8.9,
                         color='#222', linespacing=1.05, zorder=5)
            if j < len(entries) - 1:
                ax_call.plot([bx - box_w / 2 + 0.3, bx + box_w / 2 - 0.3],
                             [y0 - PER_FEAT_H + 0.1, y0 - PER_FEAT_H + 0.1],
                             color='#e6e9ee', lw=0.4, zorder=5)

        # --- bracket on heatmap (no inverse triangle) ---
        bracket_y_bar = -0.18
        bracket_tick  = 0.34
        ax_hmap.plot([cl_min - 0.4, cl_max + 0.4], [bracket_y_bar, bracket_y_bar],
                     color='black', lw=2.0, clip_on=False, zorder=6,
                     solid_capstyle='butt')
        ax_hmap.plot([cl_min - 0.4, cl_min - 0.4],
                     [bracket_y_bar, bracket_y_bar + bracket_tick],
                     color='black', lw=2.0, clip_on=False, zorder=6)
        ax_hmap.plot([cl_max + 0.4, cl_max + 0.4],
                     [bracket_y_bar, bracket_y_bar + bracket_tick],
                     color='black', lw=2.0, clip_on=False, zorder=6)

        # leader line: ConnectionPatch re-resolves both endpoints at draw time,
        # so it stays attached to the box and bracket through savefig's relayout
        mid_pos = (cl_min + cl_max) / 2
        con = ConnectionPatch(
            xyA=(bx, BY_BOT),                        coordsA=ax_call.transData,
            xyB=(mid_pos, bracket_y_bar + 0.05),     coordsB=ax_hmap.transData,
            color='black', lw=1.8, zorder=5,
            capstyle='butt')
        fig.add_artist(con)

    plt.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'[plot] {assay:<12s} -> {outpath.name}')


def main():
    print('[load] MOESM6 + validation_2 tables')
    m6        = pd.read_excel(MOESM6, sheet_name='TableS5')
    clusters  = pd.read_parquet(V2 /'clusters.parquet')
    features  = pd.read_parquet(V2 /'cluster_features.parquet')
    desc_path = V2 /'descriptive_features.parquet'
    descriptive = pd.read_parquet(desc_path) if desc_path.exists() else None
    if descriptive is None:
        print('[warn] descriptive_features.parquet not found -- run '
              '09_validation2_descriptive.py first (needed for the descriptive callouts)')
    for a in ASSAYS:
        if (clusters['assay'] == a).sum() == 0:
            print(f'[skip] {a}: no clusters'); continue
        a_tag = a.replace(' ', '_')
        plot_assay(m6, a, clusters, features, descriptive, 'permutation',
                   V2 /f'callouts_permutation_{a_tag}.png')
        if descriptive is not None:
            plot_assay(m6, a, clusters, features, descriptive, 'descriptive',
                       V2 /f'callouts_descriptive_{a_tag}.png')


if __name__ == '__main__':
    main()
