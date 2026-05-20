"""
Generate figures for the SAE feature disruption POC.
scripts_v2 version: reads from results_evidence_text/, writes to results_evidence_text/figures/.

Figure 1: Heatmap — mechanism_tag (UniProt evidence text) × SAE feature category
           Values: median max feature_loss per (mechanism, category) cell,
           normalised by the nearby_control baseline so the diagonal
           shows fold-enrichment over controls.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from config import BASE_DATA
DELTAS_PATH = os.path.join(BASE_DATA, "results_evidence_text", "tables", "variant_feature_deltas.tsv.gz")
SCORES_PATH = os.path.join(BASE_DATA, "results_evidence_text", "tables", "variant_scores.tsv")
FEATURE_TABLE = os.path.join(BASE_DATA, "data", "uniref90_feature_table.parquet")
FIG_DIR     = os.path.join(BASE_DATA, "results_evidence_text", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

FUNCTIONAL_CATEGORIES = [
    "Catalytic function",
    "Ligand-binding site",
    "Interaction site",
    "Structural motif",
    "Domain",
    "Post-translational modification",
    "Sequence motif",
]

MECHANISM_ORDER = [
    "catalytic_LoF",
    "binding_LoF",
    "metal_binding_LoF",
    "structural_stability_LoF",
    "regulatory_LoF",
    "domain_or_motif_LoF",
]

MECHANISM_LABELS = {
    "catalytic_LoF":           "Catalytic",
    "binding_LoF":             "Binding",
    "metal_binding_LoF":       "Metal binding",
    "structural_stability_LoF":"Structural\nstability",
    "regulatory_LoF":          "Regulatory\n(PTM)",
    "domain_or_motif_LoF":     "Domain /\nmotif",
}

CATEGORY_LABELS = {
    "Catalytic function":            "Catalytic\nfunction",
    "Ligand-binding site":           "Ligand-\nbinding",
    "Interaction site":              "Interaction\nsite",
    "Structural motif":              "Structural\nmotif",
    "Domain":                        "Domain",
    "Post-translational modification": "PTM",
    "Sequence motif":                "Sequence\nmotif",
}


def compute_heatmap_values(deltas, features, scores):
    d = deltas.merge(
        features[["feature_id", "category"]],
        on="feature_id", how="left"
    )
    d = d[d["category"].isin(FUNCTIONAL_CATEGORIES)]

    meta = scores[["variant_id", "label", "mechanism_tag"]]
    d = d.merge(meta, on="variant_id", how="inner")

    var_cat = (
        d.groupby(["variant_id", "label", "mechanism_tag", "category"])["feature_loss"]
        .max()
        .reset_index()
    )

    lof = var_cat[var_cat["label"] == "LoF_like"]
    ctrl = var_cat[var_cat["label"] == "nearby_control"]

    lof_pivot = (
        lof.groupby(["mechanism_tag", "category"])["feature_loss"]
        .median()
        .unstack("category")
        .reindex(index=MECHANISM_ORDER, columns=FUNCTIONAL_CATEGORIES)
        .fillna(0.0)
    )

    ctrl_baseline = (
        ctrl.groupby("category")["feature_loss"]
        .median()
        .reindex(FUNCTIONAL_CATEGORIES)
        .fillna(1e-9)
    )

    fold = lof_pivot.div(ctrl_baseline, axis=1)

    return lof_pivot, ctrl_baseline, fold


def plot_heatmap(lof_pivot, ctrl_baseline, fold, n_lof_by_mech):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.45})

    row_labels = [MECHANISM_LABELS[m] for m in MECHANISM_ORDER]
    col_labels = [CATEGORY_LABELS[c] for c in FUNCTIONAL_CATEGORIES]

    def _heatmap(ax, data, title, cmap, fmt, vmin, vmax, cbar_label):
        im = ax.imshow(data.values, cmap=cmap, aspect="auto",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8, ha="center")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("SAE feature category", fontsize=9)
        ax.set_ylabel("UniProt mechanism tag", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=10)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data.values[i, j]
                text = fmt.format(val)
                brightness = (im.norm(val) if vmax > vmin else 0.5)
                color = "white" if brightness > 0.6 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=7, color=color)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label(cbar_label, fontsize=8)

        for i, mech in enumerate(MECHANISM_ORDER):
            n = n_lof_by_mech.get(mech, 0)
            ax.text(data.shape[1] - 0.5 + 0.7, i, f"n={n}",
                    va="center", fontsize=7, color="#555555")

    _heatmap(axes[0], lof_pivot,
             title="Median max feature loss\n(LoF variants)",
             cmap="YlOrRd", fmt="{:.3f}",
             vmin=0, vmax=lof_pivot.values.max(),
             cbar_label="Median max feature_loss")

    fold_clipped = fold.clip(upper=4.0)
    _heatmap(axes[1], fold_clipped,
             title="Fold enrichment over\nnearby_control baseline\n(1× = no enrichment)",
             cmap="PuOr", fmt="{:.1f}×",
             vmin=0, vmax=4.0,
             cbar_label="Fold enrichment (capped at 4×)")

    plt.suptitle(
        "SAE feature disruption: UniProt mechanism vs SAE feature category\n"
        f"({sum(n_lof_by_mech.values())} LoF variants, {len(FUNCTIONAL_CATEGORIES)} SAE categories)\n"
        "[evidence-text mechanism assignment]",
        fontsize=11, y=1.02
    )

    out = os.path.join(FIG_DIR, "fig_mechanism_category_heatmap.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=150)
    out_png = out.replace(".pdf", ".png")
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"Saved: {out_png}")
    return out_png


def main():
    print("Loading data ...")
    scores  = pd.read_csv(SCORES_PATH, sep="\t")
    features = pd.read_parquet(FEATURE_TABLE)[["feature_id", "category"]]
    deltas  = pd.read_csv(DELTAS_PATH, sep="\t", compression="gzip")

    print("Computing heatmap values ...")
    lof_pivot, ctrl_baseline, fold = compute_heatmap_values(deltas, features, scores)

    lof_scores = scores[scores["label"] == "LoF_like"]
    n_lof_by_mech = lof_scores["mechanism_tag"].value_counts().to_dict()

    print("\nMedian max feature_loss (LoF variants):")
    print(lof_pivot.round(4).to_string())
    print("\nNearby_control baseline (median per category):")
    print(ctrl_baseline.round(4).to_string())
    print("\nFold enrichment:")
    print(fold.round(2).to_string())

    print("\nGenerating heatmap ...")
    plot_heatmap(lof_pivot, ctrl_baseline, fold, n_lof_by_mech)


if __name__ == "__main__":
    main()
