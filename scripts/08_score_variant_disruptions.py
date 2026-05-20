"""
Compute SAE disruption scores for all variants with delta data.
scripts_v2 version: reads/writes results_evidence_text/.

Inputs:
  results_evidence_text/tables/variant_feature_deltas.tsv.gz
  results_evidence_text/tables/variant_scores_raw.tsv
  data/uniref90_feature_table.parquet

Outputs:
  results_evidence_text/tables/variant_scores.tsv
  results_evidence_text/tables/variant_top_disrupted_features.tsv
"""

import os

import pandas as pd
import numpy as np

from config import BASE_DATA
DELTAS_PATH  = os.path.join(BASE_DATA, "results_evidence_text", "tables", "variant_feature_deltas.tsv.gz")
RAW_SCORES   = os.path.join(BASE_DATA, "results_evidence_text", "tables", "variant_scores_raw.tsv")
FEATURE_TABLE = os.path.join(BASE_DATA, "data", "uniref90_feature_table.parquet")
OUT_SCORES   = os.path.join(BASE_DATA, "results_evidence_text", "tables", "variant_scores.tsv")
OUT_TOP      = os.path.join(BASE_DATA, "results_evidence_text", "tables", "variant_top_disrupted_features.tsv")

FUNCTIONAL_CATEGORIES = {
    "Catalytic function",
    "Ligand-binding site",
    "Interaction site",
    "Post-translational modification",
    "Structural motif",
    "Domain",
    "Sequence motif",
}

MECHANISM_CATEGORIES = {
    "catalytic_LoF":          {"Catalytic function"},
    "binding_LoF":            {"Ligand-binding site", "Interaction site"},
    "metal_binding_LoF":      {"Ligand-binding site", "Structural motif"},
    "structural_stability_LoF": {"Structural motif"},
    "regulatory_LoF":         {"Post-translational modification"},
    "domain_or_motif_LoF":    {"Domain", "Sequence motif", "Structural motif"},
}

TOP_N = 10


def load_and_clean_scores() -> pd.DataFrame:
    df = pd.read_csv(RAW_SCORES, sep="\t")
    before = len(df)
    df = df.drop_duplicates(subset="variant_id", keep="last")
    if before != len(df):
        print(f"  Deduplicated scores: {before} -> {len(df)} rows")
    return df


def load_features() -> pd.DataFrame:
    ft = pd.read_parquet(FEATURE_TABLE)
    ft["is_functional"] = ft["category"].isin(FUNCTIONAL_CATEGORIES)
    ft["label_weight"] = ft["is_functional"].astype(float)
    return ft[["feature_id", "category", "summary", "label_weight", "is_functional", "threshold"]]


def compute_variant_scores(deltas: pd.DataFrame, features: pd.DataFrame,
                           raw_scores: pd.DataFrame) -> pd.DataFrame:
    d = deltas.merge(features[["feature_id", "category", "label_weight", "is_functional"]],
                     on="feature_id", how="left")
    d["label_weight"] = d["label_weight"].fillna(0.0)

    results = []
    for vid, grp in d.groupby("variant_id"):
        functional = grp[grp["is_functional"]]

        lof_score = (functional["feature_loss"] * functional["label_weight"]).sum()

        sorted_loss = functional["feature_loss"].sort_values(ascending=False)
        max_functional_loss = sorted_loss.iloc[0] if len(sorted_loss) > 0 else 0.0
        mean_top5_functional_loss = sorted_loss.iloc[:5].mean() if len(sorted_loss) > 0 else 0.0

        cat_scores = {}
        for mech, cats in MECHANISM_CATEGORIES.items():
            subset = grp[grp["category"].isin(cats)]
            cat_scores[f"max_{mech}_feature_loss"] = (
                subset["feature_loss"].max() if len(subset) > 0 else 0.0
            )

        total_loss = grp["feature_loss"].sum()
        functional_loss_fraction = (
            functional["feature_loss"].sum() / total_loss if total_loss > 0 else 0.0
        )

        results.append({
            "variant_id": vid,
            "LoF_functional_feature_loss_score": lof_score,
            "max_functional_feature_loss": max_functional_loss,
            "mean_top5_functional_feature_loss": mean_top5_functional_loss,
            "functional_loss_fraction": functional_loss_fraction,
            "n_functional_features_active": int((functional["feature_loss"] > 0).sum()),
            **cat_scores,
        })

    scores_df = pd.DataFrame(results)
    merged = raw_scores.merge(scores_df, on="variant_id", how="left")
    return merged


def compute_top_features(deltas: pd.DataFrame, features: pd.DataFrame,
                         raw_scores: pd.DataFrame) -> pd.DataFrame:
    d = deltas.merge(features[["feature_id", "category", "summary",
                                "label_weight", "is_functional"]],
                     on="feature_id", how="left")
    d["label_weight"] = d["label_weight"].fillna(0.0)

    meta_cols = raw_scores[["variant_id", "label", "mechanism_tag"]].drop_duplicates()

    rows = []
    for vid, grp in d.groupby("variant_id"):
        grp = grp.copy()
        grp["abs_delta"] = grp["delta_window_max"].abs()
        grp["rank_by_loss"]      = grp["feature_loss"].rank(ascending=False, method="first").astype(int)
        grp["rank_by_gain"]      = grp["feature_gain"].rank(ascending=False, method="first").astype(int)
        grp["rank_by_abs_delta"] = grp["abs_delta"].rank(ascending=False, method="first").astype(int)

        top_mask = (
            (grp["rank_by_loss"]      <= TOP_N) |
            (grp["rank_by_gain"]      <= TOP_N) |
            (grp["rank_by_abs_delta"] <= TOP_N)
        )
        top = grp[top_mask].copy()
        rows.append(top)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.merge(meta_cols, on="variant_id", how="left")

    col_order = [
        "variant_id", "label", "mechanism_tag",
        "feature_id", "category", "summary", "label_weight", "is_functional",
        "ref_site", "mut_site", "delta_site",
        "ref_window_max", "mut_window_max", "delta_window_max",
        "feature_loss", "feature_gain", "abs_delta",
        "rank_by_abs_delta", "rank_by_loss", "rank_by_gain",
    ]
    return out[[c for c in col_order if c in out.columns]]


def main():
    print("Loading inputs ...")
    raw_scores = load_and_clean_scores()
    features   = load_features()

    print(f"Loading deltas ({DELTAS_PATH}) ...")
    deltas = pd.read_csv(DELTAS_PATH, sep="\t", compression="gzip")
    print(f"  {len(deltas):,} rows, {deltas['variant_id'].nunique()} unique variants")

    print("\nComputing per-variant disruption scores ...")
    scores = compute_variant_scores(deltas, features, raw_scores)
    scores.to_csv(OUT_SCORES, sep="\t", index=False)
    print(f"Saved {len(scores)} rows to {OUT_SCORES}")

    print("\nComputing top disrupted features per variant ...")
    top = compute_top_features(deltas, features, raw_scores)
    top.to_csv(OUT_TOP, sep="\t", index=False)
    print(f"Saved {len(top):,} rows to {OUT_TOP}")

    print("\n=== Score summary by label ===")
    score_cols = [
        "LoF_functional_feature_loss_score",
        "max_functional_feature_loss",
        "functional_loss_fraction",
    ]
    rows = []
    for label, grp in scores.groupby("label"):
        row = {"label": label, "n": len(grp)}
        for col in score_cols:
            row[f"{col}_mean"]   = grp[col].mean()
            row[f"{col}_median"] = grp[col].median()
            row[f"{col}_std"]    = grp[col].std()
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("label")

    for col in score_cols:
        print(f"\n  {col}")
        print(f"  {'label':<20} {'n':>6}  {'mean':>8}  {'median':>8}  {'std':>8}")
        print(f"  {'-'*54}")
        for label, r in summary.iterrows():
            print(f"  {label:<20} {int(r['n']):>6}  "
                  f"{r[f'{col}_mean']:>8.4f}  "
                  f"{r[f'{col}_median']:>8.4f}  "
                  f"{r[f'{col}_std']:>8.4f}")

    print("\n=== LoF variants by mechanism (median) ===")
    lof = scores[scores["label"] == "LoF_like"]
    mech_cols = [c for c in scores.columns if c.startswith("max_") and c.endswith("_feature_loss")]
    print(lof.groupby("mechanism_tag")[mech_cols].agg(["median", "std"]).round(4).to_string())


if __name__ == "__main__":
    main()
