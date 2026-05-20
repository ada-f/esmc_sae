"""
Dataset QC checks for poc_lof_variants.tsv.
scripts_v2 version: uses evidence-text variant set; no feature_type_overlap column.

Checks:
  1. Ref AA matches UniProt canonical sequence at position
  2. Each record is a single amino-acid substitution
  3. No positive variant is duplicated
  4. Positive and negative variants do not overlap
  5. Protein length <= 2700
  6. No variant_id is duplicated

Outputs:
  data/variants_with_evidence_text/qc_report.txt
  data/variants_with_evidence_text/qc_passed.tsv
"""

import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

from config import BASE_DATA
VARIANT_PATH = os.path.join(BASE_DATA, "data", "variants_with_evidence_text", "poc_lof_variants.tsv")
UNIPROT_JSONL = os.path.join(BASE_DATA, "data", "raw", "uniprot", "entries_full.jsonl.gz")
OUT_DIR = os.path.join(BASE_DATA, "data", "variants_with_evidence_text")

AA_VALID = set("ACDEFGHIKLMNPQRSTVWY")


def load_sequences():
    print("Loading canonical sequences...", flush=True)
    seqs = {}
    with gzip.open(UNIPROT_JSONL, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            acc = entry.get("primaryAccession", "")
            seq = entry.get("sequence", {}).get("value", "")
            seqs[acc] = seq
    print(f"Loaded sequences for {len(seqs)} UniProt entries")
    return seqs


def qc_check(df, seqs):
    issues = []
    passed_flags = [True] * len(df)

    def fail(i, msg):
        issues.append(f"Row {i} ({df.iloc[i]['variant_id']}): {msg}")
        passed_flags[i] = False

    print("Running QC checks...", flush=True)

    # Check 1: single AA substitution
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        ref = str(row["ref_aa"])
        alt = str(row["alt_aa"])
        if len(ref) != 1 or ref not in AA_VALID:
            fail(idx, f"invalid ref_aa '{ref}'")
        elif len(alt) != 1 or alt not in AA_VALID:
            fail(idx, f"invalid alt_aa '{alt}'")
        elif ref == alt:
            fail(idx, "ref_aa == alt_aa (synonymous)")

    # Check 2: ref AA matches canonical sequence
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        acc = row["uniprot_accession"]
        pos = int(row["position_1idx"])
        ref = str(row["ref_aa"])
        seq = seqs.get(acc, "")
        if not seq:
            fail(idx, f"no sequence found for {acc}")
        elif pos < 1 or pos > len(seq):
            fail(idx, f"position {pos} out of range for {acc} (len={len(seq)})")
        elif seq[pos - 1] != ref:
            fail(idx, f"ref_aa {ref} != sequence[{pos}] = {seq[pos-1]} for {acc}")

    # Check 3: protein length <= 2700
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        if int(row["protein_length"]) > 2700:
            fail(idx, f"protein_length {row['protein_length']} > 2700")

    # Check 4: no duplicate variant_id
    dup_ids = df["variant_id"].duplicated()
    for i, is_dup in enumerate(dup_ids):
        if is_dup:
            fail(i, "duplicate variant_id")

    # Check 5: positives and negatives do not overlap
    key_col = df["uniprot_accession"] + "_" + df["position_1idx"].astype(str) + "_" + df["alt_aa"]
    pos_keys = set(key_col[df["label"] == "LoF_like"])
    neg_keys = set(key_col[df["label"] != "LoF_like"])
    overlap = pos_keys & neg_keys
    if overlap:
        issues.append(f"CRITICAL: {len(overlap)} variants appear in both positive and negative sets: {list(overlap)[:5]}")

    return issues, passed_flags


def print_summary(df, f=None):
    def p(s):
        print(s)
        if f:
            f.write(s + "\n")

    p("\n" + "=" * 60)
    p("DATASET QC SUMMARY")
    p("=" * 60)
    p(f"\nTotal variants: {len(df)}")
    p(f"Unique proteins: {df['uniprot_accession'].nunique()}")
    p(f"Unique genes: {df['gene'].nunique()}")

    p("\n--- Variants by label ---")
    p(df["label"].value_counts().to_string())

    p("\n--- Variants by source ---")
    p(df["source"].value_counts().to_string())

    p("\n--- Variants by mechanism_tag ---")
    p(df["mechanism_tag"].value_counts().to_string())

    p("\n--- Label confidence ---")
    p(df["label_confidence"].value_counts().to_string())

    pos = df[df["label"] == "LoF_like"]
    neg = df[df["label"] != "LoF_like"]
    p(f"\n--- Positive/negative ratio ---")
    p(f"Positives: {len(pos)}, Negatives: {len(neg)}, Ratio: 1:{len(neg)/max(1,len(pos)):.1f}")

    p("\n--- Positives by mechanism_tag ---")
    p(pos["mechanism_tag"].value_counts().to_string())

    p("\n--- Protein length distribution (positives) ---")
    plen = pos["protein_length"]
    p(f"  min={plen.min()}, median={plen.median():.0f}, max={plen.max()}, mean={plen.mean():.0f}")

    p("\n--- Unique proteins in positive set: top genes ---")
    p(pos["gene"].value_counts().head(20).to_string())

    p("\n--- Nearby controls: distance summary ---")
    nb = df[df["label"] == "nearby_control"]
    if len(nb) > 0:
        dists = nb["notes"].str.extract(r"dist_to_nearest_site=(\d+)")[0].astype(float)
        p(f"  min dist={dists.min():.0f}, median={dists.median():.0f}, max={dists.max():.0f}")


def main():
    df = pd.read_csv(VARIANT_PATH, sep="\t")
    print(f"Loaded {len(df)} variants from {VARIANT_PATH}")

    seqs = load_sequences()
    issues, passed_flags = qc_check(df, seqs)

    qc_path = os.path.join(OUT_DIR, "qc_report.txt")
    with open(qc_path, "w") as f:
        print_summary(df, f)
        f.write("\n--- QC Issues ---\n")
        if issues:
            for issue in issues[:100]:
                print(f"  ISSUE: {issue}")
                f.write(f"  ISSUE: {issue}\n")
            if len(issues) > 100:
                f.write(f"  ... and {len(issues)-100} more\n")
        else:
            f.write("  No issues found!\n")
        f.write(f"\nTotal issues: {len(issues)}\n")
        f.write(f"Variants passing QC: {sum(passed_flags)}\n")

    print_summary(df)
    print(f"\nQC issues: {len(issues)}")
    for issue in issues[:20]:
        print(f"  ISSUE: {issue}")

    mask = np.array(passed_flags)
    df_passed = df[mask]
    passed_path = os.path.join(OUT_DIR, "qc_passed.tsv")
    df_passed.to_csv(passed_path, sep="\t", index=False)
    print(f"\nSaved {len(df_passed)} QC-passing variants to {passed_path}")
    print(f"QC report: {qc_path}")

    if issues:
        sys.exit(1)
    else:
        print("\nAll QC checks passed!")


if __name__ == "__main__":
    main()
