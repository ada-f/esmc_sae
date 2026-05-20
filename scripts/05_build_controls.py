"""
Build negative control sets and assemble the final variant table.
scripts_v2 version: uses evidence-text positive set; no feature_type_overlap column.

Negative set A: ClinVar benign missense variants
  - Matched to UniProt canonical sequences via gene symbol
  - Ref AA validated against UniProt sequence
  - Not overlapping any functional site in that UniProt entry

Negative set B: AlphaMissense 'benign' nearby controls
  - AlphaMissense (am_class == 'benign') variants in the same proteins as positives
  - Ref AA validated against UniProt canonical sequence
  - Not overlapping any functional site
  - At least MIN_DIST_FROM_SITE residues from any functional site
  - "Nearby": within NEARBY_WINDOW residues of at least one positive variant in the
    same protein (ensures matched local sequence context), OR in the same annotated
    domain as a positive variant
  - Sampled to MAX_PER_PROTEIN per protein

Output:
  data/variants_with_evidence_text/negative_set_A.tsv
  data/variants_with_evidence_text/negative_set_B.tsv
  data/variants_with_evidence_text/poc_lof_variants.tsv  (final combined table)
"""

import gzip
import json
import os
import random

import pandas as pd

from config import BASE_DATA
UNIPROT_JSONL = os.path.join(BASE_DATA, "data", "raw", "uniprot", "entries_full.jsonl.gz")
POSITIVE_PATH = os.path.join(BASE_DATA, "data", "variants_with_evidence_text", "positive_tier1.tsv")
CLINVAR_PATH = os.path.join(BASE_DATA, "data", "raw", "clinvar", "clinvar_benign_missense.tsv.gz")
AM_BENIGN_PATH = os.path.join(
    BASE_DATA, "data", "raw", "alphamissense", "am_likely_benign_positive_proteins.tsv.gz"
)
OUT_DIR = os.path.join(BASE_DATA, "data", "variants_with_evidence_text")
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)

FUNCTIONAL_SITE_TYPES = {
    "Active site", "Binding site", "Metal binding", "Disulfide bond",
    "Modified residue", "Lipidation", "Glycosylation", "Cross-link",
    "Site", "Motif", "Region", "Domain", "DNA binding", "Zinc finger",
}
MAX_PROTEIN_LENGTH = 2700
MIN_DIST_FROM_SITE = 10
NEARBY_WINDOW = 50
MAX_PER_PROTEIN = 10

# No feature_type_overlap or domain_overlap columns in this pipeline
FINAL_COLS = [
    "variant_id", "source", "label", "label_confidence", "mechanism_tag",
    "uniprot_accession", "gene", "protein_name", "protein_length",
    "position_1idx", "ref_aa", "alt_aa", "mutation_hgvs_p",
    "sequence_context_21aa", "evidence_text", "include_in_poc", "notes",
]


# ── helpers ────────────────────────────────────────────────────────────────────

def load_uniprot_index():
    print("Loading UniProt entries...", flush=True)
    gene2acc = {}
    acc2entry = {}
    with gzip.open(UNIPROT_JSONL, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1000 == 0 and i > 0:
                print(f"  {i} entries loaded...", flush=True)
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            acc = entry.get("primaryAccession", "")
            seq = entry.get("sequence", {}).get("value", "")
            genes = entry.get("genes", [])
            gene = genes[0].get("geneName", {}).get("value", "") if genes else ""
            protein = entry.get("proteinDescription", {})
            rec = protein.get("recommendedName", {})
            protein_name = rec.get("fullName", {}).get("value", "")
            sites = []
            for feat in entry.get("features", []):
                ftype = feat.get("type", "")
                if ftype not in FUNCTIONAL_SITE_TYPES:
                    continue
                loc = feat.get("location", {})
                s = loc.get("start", {}).get("value")
                e = loc.get("end", {}).get("value")
                if s is not None and e is not None:
                    sites.append((ftype, int(s), int(e), feat.get("description", "")))
            acc2entry[acc] = {
                "seq": seq, "gene": gene, "protein_name": protein_name,
                "protein_length": len(seq), "functional_sites": sites,
            }
            if gene:
                gene2acc.setdefault(gene, []).append(acc)
    print(f"Loaded {len(acc2entry)} entries, {len(gene2acc)} genes")
    return gene2acc, acc2entry


def is_functional_site(sites, pos):
    return any(s <= pos <= e for _, s, e, _ in sites)


def min_dist_to_any_site(sites, pos):
    if not sites:
        return 9999
    return min(min(abs(pos - s), abs(pos - e)) for _, s, e, _ in sites)


def get_domain_at(entry_data, pos):
    for ftype, s, e, desc in entry_data["functional_sites"]:
        if ftype in ("Domain", "Region", "Motif", "Zinc finger") and s <= pos <= e:
            return desc
    return ""


def sequence_context(seq, pos_1idx, window=10):
    idx = pos_1idx - 1
    start = max(0, idx - window)
    end = min(len(seq), idx + window + 1)
    left_pad = "-" * max(0, window - idx)
    right_pad = "-" * max(0, idx + window + 1 - len(seq))
    return left_pad + seq[start:end] + right_pad


# ── Negative set A: ClinVar benign ─────────────────────────────────────────────

def build_negative_A(gene2acc, acc2entry, positive_df):
    print("\nBuilding Negative Set A (ClinVar benign)...", flush=True)
    positive_keys = set(
        zip(positive_df["uniprot_accession"], positive_df["position_1idx"], positive_df["alt_aa"])
    )
    clinvar = pd.read_csv(CLINVAR_PATH, sep="\t", compression="gzip", low_memory=False)
    print(f"ClinVar benign missense loaded: {len(clinvar):,}")
    records = []
    n_no_uniprot = n_bad_ref = n_on_site = n_dup = 0

    for _, row in clinvar.iterrows():
        gene = str(row["gene"]).strip()
        ref_aa = str(row["ref_aa"]).strip()
        alt_aa = str(row["alt_aa"]).strip()
        pos = int(row["position_1idx"])
        accs = gene2acc.get(gene, [])
        if not accs:
            n_no_uniprot += 1
            continue
        acc = accs[0]
        entry_data = acc2entry[acc]
        seq = entry_data["seq"]
        if not seq or pos < 1 or pos > len(seq):
            n_bad_ref += 1
            continue
        if seq[pos - 1] != ref_aa:
            n_bad_ref += 1
            continue
        if is_functional_site(entry_data["functional_sites"], pos):
            n_on_site += 1
            continue
        if (acc, pos, alt_aa) in positive_keys:
            n_dup += 1
            continue
        if entry_data["protein_length"] > MAX_PROTEIN_LENGTH:
            continue
        records.append({
            "variant_id": f"{acc}_{ref_aa}{pos}{alt_aa}_ClinVarBenign",
            "source": "CLINVAR_BENIGN",
            "label": "benign_control",
            "label_confidence": "high",
            "mechanism_tag": "benign_control",
            "uniprot_accession": acc,
            "gene": entry_data["gene"],
            "protein_name": entry_data["protein_name"],
            "protein_length": entry_data["protein_length"],
            "position_1idx": pos,
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "mutation_hgvs_p": f"p.{ref_aa}{pos}{alt_aa}",
            "sequence_context_21aa": sequence_context(seq, pos),
            "evidence_text": (
                f"ClinVar: {row.get('clinical_significance','')} "
                f"({row.get('review_status','')})"
            ),
            "include_in_poc": True,
            "notes": f"variation_id={row.get('variation_id','')}",
        })

    print(f"  No UniProt match: {n_no_uniprot:,} | Bad ref: {n_bad_ref:,} | "
          f"On site: {n_on_site:,} | Dup: {n_dup:,} | Accepted: {len(records):,}")
    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.drop_duplicates(subset=["uniprot_accession", "position_1idx", "alt_aa"])
    return df


# ── Negative set B: AlphaMissense benign nearby controls ───────────────────────

def build_negative_B(acc2entry, positive_df):
    print("\nBuilding Negative Set B (AlphaMissense benign nearby controls)...", flush=True)

    pos_positions_by_acc = (
        positive_df.groupby("uniprot_accession")["position_1idx"]
        .apply(set)
        .to_dict()
    )

    positive_keys = set(
        zip(positive_df["uniprot_accession"], positive_df["position_1idx"], positive_df["alt_aa"])
    )

    print(f"  Loading AlphaMissense benign variants from {AM_BENIGN_PATH} ...", flush=True)
    am_df = pd.read_csv(AM_BENIGN_PATH, sep="\t", compression="gzip")
    print(f"  Loaded {len(am_df):,} AlphaMissense benign records for positive-set proteins")

    am_by_acc = am_df.groupby("uniprot_accession")

    records = []
    n_proteins_with_controls = 0
    n_no_am = 0
    n_bad_ref = 0
    n_on_site = 0
    n_not_nearby = 0
    n_dup = 0

    for acc, pos_positions in pos_positions_by_acc.items():
        entry_data = acc2entry.get(acc)
        if entry_data is None:
            continue
        seq = entry_data["seq"]
        if not seq or entry_data["protein_length"] > MAX_PROTEIN_LENGTH:
            continue

        sites = entry_data["functional_sites"]

        if acc not in am_by_acc.groups:
            n_no_am += 1
            continue
        candidates_df = am_by_acc.get_group(acc)

        positive_domains = set()
        for p in pos_positions:
            d = get_domain_at(entry_data, p)
            if d:
                positive_domains.add(d)

        valid = []
        for _, row in candidates_df.iterrows():
            pos = int(row["position_1idx"])
            ref_aa = str(row["ref_aa"])
            alt_aa = str(row["alt_aa"])

            if pos < 1 or pos > len(seq) or seq[pos - 1] != ref_aa:
                n_bad_ref += 1
                continue

            if is_functional_site(sites, pos):
                n_on_site += 1
                continue

            if min_dist_to_any_site(sites, pos) < MIN_DIST_FROM_SITE:
                n_on_site += 1
                continue

            nearby = any(abs(pos - p) <= NEARBY_WINDOW for p in pos_positions)
            same_domain = False
            if positive_domains:
                d = get_domain_at(entry_data, pos)
                same_domain = d in positive_domains

            if not (nearby or same_domain):
                n_not_nearby += 1
                continue

            if (acc, pos, alt_aa) in positive_keys:
                n_dup += 1
                continue

            valid.append((pos, ref_aa, alt_aa, float(row["am_pathogenicity"])))

        if not valid:
            continue

        valid.sort(key=lambda x: x[3])
        sampled = valid[:MAX_PER_PROTEIN]

        n_proteins_with_controls += 1
        for pos, ref_aa, alt_aa, am_score in sampled:
            records.append({
                "variant_id": f"{acc}_{ref_aa}{pos}{alt_aa}_AM_Benign",
                "source": "ALPHAMISSENSE_BENIGN",
                "label": "nearby_control",
                "label_confidence": "high",
                "mechanism_tag": "nearby_control",
                "uniprot_accession": acc,
                "gene": entry_data["gene"],
                "protein_name": entry_data["protein_name"],
                "protein_length": entry_data["protein_length"],
                "position_1idx": pos,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "mutation_hgvs_p": f"p.{ref_aa}{pos}{alt_aa}",
                "sequence_context_21aa": sequence_context(seq, pos),
                "evidence_text": (
                    f"AlphaMissense: am_class=benign, am_pathogenicity={am_score:.4f}"
                ),
                "include_in_poc": True,
                "notes": (
                    f"am_pathogenicity={am_score:.4f}; "
                    f"dist_to_nearest_site={min_dist_to_any_site(sites, pos)}"
                ),
            })

    print(f"  No AlphaMissense data: {n_no_am} | Bad ref: {n_bad_ref:,} | "
          f"On/near site: {n_on_site:,} | Not nearby: {n_not_nearby:,} | "
          f"Dup: {n_dup:,}")
    print(f"  Proteins with ≥1 control: {n_proteins_with_controls} / {len(pos_positions_by_acc)}")
    print(f"  Total nearby controls: {len(records)}")
    return pd.DataFrame(records)


# ── Final assembly ──────────────────────────────────────────────────────────────

def main():
    gene2acc, acc2entry = load_uniprot_index()

    positive_df = pd.read_csv(POSITIVE_PATH, sep="\t")
    print(f"\nPositive set: {len(positive_df)} variants from "
          f"{positive_df['uniprot_accession'].nunique()} proteins")

    neg_A = build_negative_A(gene2acc, acc2entry, positive_df)
    neg_B = build_negative_B(acc2entry, positive_df)

    neg_A.to_csv(os.path.join(OUT_DIR, "negative_set_A.tsv"), sep="\t", index=False)
    neg_B.to_csv(os.path.join(OUT_DIR, "negative_set_B.tsv"), sep="\t", index=False)
    print(f"\nNeg A saved: {len(neg_A)} variants")
    print(f"Neg B saved: {len(neg_B)} variants")

    n_pos = len(positive_df)
    neg_A_sample = (
        neg_A.sample(min(len(neg_A), n_pos * 3), random_state=42)
        if len(neg_A) > 0 else neg_A
    )

    combined = pd.concat(
        [positive_df[FINAL_COLS], neg_A_sample[FINAL_COLS], neg_B[FINAL_COLS]],
        ignore_index=True,
    )
    combined = combined.drop_duplicates(
        subset=["uniprot_accession", "position_1idx", "alt_aa"]
    )

    out_path = os.path.join(OUT_DIR, "poc_lof_variants.tsv")
    combined.to_csv(out_path, sep="\t", index=False)

    print(f"\n=== Final variant table: {len(combined)} variants ===")
    print(combined["label"].value_counts().to_string())
    print(f"\nUnique proteins: {combined['uniprot_accession'].nunique()}")
    print(f"Unique genes: {combined['gene'].nunique()}")
    print("\nMechanism tag breakdown:")
    print(combined["mechanism_tag"].value_counts().to_string())
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
