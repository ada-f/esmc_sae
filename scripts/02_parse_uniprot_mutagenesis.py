"""
Parse UniProt JSON entries to extract Tier 1 LoF-like mutagenesis variants,
assigning mechanism from evidence text rather than positional functional-site overlap.

This is the scripts_v2 version of the parsing pipeline. It combines steps 02 and 03
from scripts/ into a single script using text-based mechanism assignment.

Starting from the same raw JSONL it:
  - Applies identical single-AA substitution and ref-AA validation filters as 02
  - Applies identical LoF phrase / exclusion phrase filters as 02
  - Assigns mechanism_tag by keyword-matching the evidence text (not position overlap)
  - Drops variants where evidence text does not indicate a clear mechanism
  - Applies the same protein-length, standard-AA, and dedup filters as 03

Mechanism priority (highest to lowest):
  catalytic_LoF         -> enzymatic/catalytic language (kinase, GTPase, ase activity ...)
  metal_binding_LoF     -> metal/zinc/iron-specific language
  structural_stability_LoF -> disulfide, stability, folding language
  regulatory_LoF        -> PTM language (phosphorylat, glycosylat, ubiquitinat ...)
  binding_LoF           -> binding/interaction language (broad but reliable post-LoF filter)
  domain_or_motif_LoF   -> localization, transport, transcription, signaling language

Outputs:
  data/raw/uniprot/parsed_mutagenesis_lof_evidence.tsv  -- LoF records with text mechanism
  data/variants_with_evidence_text/positive_tier1.tsv   -- Tier 1 positive set
"""

import gzip
import json
import os

import pandas as pd

from config import BASE_DATA
IN_PATH = os.path.join(BASE_DATA, "data", "raw", "uniprot", "entries_full.jsonl.gz")
OUT_RAW_DIR = os.path.join(BASE_DATA, "data", "raw", "uniprot")
OUT_VAR_DIR = os.path.join(BASE_DATA, "data", "variants_with_evidence_text")
os.makedirs(OUT_RAW_DIR, exist_ok=True)
os.makedirs(OUT_VAR_DIR, exist_ok=True)

# ── LoF phrase lists (identical to 02) ───────────────────────────────────────

LOF_HIGH = [
    "abolishes activity",
    "abolished activity",
    "no activity",
    "no detectable activity",
    "inactive",
    "catalytically inactive",
    "loss of function",
    "loss of activity",
    "loss of binding",
    "unable to bind",
    "eliminates activity",
    "abolishes binding",
    "abrogates activity",
    "abrogates binding",
    "completely abolish",
    "abolishes catalytic",
    "abolishes enzymatic",
    "no enzymatic activity",
    "no catalytic activity",
    "renders enzyme inactive",
    "abolishes the activity",
    "completely inactive",
    "totally inactive",
    "complete loss",
    "results in complete loss",
    "loss of enzyme activity",
]

LOF_EXCLUSION = [
    "no effect",
    "does not affect",
    "does not abolish",
    "unchanged",
    "retains activity",
    "retains binding",
    "similar to wild type",
    "similar to wild-type",
    "increased activity",
    "enhanced activity",
    "gain of function",
    "activating",
    "constitutively active",
    "hyperactivat",
    "not required",
    "no significant effect",
    "little effect",
    "reduces but does not abolish",
    "partial",
    "mildly",
    "mild reduction",
]

# ── Evidence-text mechanism patterns (priority order) ─────────────────────────
# Each tuple: (mechanism_tag, [list of lowercase substrings to match])
# First matching pattern wins.

TEXT_MECHANISM_PATTERNS = [
    ("catalytic_LoF", [
        "catalytic",        # catalytic activity, catalytically, catalysis
        "enzymatic",        # enzymatic activity
        "enzyme activity",
        "active site",
        "ase activity",     # catches kinase activity, GTPase activity, transferase activity ...
        "transferase",      # fucosyltransferase, methyltransferase, etc.
        "peptidase",
        "protease",
        "nuclease",
        "gtpase",
        "atpase",
        "gef",              # guanine nucleotide exchange factor activity
        "autophosphorylat", # implies kinase
        "coenzyme",
        "cofactor",
        "hydrolyze",        # hydrolyzes, hydrolyzed
        "hydrolyses",
        "hydrolysis",
        "hydrolytic",
        "hydrolase",
        "isomerase",
        "oxidase",
        "reductase",
        "ligase activity",
        "lyase",
        "substrate bind",   # enzyme-substrate binding
        "active-site",
    ]),
    ("metal_binding_LoF", [
        "metal bind",
        "metal-bind",
        "metal coordination",
        "metal ion",
        "zinc bind",
        "zinc-bind",
        "iron bind",
        "copper bind",
        "manganese",
        "metalloenzyme",
        "metal chelat",
    ]),
    ("structural_stability_LoF", [
        "disulfide",
        "disulphide",
        "protein stability",
        "structural stability",
        "thermal stability",
        "misfolding",
        "aggregation",
        "unfolded",
        "destabiliz",
        "protein folding",
    ]),
    ("regulatory_LoF", [
        "phosphorylat",
        "glycosylat",
        "lipidation",
        "acetylat",
        "methylat",
        "ubiquitinat",
        "sumoylat",
        "palmitoylat",
        "myristoylat",
        "neddylat",
        "prenylat",
        "farnesylat",
        "geranylgeranylat",
    ]),
    ("binding_LoF", [
        "bind",             # binding, binds, bound, unable to bind
        "interact",         # interaction, interacts
        "affinity",
        # "associat" intentionally excluded — "when associated with [mutation]" is a
        # UniProt annotation idiom unrelated to protein association/binding
        "complex formation",
        "dna",              # DNA-binding implied
        "rna",              # RNA-binding implied
        "recognition",
    ]),
    ("domain_or_motif_LoF", [
        "localiz",          # localization, localizes
        "nuclear export",
        "nuclear import",
        "membrane target",
        "membrane local",
        "secretion",
        "sorting signal",
        "transit peptide",
        "signal peptide",
        "transport",        # ion/protein transport
        "transcription",    # transcriptional activity
        "signaling",
        "signal transduct",
    ]),
]

# ── 03-equivalent filters ─────────────────────────────────────────────────────

MAX_PROTEIN_LENGTH = 2700
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def has_lof_phrase(text):
    t = text.lower()
    return any(p in t for p in LOF_HIGH)


def has_exclusion_phrase(text):
    t = text.lower()
    return any(p in t for p in LOF_EXCLUSION)


def assign_mechanism_from_text(evidence_text):
    t = evidence_text.lower()
    for mechanism_tag, keywords in TEXT_MECHANISM_PATTERNS:
        if any(kw in t for kw in keywords):
            return mechanism_tag
    return ""


def get_sequence(entry):
    return entry.get("sequence", {}).get("value", "")


def get_gene_name(entry):
    genes = entry.get("genes", [])
    if not genes:
        return ""
    return genes[0].get("geneName", {}).get("value", "")


def get_protein_name(entry):
    protein = entry.get("proteinDescription", {})
    rec = protein.get("recommendedName", {})
    return rec.get("fullName", {}).get("value", "")


def sequence_context(seq, pos_1idx, window=10):
    idx = pos_1idx - 1
    start = max(0, idx - window)
    end = min(len(seq), idx + window + 1)
    left_pad = "-" * max(0, window - idx)
    right_pad = "-" * max(0, idx + window + 1 - len(seq))
    return left_pad + seq[start:end] + right_pad


def parse_entries():
    records_lof = []
    n_entries = 0

    print(f"Reading {IN_PATH} ...", flush=True)
    with gzip.open(IN_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_entries += 1
            if n_entries % 500 == 0:
                print(f"  {n_entries} entries, {len(records_lof)} LoF so far...", flush=True)

            acc = entry.get("primaryAccession", "")
            gene = get_gene_name(entry)
            protein_name = get_protein_name(entry)
            seq = get_sequence(entry)
            protein_length = len(seq)

            mutagenesis_features = [
                f for f in entry.get("features", []) if f.get("type") == "Mutagenesis"
            ]
            if not mutagenesis_features:
                continue

            for feat in mutagenesis_features:
                loc = feat.get("location", {})
                pos_start = loc.get("start", {}).get("value")
                pos_end = loc.get("end", {}).get("value")
                if pos_start is None or pos_end is None:
                    continue
                pos_start, pos_end = int(pos_start), int(pos_end)
                if pos_start != pos_end:
                    continue
                pos_1idx = pos_start

                alt_seq_obj = feat.get("alternativeSequence", {})
                original_seq = alt_seq_obj.get("originalSequence", "")
                alt_seqs = alt_seq_obj.get("alternativeSequences", [])
                evidence_text = feat.get("description", "")

                if not original_seq or not alt_seqs:
                    continue
                if len(original_seq) != 1:
                    continue

                ref_aa = original_seq
                if pos_1idx < 1 or pos_1idx > len(seq):
                    continue
                if seq[pos_1idx - 1] != ref_aa:
                    continue

                is_lof = has_lof_phrase(evidence_text)
                is_excluded = has_exclusion_phrase(evidence_text)
                if not is_lof or is_excluded:
                    continue

                mechanism_tag = assign_mechanism_from_text(evidence_text)
                if not mechanism_tag:
                    continue  # ambiguous evidence text — drop

                for alt_aa in alt_seqs:
                    if len(alt_aa) != 1 or alt_aa == ref_aa:
                        continue
                    records_lof.append({
                        "uniprot_accession": acc,
                        "gene": gene,
                        "protein_name": protein_name,
                        "protein_length": protein_length,
                        "position_1idx": pos_1idx,
                        "ref_aa": ref_aa,
                        "alt_aa": alt_aa,
                        "mutation_hgvs_p": f"p.{ref_aa}{pos_1idx}{alt_aa}",
                        "sequence_context_21aa": sequence_context(seq, pos_1idx),
                        "evidence_text": evidence_text,
                        "mechanism_tag": mechanism_tag,
                    })

    print(f"\nEntries processed: {n_entries}")
    print(f"LoF records with clear text mechanism: {len(records_lof)}")

    df = pd.DataFrame(records_lof)

    intermediate_path = os.path.join(OUT_RAW_DIR, "parsed_mutagenesis_lof_evidence.tsv")
    df.to_csv(intermediate_path, sep="\t", index=False)
    print(f"Saved intermediate to {intermediate_path}")

    # ── Apply Tier 1 filters ───────────────────────────────────────────────────

    df = df[df["protein_length"] <= MAX_PROTEIN_LENGTH]
    print(f"After protein length filter (<= {MAX_PROTEIN_LENGTH}): {len(df)}")

    df = df[df["ref_aa"].isin(STANDARD_AA) & df["alt_aa"].isin(STANDARD_AA)]
    print(f"After standard-AA filter: {len(df)}")

    df = df.drop_duplicates(subset=["uniprot_accession", "position_1idx", "alt_aa"])
    print(f"After deduplication: {len(df)}")

    df = df.copy()
    df["label"] = "LoF_like"
    df["label_confidence"] = df["mechanism_tag"].apply(
        lambda m: "high" if m in {
            "catalytic_LoF", "binding_LoF", "metal_binding_LoF", "structural_stability_LoF"
        } else "medium"
    )
    df["source"] = "UNIPROT_MUTAGENESIS_LOF_EVIDENCE_TEXT"
    df["include_in_poc"] = True
    df["notes"] = ""

    df["variant_id"] = (
        df["uniprot_accession"] + "_"
        + df["ref_aa"] + df["position_1idx"].astype(str) + df["alt_aa"]
    )

    cols = [
        "variant_id", "source", "label", "label_confidence", "mechanism_tag",
        "uniprot_accession", "gene", "protein_name", "protein_length",
        "position_1idx", "ref_aa", "alt_aa", "mutation_hgvs_p",
        "sequence_context_21aa", "evidence_text", "include_in_poc", "notes",
    ]
    df = df[cols]

    out_path = os.path.join(OUT_VAR_DIR, "positive_tier1.tsv")
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\nSaved {len(df)} Tier 1 positive variants to {out_path}")

    print("\n=== Tier 1 summary ===")
    print(f"Unique proteins: {df['uniprot_accession'].nunique()}")
    print(f"Unique genes:    {df['gene'].nunique()}")
    print("\nMechanism tag breakdown:")
    print(df["mechanism_tag"].value_counts().to_string())
    print("\nLabel confidence breakdown:")
    print(df["label_confidence"].value_counts().to_string())

    return df


if __name__ == "__main__":
    df = parse_entries()
