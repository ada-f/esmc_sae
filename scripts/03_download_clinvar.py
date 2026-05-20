"""
Download ClinVar variant_summary.txt.gz and filter for:
  - Human GRCh38
  - Benign or Likely benign (germline)
  - Single nucleotide variant, missense (has protein change, not synonymous)
  - No conflicting interpretations

Output:
  data/raw/clinvar/clinvar_benign_missense.tsv.gz
"""

import gzip
import os
import re
import urllib.request

from config import BASE_DATA
OUT_DIR = os.path.join(BASE_DATA, "data", "raw", "clinvar")
os.makedirs(OUT_DIR, exist_ok=True)

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
)

BENIGN_TERMS = {"Benign", "Likely benign", "Benign/Likely benign"}

# 3-letter to 1-letter amino acid map
AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O", "Ter": "*",
}

# Match p.Arg259Pro or p.R259P (single-letter also possible)
HGVS_P_3LET = re.compile(
    r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)"
)
HGVS_P_1LET = re.compile(
    r"\(p\.([A-Z])(\d+)([A-Z])\)"
)


def extract_missense(name):
    """
    Extract (ref_aa_1, position, alt_aa_1) from a ClinVar Name string.
    Returns None if not a missense or not parseable.
    """
    # Try 3-letter first
    m = HGVS_P_3LET.search(name)
    if m:
        ref3 = m.group(1)
        pos = int(m.group(2))
        alt3 = m.group(3)
        ref1 = AA3TO1.get(ref3)
        alt1 = AA3TO1.get(alt3)
        if ref1 and alt1 and ref1 != alt1 and ref1 != "*" and alt1 != "*":
            return ref1, pos, alt1
        return None
    # Try 1-letter
    m = HGVS_P_1LET.search(name)
    if m:
        ref1 = m.group(1)
        pos = int(m.group(2))
        alt1 = m.group(3)
        if ref1 != alt1:
            return ref1, pos, alt1
    return None


def download_clinvar():
    raw_path = os.path.join(OUT_DIR, "variant_summary.txt.gz")
    if not os.path.exists(raw_path):
        print(f"Downloading ClinVar variant_summary from {CLINVAR_URL} ...")
        urllib.request.urlretrieve(CLINVAR_URL, raw_path)
        print(f"Saved to {raw_path}")
    else:
        print(f"Using cached: {raw_path}")
    return raw_path


def parse_clinvar(raw_path):
    out_path = os.path.join(OUT_DIR, "clinvar_benign_missense.tsv.gz")
    if os.path.exists(out_path):
        print(f"Already parsed: {out_path}")
        return

    print("Filtering ClinVar for human benign missense SNVs...")
    kept = []
    n_total = 0

    with gzip.open(raw_path, "rt", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split("\t")
        col = {name: i for i, name in enumerate(header)}

        def g(parts, name):
            i = col.get(name)
            return parts[i] if i is not None and i < len(parts) else ""

        for line in f:
            n_total += 1
            if n_total % 500000 == 0:
                print(f"  {n_total:,} lines, {len(kept):,} kept...", flush=True)
            parts = line.rstrip("\n").split("\t")

            if g(parts, "Assembly") != "GRCh38":
                continue
            if g(parts, "Type") != "single nucleotide variant":
                continue

            clinsig = g(parts, "ClinicalSignificance")
            if clinsig not in BENIGN_TERMS:
                continue

            name = g(parts, "Name")
            result = extract_missense(name)
            if result is None:
                continue
            ref_aa, pos, alt_aa = result

            gene = g(parts, "GeneSymbol").strip()
            if not gene or ";" in gene:
                continue  # intergenic or multi-gene

            kept.append({
                "gene": gene,
                "ref_aa": ref_aa,
                "position_1idx": pos,
                "alt_aa": alt_aa,
                "mutation_hgvs_p": f"p.{ref_aa}{pos}{alt_aa}",
                "clinical_significance": clinsig,
                "review_status": g(parts, "ReviewStatus"),
                "variation_id": g(parts, "VariationID"),
                "rs_id": g(parts, "RS# (dbSNP)"),
                "number_submitters": g(parts, "NumberSubmitters"),
                "name_field": name,
            })

    print(f"Total lines: {n_total:,}, Kept benign missense: {len(kept):,}")

    import pandas as pd
    df = pd.DataFrame(kept)
    df = df.drop_duplicates(subset=["gene", "ref_aa", "position_1idx", "alt_aa"])
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    print(f"Saved {len(df)} unique records to {out_path}")
    print("\nTop genes:")
    print(df["gene"].value_counts().head(20).to_string())


def main():
    raw_path = download_clinvar()
    parse_clinvar(raw_path)


if __name__ == "__main__":
    main()
