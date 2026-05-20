"""
Download AlphaMissense protein-level pathogenicity predictions and extract
likely_benign variants for the proteins in our positive set.

Source: AlphaMissense (Cheng et al., Science 2023)
Zenodo: https://zenodo.org/records/8208688
File:   AlphaMissense_aa_substitutions.tsv.gz

Columns in source file:
  #uniprot_id  protein_variant  am_pathogenicity  am_class

am_class values: likely_benign | ambiguous | likely_pathogenic

Output:
  data/raw/alphamissense/am_likely_benign_positive_proteins.tsv.gz
    -- likely_benign variants restricted to the 720 positive-set proteins
"""

import gzip
import os
import re
import urllib.request

from config import BASE_DATA
OUT_DIR = os.path.join(BASE_DATA, "data", "raw", "alphamissense")
os.makedirs(OUT_DIR, exist_ok=True)

POSITIVE_PATH = os.path.join(BASE_DATA, "data", "variants", "positive_tier1.tsv")
AM_URL = "https://zenodo.org/records/8208688/files/AlphaMissense_aa_substitutions.tsv.gz"
AM_RAW = os.path.join(OUT_DIR, "AlphaMissense_aa_substitutions.tsv.gz")
AM_FILTERED = os.path.join(OUT_DIR, "am_likely_benign_positive_proteins.tsv.gz")

VARIANT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def load_positive_accessions():
    import pandas as pd
    df = pd.read_csv(POSITIVE_PATH, sep="\t")
    accs = set(df["uniprot_accession"].unique())
    print(f"Positive-set proteins: {len(accs)}")
    return accs


def download_am():
    if os.path.exists(AM_RAW):
        print(f"Already downloaded: {AM_RAW}")
        return
    print(f"Downloading AlphaMissense from {AM_URL} ...")
    print("(~1.2 GB, this will take a few minutes)", flush=True)
    urllib.request.urlretrieve(AM_URL, AM_RAW)
    print(f"Saved to {AM_RAW}")


def filter_am(positive_accs):
    if os.path.exists(AM_FILTERED):
        print(f"Already filtered: {AM_FILTERED}")
        return

    print("Filtering AlphaMissense to likely_benign in positive-set proteins...", flush=True)
    kept = []
    n_total = 0
    n_benign = 0

    with gzip.open(AM_RAW, "rt", encoding="utf-8") as fin:
        # Skip leading comment lines (start with #), read the real header
        col = {}
        for raw_header in fin:
            if raw_header.startswith("#"):
                continue
            # This is the real header: uniprot_id  protein_variant  am_pathogenicity  am_class
            col = {name: i for i, name in enumerate(raw_header.strip().split("\t"))}
            break
        print(f"  Header columns: {col}", flush=True)

        for line in fin:
            n_total += 1
            if n_total % 5_000_000 == 0:
                print(f"  {n_total/1e6:.0f}M lines, {n_benign:,} benign kept...", flush=True)

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue

            def g(name):
                i = col.get(name)
                return parts[i] if i is not None else ""

            acc = g("uniprot_id")
            if acc not in positive_accs:
                continue

            am_class = g("am_class")
            if am_class != "benign":
                continue

            variant = g("protein_variant")
            m = VARIANT_RE.match(variant)
            if not m:
                continue

            ref_aa = m.group(1)
            pos = int(m.group(2))
            alt_aa = m.group(3)
            am_score = g("am_pathogenicity")

            n_benign += 1
            kept.append(f"{acc}\t{ref_aa}\t{pos}\t{alt_aa}\t{am_score}\n")

    print(f"Scanned {n_total:,} lines, kept {n_benign:,} likely_benign variants in positive-set proteins")

    with gzip.open(AM_FILTERED, "wt", encoding="utf-8") as fout:
        fout.write("uniprot_accession\tref_aa\tposition_1idx\talt_aa\tam_pathogenicity\n")
        for line in kept:
            fout.write(line)

    print(f"Saved to {AM_FILTERED}")


def main():
    positive_accs = load_positive_accessions()
    download_am()
    filter_am(positive_accs)


if __name__ == "__main__":
    main()
