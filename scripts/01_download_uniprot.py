"""
Download reviewed human UniProt/Swiss-Prot entries that have Mutagenesis annotations.
Uses the UniProt stream endpoint to download all matching entries in JSON format at once.

Output:
  data/raw/uniprot/entries_full.jsonl.gz  -- one JSON entry per line
  data/raw/uniprot/accessions.txt         -- list of accessions
"""

import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(BASE_DIR, "data", "raw", "uniprot")
os.makedirs(OUT_DIR, exist_ok=True)

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

QUERY = (
    "reviewed:true "
    "AND organism_id:9606 "
    "AND ft_mutagen:*"
)


def stream_download():
    out_path = os.path.join(OUT_DIR, "entries_full.jsonl.gz")
    acc_path = os.path.join(OUT_DIR, "accessions.txt")

    if os.path.exists(out_path) and os.path.exists(acc_path):
        with open(acc_path) as f:
            accs = [l.strip() for l in f if l.strip()]
        print(f"Already downloaded: {len(accs)} entries at {out_path}")
        return

    params = {
        "query": QUERY,
        "format": "json",
        "compressed": "false",
    }
    url = f"{UNIPROT_STREAM}?" + urllib.parse.urlencode(params)
    print(f"Streaming from: {url}")
    print("This may take a few minutes for ~5600 entries...", flush=True)

    n_written = 0
    accessions = []

    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = resp.read()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Downloaded {len(raw):,} bytes. Parsing JSON...", flush=True)
    data = json.loads(raw.decode("utf-8"))
    entries = data.get("results", [])
    print(f"Parsed {len(entries)} entries.", flush=True)

    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for entry in entries:
            acc = entry.get("primaryAccession", "")
            accessions.append(acc)
            fout.write(json.dumps(entry) + "\n")
            n_written += 1

    with open(acc_path, "w") as f:
        f.write("\n".join(accessions) + "\n")

    print(f"Saved {n_written} entries to {out_path}")
    print(f"Saved {len(accessions)} accessions to {acc_path}")


if __name__ == "__main__":
    stream_download()
