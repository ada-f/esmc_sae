"""
Compute reference and mutant SAE activations for all variants in poc_lof_variants.tsv.
scripts_v2 version: uses evidence-text variant set; reuses shared ref_cache from results/.

Strategy:
  - For each unique protein, run the canonical (ref) sequence through ESM + SAE once
    and cache the sparse activation tensor to disk.
  - CACHE REUSE: ref activations are read from (and written to) the same shared cache
    as the original pipeline at results/activations/ref_cache/. Any protein already
    computed there will be reused — only new proteins require API calls.
  - For each variant, apply the single-AA substitution, run the mutant sequence, and
    compute per-feature delta statistics over four windows (±1, ±4, ±8, ±16 residues).
  - Handles rate limiting: exponential backoff on 429s, graceful stop with progress saved.
  - Fully resumable: skips already-completed variants on restart.

Outputs (all under BASE_DATA/results_evidence_text/):
  activations/progress.tsv             -- run log: variant_id, status, timestamp
  tables/variant_feature_deltas.tsv.gz -- per-(variant, feature) delta table
  tables/variant_scores_raw.tsv        -- aggregate per-variant stats

Shared (read-only if already populated):
  results/activations/ref_cache/{acc}.npz  -- sparse ref activations (seq_len × 16384)

Usage:
  python 08_compute_sae_activations.py [--model 600m|6b] [--max-variants N] [--dry-run]
"""

import argparse
import gzip
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── paths ──────────────────────────────────────────────────────────────────────
from config import BASE_DATA, require
ESM_DIR = require("SAE_ESM_DIR")
VARIANT_TABLE = os.path.join(BASE_DATA, "data", "variants_with_evidence_text", "poc_lof_variants.tsv")
UNIPROT_JSONL = os.path.join(BASE_DATA, "data", "raw", "uniprot", "entries_full.jsonl.gz")
RESULTS_DIR = os.path.join(BASE_DATA, "results_evidence_text")

# Shared ref cache — reuse activations already computed by the original pipeline
REF_CACHE_DIR = os.path.join(BASE_DATA, "results", "activations", "ref_cache")
PROGRESS_PATH = os.path.join(RESULTS_DIR, "activations", "progress.tsv")
DELTAS_PATH = os.path.join(RESULTS_DIR, "tables", "variant_feature_deltas.tsv.gz")
SCORES_PATH = os.path.join(RESULTS_DIR, "tables", "variant_scores_raw.tsv")

# Create output dirs (do NOT recreate REF_CACHE_DIR — it is shared)
for d in [os.path.join(RESULTS_DIR, "activations"),
          os.path.join(RESULTS_DIR, "tables"),
          REF_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, ESM_DIR)

# ── model configs ──────────────────────────────────────────────────────────────
MODELS = {
    "600m": {
        "esm": "esmc-600m-2024-12",
        "sae": "esmc-600m-2024-12_k64_codebook16384_layer27",
    },
    "6b": {
        "esm": "esmc-6b-2024-12",
        "sae": "esmc-6b-2024-12_k64_codebook16384_layer60",
    },
}

WINDOWS = [1, 4, 8, 16]
PRIMARY_WINDOW = 8

MAX_RETRIES = 6
BACKOFF_BASE = 15  # seconds


# ── sparse tensor utilities ────────────────────────────────────────────────────

def sparse_to_npz(tensor: torch.Tensor, path: str):
    t = tensor.coalesce()
    idx = t.indices().numpy()
    vals = t.values().numpy().astype(np.float32)
    shape = np.array(t.shape)
    np.savez_compressed(path, positions=idx[0], features=idx[1],
                        values=vals, shape=shape)


def npz_to_sparse(path: str) -> torch.Tensor:
    d = np.load(path)
    idx = torch.tensor(np.stack([d["positions"], d["features"]]), dtype=torch.long)
    vals = torch.tensor(d["values"], dtype=torch.float64)
    shape = tuple(d["shape"].tolist())
    return torch.sparse_coo_tensor(idx, vals, size=shape).coalesce()


def window_max_per_feature(sparse_tensor: torch.Tensor,
                            win_start: int, win_end: int) -> dict:
    t = sparse_tensor.coalesce()
    idx = t.indices()
    vals = t.values()
    mask = (idx[0] >= win_start) & (idx[0] <= win_end)
    f_idx = idx[1, mask].tolist()
    f_vals = vals[mask].tolist()
    feat_max: dict = {}
    for f, v in zip(f_idx, f_vals):
        if v > feat_max.get(f, float("-inf")):
            feat_max[f] = v
    return feat_max


def site_values(sparse_tensor: torch.Tensor, pos: int) -> dict:
    t = sparse_tensor.coalesce()
    idx = t.indices()
    vals = t.values()
    mask = idx[0] == pos
    return dict(zip(idx[1, mask].tolist(), vals[mask].tolist()))


def compute_delta_records(ref_t: torch.Tensor, mut_t: torch.Tensor,
                           pos_0idx: int) -> tuple[list, dict]:
    seq_len = ref_t.shape[0]

    ref_site = site_values(ref_t, pos_0idx)
    mut_site = site_values(mut_t, pos_0idx)

    win_data: dict = {}
    for w in WINDOWS:
        ws = max(0, pos_0idx - w)
        we = min(seq_len - 1, pos_0idx + w)
        win_data[w] = {
            "ref_wmax": window_max_per_feature(ref_t, ws, we),
            "mut_wmax": window_max_per_feature(mut_t, ws, we),
        }

    pw = PRIMARY_WINDOW
    ref_wmax_p = win_data[pw]["ref_wmax"]
    mut_wmax_p = win_data[pw]["mut_wmax"]
    all_features = set(ref_wmax_p) | set(mut_wmax_p)

    records = []
    for f in all_features:
        rs = ref_site.get(f, 0.0)
        ms = mut_site.get(f, 0.0)
        rwm = ref_wmax_p.get(f, 0.0)
        mwm = mut_wmax_p.get(f, 0.0)
        records.append({
            "feature_id": f,
            "ref_site": rs,
            "mut_site": ms,
            "delta_site": ms - rs,
            "ref_window_max": rwm,
            "mut_window_max": mwm,
            "delta_window_max": mwm - rwm,
            "feature_loss": max(0.0, rwm - mwm),
            "feature_gain": max(0.0, mwm - rwm),
        })

    losses = [r["feature_loss"] for r in records]
    gains = [r["feature_gain"] for r in records]
    abs_deltas = [abs(r["delta_window_max"]) for r in records]
    losses_sorted = sorted(losses, reverse=True)
    abs_sorted = sorted(abs_deltas, reverse=True)

    win_abs_delta = {}
    for w in WINDOWS:
        rwm_w = win_data[w]["ref_wmax"]
        mwm_w = win_data[w]["mut_wmax"]
        all_f_w = set(rwm_w) | set(mwm_w)
        win_abs_delta[f"total_abs_delta_w{w}"] = sum(
            abs(mwm_w.get(f, 0.0) - rwm_w.get(f, 0.0)) for f in all_f_w
        )

    agg = {
        "n_active_features_ref": len(ref_wmax_p),
        "n_active_features_mut": len(mut_wmax_p),
        "n_active_features_union": len(all_features),
        "max_feature_loss": losses_sorted[0] if losses_sorted else 0.0,
        "mean_top5_feature_loss": float(np.mean(losses_sorted[:5])) if losses_sorted else 0.0,
        "sum_feature_loss": sum(losses),
        "max_feature_gain": max(gains) if gains else 0.0,
        "sum_feature_gain": sum(gains),
        "max_abs_delta": abs_sorted[0] if abs_sorted else 0.0,
        "mean_top10_abs_delta": float(np.mean(abs_sorted[:10])) if abs_sorted else 0.0,
        "total_abs_delta": sum(abs_deltas),
        **win_abs_delta,
    }
    return records, agg


# ── API helpers ────────────────────────────────────────────────────────────────

def run_sae(client, sae_config, sequence: str, logits_config):
    from esm.sdk.api import ESMProtein, ESMProteinError
    from cookbook.snippets.sparse_utils import remove_indexes

    for attempt in range(MAX_RETRIES):
        try:
            protein = ESMProtein(sequence=sequence)
            protein_tensor = client.encode(protein)
            if isinstance(protein_tensor, ESMProteinError):
                raise ValueError(f"Encode error: {protein_tensor.error_msg}")

            output = client.logits(
                protein_tensor,
                config=logits_config,
                return_bytes=False,
            )
            if isinstance(output, ESMProteinError):
                raise ValueError(f"Logits error: {output.error_msg}")
            if output.sae_outputs is None:
                raise ValueError("SAE outputs missing")

            t = output.sae_outputs[sae_config.model]
            t = remove_indexes(t, {0, -1})
            return t.coalesce()

        except Exception as e:
            err = str(e)
            is_rate_limit = any(x in err.lower() for x in
                                ["429", "rate limit", "too many requests",
                                 "quota", "throttle", "credit", "exceeded"])
            if is_rate_limit:
                raise
            elif attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                raise


# ── progress log ───────────────────────────────────────────────────────────────

def load_progress() -> set:
    if not os.path.exists(PROGRESS_PATH):
        return set()
    df = pd.read_csv(PROGRESS_PATH, sep="\t")
    return set(df.loc[df["status"] == "done", "variant_id"].tolist())


def log_progress(variant_id: str, status: str, note: str = ""):
    line = f"{variant_id}\t{status}\t{datetime.now().isoformat()}\t{note}\n"
    with open(PROGRESS_PATH, "a") as f:
        f.write(line)


def init_progress_file():
    if not os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "w") as f:
            f.write("variant_id\tstatus\ttimestamp\tnote\n")


# ── output helpers ─────────────────────────────────────────────────────────────

_scores_initialized = False

def append_scores(row: dict):
    global _scores_initialized
    df = pd.DataFrame([row])
    if not _scores_initialized and not os.path.exists(SCORES_PATH):
        df.to_csv(SCORES_PATH, sep="\t", index=False)
    else:
        df.to_csv(SCORES_PATH, sep="\t", index=False,
                  mode="a", header=not os.path.exists(SCORES_PATH))
    _scores_initialized = True


def write_deltas(variant_id: str, records: list):
    if not records:
        return
    for r in records:
        r["variant_id"] = variant_id
    df = pd.DataFrame(records)
    col_order = ["variant_id", "feature_id",
                 "ref_site", "mut_site", "delta_site",
                 "ref_window_max", "mut_window_max", "delta_window_max",
                 "feature_loss", "feature_gain"]
    df = df[col_order]
    write_header = not os.path.exists(DELTAS_PATH)
    df.to_csv(DELTAS_PATH, sep="\t", index=False,
              mode="a", header=write_header, compression="gzip")


# ── main ───────────────────────────────────────────────────────────────────────

def load_sequences() -> dict:
    print("Loading canonical sequences from UniProt ...", flush=True)
    seqs = {}
    with gzip.open(UNIPROT_JSONL, "rt") as f:
        for line in f:
            entry = json.loads(line)
            acc = entry.get("primaryAccession", "")
            seq = entry.get("sequence", {}).get("value", "")
            seqs[acc] = seq
    print(f"  Loaded {len(seqs)} sequences")
    return seqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["600m", "6b"], default="6b")
    parser.add_argument("--max-variants", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = MODELS[args.model]
    print(f"Model: {cfg['esm']}  SAE: {cfg['sae']}")
    print(f"Ref cache (shared): {REF_CACHE_DIR}")
    print(f"Results dir: {RESULTS_DIR}")

    from esm.sdk.api import SAEConfig, LogitsConfig
    from esm.sdk.forge import ESMCForgeInferenceClient

    client = ESMCForgeInferenceClient(
        model=cfg["esm"],
        url="https://forge.evolutionaryscale.ai",
        token=os.environ["ESM_API_KEY"],
    )
    sae_config = SAEConfig(model=cfg["sae"], normalize_features=True)
    logits_config = LogitsConfig(sae_config=sae_config)

    variants = pd.read_csv(VARIANT_TABLE, sep="\t")
    seqs = load_sequences()
    init_progress_file()
    completed = load_progress()
    print(f"Variants total: {len(variants)} | Already done: {len(completed)}")

    n_cached = sum(
        1 for acc in variants["uniprot_accession"].unique()
        if os.path.exists(os.path.join(REF_CACHE_DIR, f"{acc}.npz"))
    )
    print(f"Ref cache hits: {n_cached}/{variants['uniprot_accession'].nunique()} proteins")

    label_order = {"LoF_like": 0, "benign_control": 1, "nearby_control": 2}
    variants = variants.copy()
    variants["_order"] = variants["label"].map(label_order).fillna(3)
    variants = variants.sort_values(["_order", "protein_length"]).drop(columns=["_order"])
    variants = variants[~variants["variant_id"].isin(completed)].reset_index(drop=True)

    if args.max_variants is not None:
        variants = variants.head(args.max_variants)
        print(f"Limiting to {args.max_variants} variants")

    print(f"To process: {len(variants)} variants\n")

    if args.dry_run:
        print("DRY RUN — no API calls made.")
        print(variants[["variant_id", "label", "gene", "protein_length"]].head(20).to_string())
        return

    proteins_order = list(dict.fromkeys(variants["uniprot_accession"].tolist()))

    n_done = 0
    n_errors = 0
    rate_limit_hit = False
    t_start = time.time()

    for acc in proteins_order:
        prot_variants = variants[variants["uniprot_accession"] == acc]
        ref_seq = seqs.get(acc, "")
        if not ref_seq:
            for _, row in prot_variants.iterrows():
                log_progress(row["variant_id"], "skip", "no sequence")
            continue

        ref_cache_path = os.path.join(REF_CACHE_DIR, f"{acc}.npz")
        if os.path.exists(ref_cache_path):
            ref_t = npz_to_sparse(ref_cache_path)
        else:
            print(f"  [{acc}] Running ref ({len(ref_seq)} AA) ...", flush=True)
            try:
                ref_t = run_sae(client, sae_config, ref_seq, logits_config)
                sparse_to_npz(ref_t, ref_cache_path)
                print(f"    -> shape {tuple(ref_t.shape)}, nnz={ref_t._nnz()}")
            except Exception as e:
                err = str(e)
                is_rl = any(x in err.lower() for x in
                            ["429", "rate limit", "too many requests", "quota", "credit", "exceeded"])
                print(f"    ERROR on ref {acc}: {err[:120]}", flush=True)
                if is_rl:
                    rate_limit_hit = True
                    break
                for _, row in prot_variants.iterrows():
                    log_progress(row["variant_id"], "error", f"ref_failed: {err[:80]}")
                    n_errors += 1
                continue

        if ref_t.shape[0] != len(ref_seq):
            for _, row in prot_variants.iterrows():
                log_progress(row["variant_id"], "skip",
                             f"len mismatch: tensor={ref_t.shape[0]} seq={len(ref_seq)}")
            continue

        for _, row in prot_variants.iterrows():
            vid = row["variant_id"]
            pos_1idx = int(row["position_1idx"])
            ref_aa = row["ref_aa"]
            alt_aa = row["alt_aa"]
            pos_0idx = pos_1idx - 1

            if pos_0idx < 0 or pos_0idx >= len(ref_seq) or ref_seq[pos_0idx] != ref_aa:
                log_progress(vid, "skip", "position mismatch")
                continue

            mut_seq = ref_seq[:pos_0idx] + alt_aa + ref_seq[pos_0idx + 1:]

            print(f"  [{acc}] {vid} ({row['label']}) pos={pos_1idx} "
                  f"{ref_aa}->{alt_aa} ...", flush=True, end=" ")

            try:
                t0 = time.time()
                mut_t = run_sae(client, sae_config, mut_seq, logits_config)
                elapsed = time.time() - t0
                print(f"{elapsed:.1f}s", flush=True)
            except Exception as e:
                err = str(e)
                is_rl = any(x in err.lower() for x in
                            ["429", "rate limit", "too many requests", "quota", "credit", "exceeded"])
                print(f"ERROR: {err[:80]}", flush=True)
                log_progress(vid, "error", err[:120])
                n_errors += 1
                if is_rl:
                    rate_limit_hit = True
                    break
                continue

            try:
                records, agg = compute_delta_records(ref_t, mut_t, pos_0idx)
            except Exception as e:
                log_progress(vid, "error", f"delta_failed: {str(e)[:80]}")
                n_errors += 1
                continue

            score_row = {
                "variant_id": vid,
                "label": row["label"],
                "label_confidence": row["label_confidence"],
                "mechanism_tag": row["mechanism_tag"],
                "uniprot_accession": acc,
                "gene": row["gene"],
                "protein_length": row["protein_length"],
                "position_1idx": pos_1idx,
                "ref_aa": ref_aa,
                "alt_aa": alt_aa,
                "model": cfg["esm"],
                "sae_model": cfg["sae"],
                **agg,
            }
            append_scores(score_row)
            write_deltas(vid, records)

            log_progress(vid, "done",
                         f"n_active={agg['n_active_features_union']}"
                         f" max_loss={agg['max_feature_loss']:.4f}")
            n_done += 1

        if rate_limit_hit:
            break

    elapsed_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Run complete.")
    print(f"  Variants processed: {n_done}")
    print(f"  Errors: {n_errors}")
    print(f"  Rate limit hit: {rate_limit_hit}")
    print(f"  Elapsed: {elapsed_total/60:.1f} min")
    if n_done > 0:
        print(f"  Avg time/variant: {elapsed_total/n_done:.1f}s")
    if os.path.exists(SCORES_PATH):
        s = pd.read_csv(SCORES_PATH, sep="\t")
        print(f"\nScores table: {len(s)} rows")
        print(s["label"].value_counts().to_string())
        print(f"\nMax feature loss (LoF_like): "
              f"{s[s.label=='LoF_like']['max_feature_loss'].median():.4f} median")
        print(f"Max feature loss (benign_control): "
              f"{s[s.label=='benign_control']['max_feature_loss'].median():.4f} median")
    if os.path.exists(DELTAS_PATH):
        d = pd.read_csv(DELTAS_PATH, sep="\t", compression="gzip")
        print(f"\nDeltas table: {len(d):,} rows ({d['variant_id'].nunique()} variants)")


if __name__ == "__main__":
    main()
