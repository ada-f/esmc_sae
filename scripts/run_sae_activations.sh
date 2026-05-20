#!/bin/bash
#SBATCH --job-name=sae_activations_v2
#SBATCH --account=kempner_grads -p kempner_h100,kempner
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

# Evidence-text mechanism assignment pipeline — GPU step.
# Reads variants from data/variants_with_evidence_text/poc_lof_variants.tsv.
# Writes results to results_evidence_text/.
# Ref activations are cached in results/activations/ref_cache/ and reused across runs.
#
# Fully resumable: resubmit the same script after a credit-limit stop or preemption.
#   - Completed variants are tracked in results_evidence_text/activations/progress.tsv.
#   - variant_feature_deltas.tsv.gz and variant_scores_raw.tsv are appended per-variant.
#
# Before submitting: confirm no interactive run is already active:
#   pgrep -a python | grep 07_compute_sae_activations

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPTS_DIR}/config.env"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: ${CONFIG} not found."
    echo "Copy scripts/config.env.template to scripts/config.env and fill in your paths."
    exit 1
fi
source "$CONFIG"

PYTHON="${SAE_PYTHON}"
SCRIPT="${SCRIPTS_DIR}/07_compute_sae_activations.py"
SCORES="${SAE_BASE_DATA}/results_evidence_text/tables/variant_scores_raw.tsv"

echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURMD_NODENAME"
echo "Start:  $(date)"
echo "API key set: $([ -n "${ESM_API_KEY:-}" ] && echo YES || echo NO)"

if [ -z "${ESM_API_KEY:-}" ]; then
    echo "ERROR: ESM_API_KEY is not set. Submit with: sbatch --export=ALL run_sae_activations.sh"
    exit 1
fi

if [ -f "$SCORES" ]; then
    echo "Deduplicating scores file ..."
    $PYTHON - "$SCORES" <<'PYEOF'
import sys, pandas as pd
path = sys.argv[1]
df = pd.read_csv(path, sep="\t")
before = len(df)
df = df.drop_duplicates(subset="variant_id", keep="last")
df.to_csv(path, sep="\t", index=False)
print(f"  {before} rows -> {len(df)} rows ({before - len(df)} duplicates removed)")
PYEOF
fi

$PYTHON "$SCRIPT" --model 6b

echo "Finished: $(date)"
