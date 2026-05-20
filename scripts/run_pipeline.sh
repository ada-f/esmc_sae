#!/bin/bash
#SBATCH --job-name=sae_activations_v2
#SBATCH --account=kempner_grads -p gpu_test
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

set -euo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPTS}/config.env"
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: ${CONFIG} not found."
    echo "Copy scripts/config.env.template to scripts/config.env and fill in your paths."
    exit 1
fi
source "$CONFIG"

PYTHON="${SAE_PYTHON}"
POSITIVE="${SAE_BASE_DATA}/data/variants_with_evidence_text/positive_tier1.tsv"

START=${1:-"03"}
if [ "${1:-}" = "--start" ]; then
    START="${2:-03}"
fi

echo "======================================================"
echo "pipeline — starting from step ${START}"
echo "$(date)"
echo "======================================================"

# ── guard: positive set must exist ────────────────────────────────────────────
if [ ! -f "$POSITIVE" ]; then
    echo "ERROR: positive_tier1.tsv not found at $POSITIVE"
    echo "Run 02_parse_uniprot_mutagenesis.py first."
    exit 1
fi

run_step() {
    local step="$1"
    local script="$2"
    if [[ "$step" < "$START" ]]; then
        echo "Skipping step $step (before --start ${START})"
        return
    fi
    echo ""
    echo "------------------------------------------------------"
    echo "Step $step: $(basename $script)"
    echo "$(date)"
    echo "------------------------------------------------------"
    $PYTHON "$script"
}

# ── CPU steps before GPU ───────────────────────────────────────────────────────
run_step "03" "$SCRIPTS/03_download_clinvar.py"
run_step "04" "$SCRIPTS/04_download_alphamissense.py"
run_step "05" "$SCRIPTS/05_build_controls.py"
run_step "06" "$SCRIPTS/06_dataset_qc.py"

if [[ "$START" < "07" || "$START" = "03" || "$START" = "04" || "$START" = "05" || "$START" = "06" ]]; then
    echo ""
    echo "======================================================"
    echo "Steps 03–06 complete."
    echo ""
    echo "Now submit the GPU job for step 07:"
    echo "  sbatch --export=ALL ${SCRIPTS}/run_sae_activations.sh"
    echo ""
    echo "After step 07 finishes, run:"
    echo "  bash ${SCRIPTS}/run_pipeline.sh --start 08"
    echo "======================================================"
    exit 0
fi

# ── CPU steps after GPU ────────────────────────────────────────────────────────
run_step "08" "$SCRIPTS/08_score_variant_disruptions.py"
run_step "09" "$SCRIPTS/09_make_figures.py"

echo ""
echo "======================================================"
echo "Pipeline complete. $(date)"
echo "Results: $(dirname $POSITIVE | sed 's/data.*//')/results_evidence_text/"
echo "======================================================"
