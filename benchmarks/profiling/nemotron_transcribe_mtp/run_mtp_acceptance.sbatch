#!/bin/bash
#SBATCH --job-name=nemo_mtp_acc
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yf3005/logs/nemo_mtp/acceptance_%j.log
# Native MTP acceptance of Nemotron-3.5-Transcribe-30B-A4B in PyTorch/Transformers (see mtp_acceptance.py):
# a quick seed-tts-eval-50 pass first, then LibriSpeech test-clean. Env: torch 2.9.1+cu128, transformers 5.23.0.dev0.
set -uo pipefail
E=/scratch/yf3005/envs/hf-transcribe
HERE=/scratch/yf3005/sglang-omni/benchmarks/profiling/nemotron_transcribe_mtp
RUN_DIR=/scratch/yf3005/sglang-omni/.profiling-runs/nemotron_transcribe_mtp/acceptance_${SLURM_JOB_ID:-local}
LIBRISPEECH_PARQUET=/scratch/yf3005/data/librispeech_hf/clean/test/0000.parquet
mkdir -p "$RUN_DIR"
export HF_HOME=/scratch/yf3005/.huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
$E/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"

echo "== seedtts50"
$E/bin/python "$HERE/mtp_acceptance.py" --data seedtts50 --k-max 8 --out "$RUN_DIR/seedtts50.json" || exit 1
echo "== librispeech test-clean (${LIBRI_LIMIT:-500})"
$E/bin/python "$HERE/mtp_acceptance.py" --data librispeech --librispeech-parquet "$LIBRISPEECH_PARQUET" --limit "${LIBRI_LIMIT:-500}" --k-max 8 --out "$RUN_DIR/librispeech.json"
echo "done $(date)"
