#!/bin/bash
#SBATCH --job-name=nemo_mtp_l40s
#SBATCH --account=torch_pr_111_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yf3005/logs/nemo_mtp/acceptance_%j.log
# Same as run_mtp_acceptance.sbatch on 2x L40S (48 GB each): the 67.5 GB checkpoint is split with device_map=auto.
set -uo pipefail
E=/scratch/yf3005/envs/hf-transcribe
HERE=/scratch/yf3005/sglang-omni/benchmarks/profiling/nemotron_transcribe_mtp
RUN_DIR=/scratch/yf3005/sglang-omni/.profiling-runs/nemotron_transcribe_mtp/acceptance_${SLURM_JOB_ID:-local}
LIBRISPEECH_PARQUET=/scratch/yf3005/data/librispeech_hf/clean/test/0000.parquet
mkdir -p "$RUN_DIR"
export HF_HOME=/scratch/yf3005/.huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
$E/bin/python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"

echo "== seedtts50"
$E/bin/python "$HERE/mtp_acceptance.py" --data seedtts50 --k-max 8 --out "$RUN_DIR/seedtts50.json" || exit 1
echo "== librispeech test-clean (${LIBRI_LIMIT:-300})"
$E/bin/python "$HERE/mtp_acceptance.py" --data librispeech --librispeech-parquet "$LIBRISPEECH_PARQUET" --limit "${LIBRI_LIMIT:-300}" --k-max 8 --out "$RUN_DIR/librispeech.json"
echo "done $(date)"
