#!/bin/bash
#SBATCH --job-name=omni_ar_ratio
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/yf3005/logs/omni_profile/ar_ratio_%j.log
set -uo pipefail

REPO=/scratch/yf3005/sglang-omni
PROFILE_DIR=$REPO/benchmarks/profiling/qwen3_omni_ar_ratio
RUN_DIR=$REPO/.profiling-runs/qwen3_omni_ar_ratio/run_${SLURM_JOB_ID}
PORT=8008
BASE_URL=http://127.0.0.1:$PORT
mkdir -p "$RUN_DIR"
cd "$REPO"
source .venv/bin/activate
export HF_HOME=/scratch/yf3005/.huggingface
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for i in $(seq 1 240); do
  grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/venv_install.log 2>/dev/null \
    && grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/hf_download2.log 2>/dev/null && break
  [ $i -eq 1 ] && echo "waiting for venv install and checkpoint download to finish"
  sleep 15
done
grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/venv_install.log || { echo "venv install not finished"; exit 1; }
grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/hf_download2.log || { echo "checkpoint download not finished"; exit 1; }
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$GPU_NAME" in
  *H200*) CONFIG=examples/configs/qwen3_omni_colocated_h200.yaml ;;
  *H100*) CONFIG=examples/configs/qwen3_omni_colocated_h100_bf16.yaml ;;
  *) echo "no colocated config for $GPU_NAME"; exit 1 ;;
esac
echo "gpu=$GPU_NAME config=$CONFIG"
python -c "import torch, sglang, sglang_omni; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'sglang', sglang.__version__, 'omni', sglang_omni.__version__)"
python -c "import torch; print('cuda ok', torch.cuda.is_available(), torch.cuda.get_device_name(0))" || exit 1

setsid sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config $CONFIG \
  --colocate \
  --host 127.0.0.1 \
  --port $PORT > "$RUN_DIR/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  echo "stopping server pgid=$SERVER_PID"
  kill -TERM -- -"$SERVER_PID" 2>/dev/null
  sleep 10
  kill -KILL -- -"$SERVER_PID" 2>/dev/null
}
trap cleanup EXIT

for i in $(seq 1 180); do
  if curl -sf "$BASE_URL/health" | grep -q healthy; then
    echo "server healthy after $((i * 15))s"
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited early"; tail -50 "$RUN_DIR/server.log"; exit 1
  fi
  sleep 15
done
curl -sf "$BASE_URL/health" | grep -q healthy || { echo "server never became healthy"; tail -50 "$RUN_DIR/server.log"; exit 1; }

echo "== warmup"
python "$PROFILE_DIR/ar_ratio_workload.py" --base-url $BASE_URL --run-id warmup --event-dir "$RUN_DIR/warmup" --mode text --num-requests 3 --concurrency 1 --max-tokens 128 --no-profile
python "$PROFILE_DIR/ar_ratio_workload.py" --base-url $BASE_URL --run-id warmup_audio --event-dir "$RUN_DIR/warmup" --mode audio --num-requests 2 --concurrency 1 --max-tokens 128 --no-profile

run_case() {
  local run_id=$1 mode=$2 conc=$3 n=$4 max_tokens=$5
  echo "== case $run_id mode=$mode c=$conc n=$n max_tokens=$max_tokens"
  python "$PROFILE_DIR/ar_ratio_workload.py" --base-url $BASE_URL --run-id "$run_id" \
    --event-dir "$RUN_DIR/$run_id/events" --mode "$mode" --num-requests "$n" --concurrency "$conc" --max-tokens "$max_tokens"
  sleep 5
  python "$PROFILE_DIR/ar_ratio_report.py" "$RUN_DIR/$run_id/events" --out "$RUN_DIR/$run_id/ar_ratio.json" | tee "$RUN_DIR/$run_id/ar_ratio.txt"
  python -m sglang_omni.profiler "$RUN_DIR/$run_id/events" --format table > "$RUN_DIR/$run_id/stage_breakdown.txt" 2>&1
}

run_case text_c1 text 1 8 128
run_case audio_c1 audio 1 8 128
run_case text_c1_long text 1 4 512
run_case text_c4 text 4 16 128
run_case text_c8 text 8 16 128

echo "done $(date)"
