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
export SEEDTTS_SIM_CACHE_DIR=/scratch/yf3005/.huggingface/seedtts_sim
export UTMOS_CACHE_DIR=/scratch/yf3005/.huggingface/utmos
ASR_PORT=8010

echo "node=$(hostname) job=$SLURM_JOB_ID start=$(date) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for i in $(seq 1 240); do
  grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/venv_install.log 2>/dev/null \
    && grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/hf_download2.log 2>/dev/null \
    && grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/seedtts_prepare.log 2>/dev/null \
    && grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/eval_assets.log 2>/dev/null && break
  [ $i -eq 1 ] && echo "waiting for venv install and checkpoint download to finish"
  sleep 15
done
grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/venv_install.log || { echo "venv install not finished"; exit 1; }
grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/hf_download2.log || { echo "checkpoint download not finished"; exit 1; }
grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/seedtts_prepare.log || { echo "seedtts dataset not prepared"; exit 1; }
grep -q "^EXIT=0" /scratch/yf3005/logs/omni_profile/eval_assets.log || { echo "ASR model and WER/SIM/UTMOS assets not prepared"; exit 1; }
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

# CI protocol: the seedtts-50 subset with voice-clone references, 256 new tokens, temperature 0.7, talker defaults.
SEEDTTS_META=zhaochenyang20/seed-tts-eval-50-arrow
SEEDTTS_SAMPLES=50
CASES="seedtts_c1:1 seedtts_c8:8 seedtts_c16:16"

# One official generate-only run wrapped in a request profile; its warmup requests are the first <concurrency> admissions and are skipped in the report.
run_seedtts_case() {
  local run_id=$1 conc=$2
  local case_dir=$RUN_DIR/$run_id
  echo "== case $run_id seedtts c=$conc samples=$SEEDTTS_SAMPLES"
  mkdir -p "$case_dir"
  curl -sf -X POST "$BASE_URL/start_request_profile" -H 'Content-Type: application/json' \
    -d "{\"run_id\":\"$run_id\",\"event_dir\":\"$case_dir/events\",\"event_buffer_capacity\":65536}" || { echo "start_request_profile failed"; return 1; }
  python -m benchmarks.eval.benchmark_omni_seedtts \
    --generate-only --disable-tqdm --fingerprint --voice-clone \
    --meta "$SEEDTTS_META" --max-samples "$SEEDTTS_SAMPLES" \
    --max-concurrency "$conc" --warmup "$conc" \
    --model qwen3-omni --base-url "$BASE_URL" \
    --output-dir "$case_dir/seedtts"
  curl -sf -X POST "$BASE_URL/stop_request_profile" -H 'Content-Type: application/json' -d '{}'
  python - "$case_dir/events" "$PROFILE_DIR" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from ar_ratio_workload import wait_for_sidecars
print("sidecars:", [s.get("complete") for s in wait_for_sidecars(Path(sys.argv[1]), timeout_s=300)])
PY
  python "$PROFILE_DIR/ar_ratio_report.py" "$case_dir/events" --skip-first "$conc" --out "$case_dir/ar_ratio.json" | tee "$case_dir/ar_ratio.txt"
  python -m sglang_omni.profiler "$case_dir/events" --format table > "$case_dir/stage_breakdown.txt" 2>&1
}

run_case() {
  local run_id=$1 mode=$2 conc=$3 n=$4 max_tokens=$5
  echo "== case $run_id mode=$mode c=$conc n=$n max_tokens=$max_tokens"
  python "$PROFILE_DIR/ar_ratio_workload.py" --base-url $BASE_URL --run-id "$run_id" \
    --event-dir "$RUN_DIR/$run_id/events" --mode "$mode" --num-requests "$n" --concurrency "$conc" --max-tokens "$max_tokens"
  sleep 5
  python "$PROFILE_DIR/ar_ratio_report.py" "$RUN_DIR/$run_id/events" --out "$RUN_DIR/$run_id/ar_ratio.json" | tee "$RUN_DIR/$run_id/ar_ratio.txt"
  python -m sglang_omni.profiler "$RUN_DIR/$run_id/events" --format table > "$RUN_DIR/$run_id/stage_breakdown.txt" 2>&1
}

for entry in $CASES; do
  run_seedtts_case "${entry%%:*}" "${entry##*:}"
done
run_case text_c1 text 1 8 128

# WER needs the GPU for Qwen3-ASR, so the omni server goes down first, as in CI.
cleanup
trap - EXIT
sleep 20
nvidia-smi --query-gpu=memory.used --format=csv

echo "== asr server"
setsid sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --host 127.0.0.1 --port $ASR_PORT > "$RUN_DIR/asr_server.log" 2>&1 &
ASR_PID=$!
cleanup_asr() {
  echo "stopping asr server pgid=$ASR_PID"
  kill -TERM -- -"$ASR_PID" 2>/dev/null
  sleep 10
  kill -KILL -- -"$ASR_PID" 2>/dev/null
}
trap cleanup_asr EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$ASR_PORT/health" | grep -q healthy && { echo "asr server healthy after $((i * 15))s"; break; }
  kill -0 "$ASR_PID" 2>/dev/null || { echo "asr server exited early"; tail -30 "$RUN_DIR/asr_server.log"; exit 1; }
  sleep 15
done

for entry in $CASES; do
  run_id=${entry%%:*}
  echo "== wer $run_id"
  python -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --disable-tqdm \
    --meta "$SEEDTTS_META" --output-dir "$RUN_DIR/$run_id/seedtts" --model qwen3-omni --lang en \
    --port $ASR_PORT --asr-model-path Qwen/Qwen3-ASR-1.7B --asr-concurrency 4
done
cleanup_asr
trap - EXIT
sleep 10

for entry in $CASES; do
  run_id=${entry%%:*}
  echo "== similarity $run_id"
  python -m benchmarks.eval.benchmark_omni_seedtts --similarity-only --disable-tqdm \
    --meta "$SEEDTTS_META" --output-dir "$RUN_DIR/$run_id/seedtts" --model qwen3-omni --device cuda:0
  echo "== utmos $run_id"
  python -m benchmarks.eval.benchmark_omni_seedtts --utmos-only --disable-tqdm \
    --meta "$SEEDTTS_META" --output-dir "$RUN_DIR/$run_id/seedtts" --model qwen3-omni --device cuda:0
done

echo "== summary"
python "$PROFILE_DIR/summarize_cases.py" "$RUN_DIR" $CASES | tee "$RUN_DIR/summary.md"
echo "done $(date)"
