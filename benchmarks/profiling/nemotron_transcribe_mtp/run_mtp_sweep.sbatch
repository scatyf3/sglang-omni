#!/bin/bash
#SBATCH --job-name=nemo_mtp_sweep
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/yf3005/logs/nemo_mtp/sweep_%j.log
# Reference sweep on one H200 or H100. An 80 GB card holds the bf16 decoder, the draft and the encoder with about
# 4 GB to spare, so it runs with a higher static fraction, a 16k context and at most 8 running requests:
# baseline and the 7-step MTP draft, pre-LM encoder on, c=1/8/16 with three repeats and request events for the stage breakdown. Serve with NEXTN, transcribe seed-tts-eval-50,
# read spec_accept_length from one response, then repeat without the draft for the WER baseline.
set -uo pipefail
# Defaults for the reference sweep; the environment can narrow them (e.g. MODES=mtp CONCS=1 PROFILE=).
MODES=${MODES-baseline mtp}
MTP_STEPS=${MTP_STEPS-7}
MTP_DRAFT=${MTP_DRAFT-8}
PROFILE=${PROFILE-1}

REPO=/scratch/yf3005/sglang-omni
RUN_DIR=$REPO/.profiling-runs/nemotron_transcribe_mtp/sweep_${SLURM_JOB_ID:-local}
MODEL=nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B
PORT=8000
mkdir -p "$RUN_DIR"
cd "$REPO"
source .venv/bin/activate
export HF_HOME=/scratch/yf3005/.huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export SGLANG_OMNI_STARTUP_TIMEOUT=3600

echo "node=$(hostname) job=${SLURM_JOB_ID:-} start=$(date) commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$GPU_MIB" -lt 100000 ]; then
  SERVE_ARGS="--mem-fraction-static 0.93 --asr.factory.context_length 16384 --asr.factory.max_new_tokens 1024 --asr.engine.max_running_requests 8"
else
  SERVE_ARGS="--mem-fraction-static 0.85 --asr.engine.max_running_requests 16"
fi
echo "serve args: $SERVE_ARGS"
CLIP=$(python - <<'PY'
import glob, io, pyarrow.parquet as pq, soundfile
rows = pq.read_table(glob.glob("/scratch/yf3005/.huggingface/hub/datasets--zhaochenyang20--seed-tts-eval-50-arrow/snapshots/*/data/en-*.parquet")[0]).to_pylist()
wav, sr = soundfile.read(io.BytesIO(rows[1]["ref_audio"]["bytes"]), dtype="float32")
soundfile.write("/tmp/nemo_smoke_clip.wav", wav, sr)
print("/tmp/nemo_smoke_clip.wav")
PY
)

serve_and_bench() {
  local tag=$1; shift
  echo "== $tag: launch"
  setsid sgl-omni serve --model-path $MODEL --model-name nemotron-transcribe \
    $SERVE_ARGS --host 127.0.0.1 --port $PORT "$@" > "$RUN_DIR/server_$tag.log" 2>&1 &
  local server_pid=$!
  local ok=0
  for i in $(seq 1 260); do
    if curl -sf "http://127.0.0.1:$PORT/health" | grep -q healthy; then ok=1; echo "server healthy after $((i * 15))s"; break; fi
    kill -0 "$server_pid" 2>/dev/null || { echo "server exited early"; break; }
    sleep 15
  done
  if [ $ok = 1 ]; then
    echo "== $tag: one request with spec details"
    curl -s "http://127.0.0.1:$PORT/v1/audio/transcriptions" -F model=nemotron-transcribe -F "file=@$CLIP" \
      -F response_format=verbose_json -F return_spec_tokens_details=true | tee "$RUN_DIR/one_request_$tag.json"; echo
    echo "== $tag: seedtts c=1"
    python -m benchmarks.eval.benchmark_asr_seedtts \
      --port $PORT --model-path nemotron-transcribe \
      --meta zhaochenyang20/seed-tts-eval-50-arrow --lang en --max-samples 0 \
      --concurrencies "${CONCS:-1,8,16}" --repeats "${REPEATS:-3}" --warmup --disable-resource-monitor \
      ${PROFILE:+--profile-events --profile-event-dir "$RUN_DIR/events_$tag"} \
      --output "$RUN_DIR/asr_bench_$tag.json" 2>&1 | tee "$RUN_DIR/asr_bench_$tag.txt" | tail -20
    if [ -n "${PROFILE:-}" ]; then
      echo "== $tag: stage breakdown"
      for d in "$RUN_DIR"/events_$tag/*/; do
        name=$(basename "$d"); conc=$(echo "$name" | sed -E 's/^asrbench-c([0-9]+)-.*/\1/')
        python "$REPO/benchmarks/profiling/nemotron_transcribe_profile/transcribe_report.py" "$d" --skip-first "$conc" \
          --out "$RUN_DIR/breakdown_${tag}_$name.json" | tee "$RUN_DIR/breakdown_${tag}_$name.txt"
        python -m sglang_omni.profiler "$d" --format table > "$RUN_DIR/stage_table_${tag}_$name.txt" 2>&1
      done
    else
      :
    fi
    echo "== $tag: accept len lines from the server log"
    grep -o "accept len: [0-9.]*, accept rate: [0-9.]*" "$RUN_DIR/server_$tag.log" | tail -5
  else
    tail -80 "$RUN_DIR/server_$tag.log"
  fi
  kill -TERM -- -"$server_pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$server_pid" 2>/dev/null || break; sleep 2; done
  kill -KILL -- -"$server_pid" 2>/dev/null
  return $((1 - ok))
}

MTP_ARGS="--asr.engine.speculative_algorithm NEXTN"
[ -n "${MTP_STEPS:-}" ] && MTP_ARGS="$MTP_ARGS --asr.engine.speculative_num_steps $MTP_STEPS --asr.engine.speculative_num_draft_tokens ${MTP_DRAFT:-$((MTP_STEPS + 1))}"
for mode in ${MODES:-mtp baseline}; do
  if [ "$mode" = mtp ]; then serve_and_bench "mtp${MTP_STEPS:+_s$MTP_STEPS}" $MTP_ARGS; else serve_and_bench baseline; fi
done
echo "done $(date)"
