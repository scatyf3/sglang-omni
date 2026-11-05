# Qwen3-Omni AR latency ratio profile

Measures how much of a speech-mode Qwen3-Omni request's end-to-end latency is spent in the autoregressive stages (thinker and talker_ar) versus preprocessing, encoders and code2wav, using the request event recorder described in docs/developer_reference/profiler.md.

- `ar_ratio_workload.py` starts a request profile on a running server, sends text or audio requests at a fixed concurrency with `modalities: ["text", "audio"]`, stops the profile and waits for the event sidecars.
- `ar_ratio_report.py` reads the event JSONL directory and prints per-stage spans plus `ar_sum_ratio` (thinker AR span + talker AR span over end-to-end) and `ar_union_ratio` (wall-clock time with at least one AR stage active over end-to-end). TTFT, TTFA and the code2wav tail after the talker finishes are reported alongside.
- `run_ar_ratio.sbatch` is the slurm driver for the NYU Torch cluster. It follows the Qwen3-Omni TTS CI protocol: one H200 or H100, colocated single-GPU speech server, then `benchmarks/eval/benchmark_omni_seedtts.py --generate-only --voice-clone` on `zhaochenyang20/seed-tts-eval-50-arrow` (50 samples, 256 new tokens, temperature 0.7, talker defaults) at concurrency 1, 8 and 16, each inside a request profile whose warmup requests are skipped in the report. The omni server is then stopped and the same GPU serves Qwen3-ASR-1.7B for WER (`--transcribe-only`, ASR concurrency 4), followed by `--similarity-only` and `--utmos-only`. One hand-written text case remains as a sanity check. Outputs land in the gitignored `.profiling-runs/qwen3_omni_ar_ratio/run_<jobid>/`.
- `summarize_cases.py` joins each case's speed, WER, similarity, UTMOS and AR ratio results into `summary.md` and `summary.json`.

```bash
sbatch benchmarks/profiling/qwen3_omni_ar_ratio/run_ar_ratio.sbatch
# or, against an already running server
python benchmarks/profiling/qwen3_omni_ar_ratio/ar_ratio_workload.py --run-id text_c1 --event-dir /tmp/ar/text_c1/events --mode text --num-requests 8
python benchmarks/profiling/qwen3_omni_ar_ratio/ar_ratio_report.py /tmp/ar/text_c1/events --out /tmp/ar/text_c1/ar_ratio.json
```
