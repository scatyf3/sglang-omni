# Qwen3-Omni AR latency ratio profile

Measures how much of a speech-mode Qwen3-Omni request's end-to-end latency is spent in the autoregressive stages (thinker and talker_ar) versus preprocessing, encoders and code2wav, using the request event recorder described in docs/developer_reference/profiler.md.

- `ar_ratio_workload.py` starts a request profile on a running server, sends text or audio requests at a fixed concurrency with `modalities: ["text", "audio"]`, stops the profile and waits for the event sidecars.
- `ar_ratio_report.py` reads the event JSONL directory and prints per-stage spans plus `ar_sum_ratio` (thinker AR span + talker AR span over end-to-end) and `ar_union_ratio` (wall-clock time with at least one AR stage active over end-to-end). TTFT, TTFA and the code2wav tail after the talker finishes are reported alongside.
- `run_ar_ratio.sbatch` is the slurm driver for the NYU Torch cluster: one H200 or H100, colocated single-GPU server, warmup, then five cases (text c1, audio-in c1, text 512 tokens c1, text c4, text c8). Outputs land in the gitignored `.profiling-runs/qwen3_omni_ar_ratio/run_<jobid>/`.

```bash
sbatch benchmarks/profiling/qwen3_omni_ar_ratio/run_ar_ratio.sbatch
# or, against an already running server
python benchmarks/profiling/qwen3_omni_ar_ratio/ar_ratio_workload.py --run-id text_c1 --event-dir /tmp/ar/text_c1/events --mode text --num-requests 8
python benchmarks/profiling/qwen3_omni_ar_ratio/ar_ratio_report.py /tmp/ar/text_c1/events --out /tmp/ar/text_c1/ar_ratio.json
```
