# Nemotron-3.5-Transcribe MTP acceptance

Measures how many of the native MTP drafts of `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` a greedy verifier accepts, as a function of draft depth, on real ASR transcripts. The model card ships one MTP layer (`llm.mtp.*`) that is reused for depths up to 8; Transformers drops those weights on load, so `mtp_acceptance.py` rebuilds the layer from the NemotronH block classes, loads the checkpoint tensors into it and chains it recursively the way the vLLM mtp method does.

- `mtp_acceptance.py` greedily transcribes each utterance with the target, replays prompt + transcript teacher-forced to get the target hidden states, and from every decode position proposes `k_max` drafts, counting how many consecutive drafts equal the target's greedy tokens. Both the post-final-norm and the pre-final-norm hidden states are tried as the MTP input (`post_norm` is the one that matches the checkpoint). The summary reports WER, the teacher-forced versus greedy agreement as a sanity check, conditional and marginal acceptance per depth, and for MTP1/2/4/8 the acceptance length (tokens per verification step) plus a walk of the actual greedy decode.
- `run_mtp_smoke_l40s.sbatch` serves the checkpoint through sglang-omni on 2x L40S (`--asr.tp_size 2`) with and without the native MTP draft and runs the seed-tts-eval-50 ASR benchmark against each; `MODES`, `MTP_STEPS`, `MTP_DRAFT` and `CONCS` select the runs. Outputs land in `.profiling-runs/nemotron_transcribe_mtp/smoke_<jobid>/`.
- `run_mtp_acceptance.sbatch` runs seed-tts-eval-50 (English) and then the first 500 LibriSpeech test-clean utterances on one H100/H200. `run_mtp_acceptance_l40s.sbatch` is the same on 2x L40S with the checkpoint split across both GPUs. Both read the HF cache at `HF_HOME` and the LibriSpeech test-clean parquet export (`openslr/librispeech_asr`, `refs/convert/parquet`, `clean/test`). Outputs land in the gitignored `.profiling-runs/nemotron_transcribe_mtp/acceptance_<jobid>/`.

```bash
sbatch benchmarks/profiling/nemotron_transcribe_mtp/run_mtp_acceptance_l40s.sbatch
# or directly
python benchmarks/profiling/nemotron_transcribe_mtp/mtp_acceptance.py --data seedtts50 --k-max 8 --out /tmp/seedtts50.json
python benchmarks/profiling/nemotron_transcribe_mtp/mtp_acceptance.py --data librispeech --librispeech-parquet <clean/test/0000.parquet> --limit 300 --k-max 8 --out /tmp/librispeech.json
```

## Results (2026-11-10, 2x L40S, bf16, greedy, post_norm)

| dataset | utterances | WER | teacher-forced agreement | MTP2 acceptance length | MTP4 acceptance length | MTP8 acceptance length | depth-1 conditional | depth-8 conditional |
|---|---|---|---|---|---|---|---|---|
| seed-tts-eval-50 en | 50 | 0.0046 | 1.000 | 2.98 | 4.74 | 7.43 | 0.993 | 0.968 |
| LibriSpeech test-clean (first 300) | 300 | 0.0130 | 0.994 | 2.92 | 4.63 | 7.32 | 0.974 | 0.935 |

Acceptance length is 1 + mean accepted drafts per verification step, so MTP8 advances about 7.3 tokens per target forward on transcripts. The conditional acceptance decays slowly with depth, which is what makes the deep single-layer chain worthwhile for ASR output.

## Serving with the draft (2026-11-10, 2x L40S tensor-parallel, bf16, sglang-omni)

seed-tts-eval-50 English through `/v1/audio/transcriptions`, one measured repeat per concurrency, corpus WER 0.0060 in every row. `MTP7` is `--asr.engine.speculative_algorithm NEXTN` with 7 steps and 8 draft tokens; `MTP3` is the default 3 steps and 4 draft tokens.

| concurrency | mode | RTFx | latency p50 | latency p95 | accept length (of max) |
|---|---|---|---|---|---|
| 1 | baseline | 21.5 | 0.206 s | 0.274 s | - |
| 1 | MTP3 | 27.5 | 0.160 s | 0.193 s | 3.9 of 4 |
| 1 | MTP7 | 30.6 | 0.142 s | 0.169 s | 7.0 to 7.5 of 8 |
| 8 | baseline | 48.9 | 0.709 s | 1.040 s | - |
| 8 | MTP7 | 98.2 | 0.350 s | 0.406 s | - |
| 16 | baseline | 71.6 | 1.012 s | 1.352 s | - |
| 16 | MTP7 | 132.1 | 0.447 s | 0.722 s | - |

The served acceptance matches the offline measurement above (MTP3 3.8 to 3.9, MTP8 7.3 to 7.4). The 30B-A4B decoder is memory-bound on L40S even at concurrency 16, so verifying eight tokens per forward keeps paying: the draft roughly doubles throughput at concurrency 8 and 16 and cuts single-stream latency by a third.
