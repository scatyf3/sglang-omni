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

### Stage breakdown at concurrency 1 (request events, 49 measured requests, server-side spans)

`PROFILE=1` makes the smoke script record request events and run `transcribe_report.py`. The pre-LM encoder is off under tensor parallelism, so audio encoding happens inside the prefill forward and is counted in prefill.

| | baseline | MTP7 |
|---|---|---|
| admission to prefill (build + queue) | 6.3 ms (2.9%) | 6.7 ms (4.5%) |
| prefill (incl. audio encoder) | 77.5 ms (36.5%) | 82.6 ms (56.0%) |
| decode | 131.7 ms (60.2%) | 58.5 ms (38.9%) |
| AR share (prefill + decode) | 209.1 ms (96.7%) | 141.1 ms (94.9%) |
| end to end | 216.2 ms | 148.7 ms |

The draft cuts decode by 56% (131.7 to 58.5 ms); prefill grows by 5 ms for the draft's own extend over the prompt. Prefill is now the larger half of a request, so the next lever on single-stream latency is the encoder and prefill, not decode.

## Reference sweep on one H200 (2026-11-11, bf16, pre-LM encoder on, `run_mtp_sweep.sbatch`)

seed-tts-eval-50 English, three measured repeats per concurrency (150 requests), corpus WER 0.0060 in every row, transcripts identical with and without the draft. `MTP7` is `--asr.engine.speculative_algorithm NEXTN` with 7 steps and 8 draft tokens.

| concurrency | mode | RTFx | latency p50 | latency p95 |
|---|---|---|---|---|
| 1 | baseline | 36.2 | 0.124 s | 0.158 s |
| 1 | MTP7 | 52.2 | 0.085 s | 0.096 s |
| 8 | baseline | 81.6 | 0.430 s | 0.653 s |
| 8 | MTP7 | 158.0 | 0.212 s | 0.282 s |
| 16 | baseline | 121.3 | 0.584 s | 0.965 s |
| 16 | MTP7 | 260.8 | 0.249 s | 0.300 s |

### Stage breakdown by the model's three parts (per-request spans, mean over the measured requests)

The decoder is the autoregressive part; `decoder prefill` is its pass over the prompt and `decoder decode` its token-by-token loop. On one GPU the Parallel Expert Encoder runs in the pre-LM encoder thread before the decoder sees the request, so its time sits inside `other` (request build, queue, dispatch); at concurrency 1 that whole bucket is 3.4 ms, which bounds the encoder. The CTC timestamp heads are not served.

| concurrency | mode | decoder prefill | decoder decode | decoder (AR share) | other | end to end |
|---|---|---|---|---|---|---|
| 1 | baseline | 54.4 ms (45%) | 65.9 ms (53%) | 120.3 ms (97.2%) | 3.4 ms | 123.8 ms |
| 1 | MTP7 | 58.0 ms (68%) | 24.0 ms (28%) | 82.0 ms (95.8%) | 3.6 ms | 85.6 ms |
| 8 | baseline | 57.3 ms (15%) | 318.6 ms (78%) | 375.9 ms (92.5%) | 30.4 ms | 406.3 ms |
| 8 | MTP7 | 61.3 ms (30%) | 80.3 ms (38%) | 141.6 ms (68.2%) | 66.5 ms | 208.1 ms |
| 16 | baseline | 56.0 ms (9%) | 523.0 ms (82%) | 579.0 ms (91.4%) | 48.0 ms | 626.9 ms |
| 16 | MTP7 | 61.3 ms (24%) | 99.2 ms (37%) | 160.6 ms (61.9%) | 98.5 ms | 259.1 ms |

The draft cuts decode by 2.7x at concurrency 1, 4.0x at 8 and 5.3x at 16; prefill grows by 4 to 5 ms for the draft's extend over the prompt and becomes the larger part of a single-stream request. At concurrency 8 and 16 the time a request spends between admission and its prefill (`other`) grows once decode is cheap, so the queue, not the decoder, is the next thing to look at under load. Unlike the L40S run, the gain keeps rising with concurrency on the H200.
