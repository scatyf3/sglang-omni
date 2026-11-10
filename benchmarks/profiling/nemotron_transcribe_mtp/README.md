# Nemotron-3.5-Transcribe MTP acceptance

Measures how many of the native MTP drafts of `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` a greedy verifier accepts, as a function of draft depth, on real ASR transcripts. The model card ships one MTP layer (`llm.mtp.*`) that is reused for depths up to 8; Transformers drops those weights on load, so `mtp_acceptance.py` rebuilds the layer from the NemotronH block classes, loads the checkpoint tensors into it and chains it recursively the way the vLLM mtp method does.

- `mtp_acceptance.py` greedily transcribes each utterance with the target, replays prompt + transcript teacher-forced to get the target hidden states, and from every decode position proposes `k_max` drafts, counting how many consecutive drafts equal the target's greedy tokens. Both the post-final-norm and the pre-final-norm hidden states are tried as the MTP input (`post_norm` is the one that matches the checkpoint). The summary reports WER, the teacher-forced versus greedy agreement as a sanity check, conditional and marginal acceptance per depth, and for MTP1/2/4/8 the acceptance length (tokens per verification step) plus a walk of the actual greedy decode.
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
