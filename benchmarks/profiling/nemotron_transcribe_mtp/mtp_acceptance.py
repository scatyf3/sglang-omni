# SPDX-License-Identifier: Apache-2.0
"""Native MTP acceptance of Nemotron-3.5-Transcribe-30B-A4B, measured in PyTorch with Hugging Face Transformers.

Transformers loads the speech LM but drops the checkpoint's MTP weights (llm.mtp.*). This script rebuilds the MTP head
the way the NemotronH MTP module defines it:

    x = eh_proj(cat(enorm(embed(next_token)), hnorm(h)))   # layer 0 start projections
    x = attention block(x)                                   # NemotronH pre-norm block, no RoPE
    x = MoE block(x)                                         # layer 1
    o = final_layernorm(x); logits = lm_head(o)              # lm_head shared with the target

and applies the single MTP layer recursively for depths 2..K, as the vLLM mtp method with num_speculative_tokens K does
(the model card: "MTP8 reuses that layer to propose eight tokens").

For every utterance the target greedily transcribes the audio, then one teacher-forced pass over prompt + transcript gives
the target hidden states h_i and the verifier's greedy tokens T_i = argmax lm_head(h_i). Depth 1 runs over the whole
sequence with causal attention (its KV is the draft KV cache of the accepted context). From each decode position j the
chain proposes d_1..d_K, where depth s attends to depth-1 KV at positions <= j plus the chain's own earlier depths, and
d_s is accepted while it equals T_{j+s}. Greedy verification makes this exactly the acceptance vLLM would see, without
batching effects.
"""

from __future__ import annotations

import argparse
import copy
import glob
import io
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import jiwer
import librosa
import numpy as np
import pyarrow.parquet as pq
import soundfile
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.models.nemotron_h.modeling_nemotron_h import (
    NemotronHBlock,
    NemotronHModel,
)

MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
SEEDTTS_DATASET_ID = "zhaochenyang20/seed-tts-eval-50-arrow"
PROMPT = (
    "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
    "Formatting: punctuation and capitalization. Output: transcript only."
)
TARGET_SAMPLE_RATE = 16000
REPORTED_DEPTHS = (1, 2, 4, 8)
GIB = 2**30


@dataclass(kw_only=True)
class Utterance:
    utterance_id: str
    waveform: np.ndarray
    sample_rate: int
    reference_text: str


@dataclass(kw_only=True)
class ChainAcceptance:
    """Per-start run lengths plus the per-depth hit and validity masks, shape [depth_max, num_starts]."""

    runs: torch.Tensor
    match: torch.Tensor
    valid: torch.Tensor


def load_utterances(
    kind: str, limit: int | None, librispeech_parquet: str | None
) -> list[Utterance]:
    utterances: list[Utterance] = []
    if kind == "seedtts50":
        dataset_dir = snapshot_download(SEEDTTS_DATASET_ID, repo_type="dataset")
        rows = pq.read_table(
            glob.glob(f"{dataset_dir}/data/en-*.parquet")[0]
        ).to_pylist()
        for row in rows[:limit] if limit else rows:
            waveform, sample_rate = soundfile.read(
                io.BytesIO(row["ref_audio"]["bytes"]), dtype="float32"
            )
            utterances.append(
                Utterance(
                    utterance_id=row.get("id") or row["ref_audio_path"],
                    waveform=waveform,
                    sample_rate=sample_rate,
                    reference_text=row["ref_text"],
                )
            )
    elif kind == "librispeech":
        if librispeech_parquet is None:
            raise ValueError(
                "--librispeech-parquet is required with --data librispeech"
            )
        else:
            pass
        # HF parquet export of LibriSpeech test-clean (openslr/librispeech_asr, refs/convert/parquet, clean/test)
        rows = pq.read_table(librispeech_parquet).to_pylist()
        rows.sort(key=lambda row: row["id"])
        for row in rows[:limit] if limit else rows:
            waveform, sample_rate = soundfile.read(
                io.BytesIO(row["audio"]["bytes"]), dtype="float32"
            )
            utterances.append(
                Utterance(
                    utterance_id=row["id"],
                    waveform=waveform,
                    sample_rate=sample_rate,
                    reference_text=row["text"],
                )
            )
    else:
        raise ValueError(kind)
    return utterances


def resample_to_16k(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    if waveform.ndim > 1:
        waveform = waveform.mean(-1)
    else:
        pass
    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE
        )
    else:
        pass
    return waveform


def normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, epsilon: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.epsilon)
        return (self.weight.float() * x).to(input_dtype)


class SharedVocab:
    """The target's input embedding and lm_head weights, copied to the device that runs the MTP head."""

    def __init__(
        self,
        embedding_weight: torch.Tensor,
        head_weight: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.embedding_weight = embedding_weight.detach().to(device)
        self.head_weight = head_weight.detach().to(device)
        self.device = device

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids.to(self.device), self.embedding_weight)

    def logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states.to(self.device), self.head_weight)


class MTPHead(torch.nn.Module):
    """One NemotronH MTP layer (attention block + MoE block) built from the HF block classes."""

    def __init__(
        self,
        backbone: NemotronHModel,
        checkpoint_path: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        config = copy.deepcopy(backbone.config)
        config.layers_block_type = ["full_attention", "moe"]
        config.num_hidden_layers = 2
        self.attn_block = NemotronHBlock(config, 0)
        self.moe_block = NemotronHBlock(config, 1)
        epsilon = config.layer_norm_epsilon
        with safe_open(checkpoint_path, "pt", device="cpu") as checkpoint:
            mtp_weights = {
                name[len("llm.mtp.layers.") :]: checkpoint.get_tensor(name)
                for name in checkpoint.keys()
                if name.startswith("llm.mtp.layers.")
                and not name.endswith("_extra_state")
            }
            moe_layer = next(
                index
                for index, block in enumerate(backbone.layers)
                if block.block_type == "moe"
            )
            target_up_proj = checkpoint.get_tensor(
                f"llm.model.layers.{moe_layer}.mixer.experts.gate_and_up_projs"
            )
        self.enorm = RMSNorm(mtp_weights.pop("0.enorm.weight"), epsilon)
        self.hnorm = RMSNorm(mtp_weights.pop("0.hnorm.weight"), epsilon)
        self.final_layernorm = RMSNorm(
            mtp_weights.pop("1.final_layernorm.weight"), epsilon
        )
        self.eh_proj = torch.nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        self.eh_proj.weight.data.copy_(mtp_weights.pop("0.eh_proj.weight"))

        # NeMo stacks experts as [E, in, out]; HF keeps the F.linear layout [E, out, in]. Check the convention on a
        # target MoE layer that Transformers already converted, then apply the same one to the MTP experts.
        converted_up_proj = (
            backbone.layers[moe_layer].mixer.experts.up_proj.detach().cpu()
        )
        if target_up_proj.shape == converted_up_proj.shape and torch.equal(
            target_up_proj.to(converted_up_proj.dtype), converted_up_proj
        ):
            self.expert_transpose = False
        elif target_up_proj.transpose(
            1, 2
        ).shape == converted_up_proj.shape and torch.equal(
            target_up_proj.transpose(1, 2).to(converted_up_proj.dtype),
            converted_up_proj,
        ):
            self.expert_transpose = True
        else:
            raise RuntimeError(
                "cannot match the NeMo expert layout to the HF converted target layer"
            )

        def expert_layout(weight: torch.Tensor) -> torch.Tensor:
            return (
                weight.transpose(1, 2).contiguous() if self.expert_transpose else weight
            )

        attn_state = {
            "norm.weight": mtp_weights.pop("0.norm.weight"),
            **{
                f"mixer.{projection}.weight": mtp_weights.pop(
                    f"0.mixer.{projection}.weight"
                )
                for projection in ("q_proj", "k_proj", "v_proj", "o_proj")
            },
        }
        correction_bias_fp32 = mtp_weights.pop(
            "1.mixer.gate.e_score_correction_bias"
        ).float()
        moe_state = {
            "norm.weight": mtp_weights.pop("1.norm.weight"),
            "mixer.gate.weight": mtp_weights.pop("1.mixer.gate.weight"),
            "mixer.gate.e_score_correction_bias": correction_bias_fp32,
            "mixer.shared_experts.up_proj.weight": mtp_weights.pop(
                "1.mixer.shared_experts.up_proj.weight"
            ),
            "mixer.shared_experts.down_proj.weight": mtp_weights.pop(
                "1.mixer.shared_experts.down_proj.weight"
            ),
            "mixer.experts.up_proj": expert_layout(
                mtp_weights.pop("1.mixer.experts.gate_and_up_projs")
            ),
            "mixer.experts.down_proj": expert_layout(
                mtp_weights.pop("1.mixer.experts.down_projs")
            ),
        }
        if mtp_weights:
            raise RuntimeError(f"unused MTP weights: {sorted(mtp_weights)}")
        else:
            pass
        for block, state in (
            (self.attn_block, attn_state),
            (self.moe_block, moe_state),
        ):
            missing, unexpected = block.load_state_dict(state, strict=False)
            missing = [name for name in missing if "latent" not in name]
            if missing or unexpected:
                raise RuntimeError(
                    f"MTP block load: missing={missing} unexpected={unexpected}"
                )
            else:
                pass
        self.to(device=device, dtype=dtype)
        # the router computes in fp32; keep the correction bias at checkpoint precision after the bf16 cast
        self.moe_block.mixer.gate.e_score_correction_bias.data = (
            correction_bias_fp32.to(device)
        )
        attention = self.attn_block.mixer
        self.num_heads = attention.q_proj.out_features // attention.head_dim
        self.num_kv_heads = attention.k_proj.out_features // attention.head_dim
        self.head_dim = attention.head_dim

    def start(
        self, next_token_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        return self.eh_proj(
            torch.cat(
                [self.enorm(next_token_embeds), self.hnorm(hidden_states)], dim=-1
            )
        )

    def qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attention = self.attn_block.mixer
        normed = self.attn_block.norm(x)
        q = attention.q_proj(normed).view(*x.shape[:-1], self.num_heads, self.head_dim)
        k = attention.k_proj(normed).view(
            *x.shape[:-1], self.num_kv_heads, self.head_dim
        )
        v = attention.v_proj(normed).view(
            *x.shape[:-1], self.num_kv_heads, self.head_dim
        )
        return q, k, v

    def attend(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """q: [N, H, D]; keys, values: [M, KV, D]; mask: [N, M] bool (True = visible)."""
        repeats = self.num_heads // self.num_kv_heads
        keys_per_head = keys.repeat_interleave(repeats, dim=1).transpose(
            0, 1
        )  # [H, M, D]
        values_per_head = values.repeat_interleave(repeats, dim=1).transpose(0, 1)
        attended = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            keys_per_head.unsqueeze(0),
            values_per_head.unsqueeze(0),
            attn_mask=mask.unsqueeze(0).unsqueeze(0),
        )
        attended = attended.squeeze(0).transpose(0, 1).reshape(x.shape[0], -1)
        return x + self.attn_block.mixer.o_proj(attended)

    def finish(self, x: torch.Tensor) -> torch.Tensor:
        x = self.moe_block(x.unsqueeze(0)).squeeze(0)
        return self.final_layernorm(x)


@torch.no_grad()
def chain_acceptance(
    mtp_head: MTPHead,
    vocab: SharedVocab,
    token_embeds: torch.Tensor,
    hidden_states: torch.Tensor,
    target_tokens: torch.Tensor,
    prompt_len: int,
    depth_max: int,
) -> ChainAcceptance | None:
    """Run lengths (accepted drafts, capped at depth_max) for every decode start j = prompt_len-1 .. seq_len-3."""
    seq_len = token_embeds.shape[0]
    num_pairs = seq_len - 1
    device = token_embeds.device
    x = mtp_head.start(
        token_embeds[1:], hidden_states[:-1]
    )  # pair i = (token i+1, h_i), i = 0..seq_len-2
    q, k1, v1 = mtp_head.qkv(x)
    causal = torch.ones(num_pairs, num_pairs, dtype=torch.bool, device=device).tril()
    depth1_states = mtp_head.finish(mtp_head.attend(x, q, k1, v1, causal))
    depth1_tokens = vocab.logits(depth1_states).argmax(
        -1
    )  # proposes token i+2, verified against target i+1

    starts = torch.arange(prompt_len - 1, seq_len - 2, device=device)
    if starts.numel() == 0:
        return None
    else:
        pass
    num_starts = starts.numel()
    alive = depth1_tokens[starts] == target_tokens[starts + 1]
    runs = alive.long().clone()
    match_by_depth = [alive.clone()]
    valid_by_depth = [torch.ones(num_starts, dtype=torch.bool, device=device)]
    previous_tokens, previous_states = depth1_tokens[starts], depth1_states[starts]
    chain_keys: list[torch.Tensor] = []
    chain_values: list[torch.Tensor] = []
    base_mask = (
        torch.arange(num_pairs, device=device)[None, :] <= starts[:, None]
    )  # depth-1 KV at positions <= j
    for depth in range(2, depth_max + 1):
        x_depth = mtp_head.start(vocab.embed(previous_tokens), previous_states)
        q_depth, k_depth, v_depth = mtp_head.qkv(x_depth)
        chain_keys.append(k_depth)
        chain_values.append(v_depth)
        eye = torch.eye(num_starts, dtype=torch.bool, device=device)
        mask = torch.cat([base_mask] + [eye] * len(chain_keys), 1)
        draft_states = mtp_head.finish(
            mtp_head.attend(
                x_depth,
                q_depth,
                torch.cat([k1] + chain_keys, 0),
                torch.cat([v1] + chain_values, 0),
                mask,
            )
        )
        draft_tokens = vocab.logits(draft_states).argmax(-1)
        target_positions = starts + depth
        valid = target_positions <= seq_len - 1
        hit = valid & (
            draft_tokens == target_tokens[target_positions.clamp(max=seq_len - 1)]
        )
        match_by_depth.append(hit)
        valid_by_depth.append(valid)
        alive = alive & hit
        runs += alive.long()
        previous_tokens, previous_states = draft_tokens, draft_states
    return ChainAcceptance(
        runs=runs.cpu(),
        match=torch.stack(match_by_depth).cpu(),
        valid=torch.stack(valid_by_depth).cpu(),
    )


def verification_steps(runs: list[int], num_tokens: int, depth: int) -> int:
    """Verification steps a greedy MTP-depth decode needs to emit num_tokens after the prefill token.

    runs[c] is the accepted-draft count when the drafts start right after the (c+1)-th emitted token.
    """
    emitted, steps = 0, 0
    while emitted < num_tokens:
        accepted = min(runs[emitted], depth) if emitted < len(runs) else 0
        steps += 1
        emitted += accepted + 1
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=None,
        help="local snapshot; defaults to the HF cache entry",
    )
    parser.add_argument(
        "--data", choices=["seedtts50", "librispeech"], default="seedtts50"
    )
    parser.add_argument(
        "--librispeech-parquet",
        default=None,
        help="clean/test parquet of openslr/librispeech_asr",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_dir = args.model_dir or snapshot_download(MODEL_ID)
    checkpoint_path = str(Path(model_dir) / "model.safetensors")
    dtype = torch.bfloat16
    num_gpus = torch.cuda.device_count()
    input_device = torch.device("cuda:0")
    head_device = torch.device(
        f"cuda:{num_gpus - 1}"
    )  # the MTP head and its vocab copies live on the last device
    load_start = time.time()
    processor = AutoProcessor.from_pretrained(model_dir)
    if num_gpus == 1:
        model = AutoModelForMultimodalLM.from_pretrained(
            model_dir, dtype=dtype, device_map="cuda"
        ).eval()
    else:
        # e.g. 2x L40S (48 GB): spread the 67.5 GB checkpoint, leaving room on the last GPU for the MTP head
        total_gib = torch.cuda.get_device_properties(0).total_memory / GIB
        max_memory = {index: f"{int(total_gib - 6)}GiB" for index in range(num_gpus)}
        max_memory[num_gpus - 1] = f"{int(total_gib - 10)}GiB"
        max_memory["cpu"] = "0GiB"  # fail instead of silently offloading layers to CPU
        model = AutoModelForMultimodalLM.from_pretrained(
            model_dir, dtype=dtype, device_map="auto", max_memory=max_memory
        ).eval()
    backbone = model.model.language_model
    vocab = SharedVocab(
        backbone.get_input_embeddings().weight,
        model.get_output_embeddings().weight,
        head_device,
    )
    mtp_head = MTPHead(backbone, checkpoint_path, head_device, dtype).eval()
    allocated_gib = [
        round(torch.cuda.memory_allocated(index) / GIB, 1) for index in range(num_gpus)
    ]
    print(
        f"loaded in {time.time() - load_start:.0f}s gpus={num_gpus} expert_transpose={mtp_head.expert_transpose} "
        f"mem={allocated_gib}GiB",
        flush=True,
    )

    captured: dict[str, torch.Tensor] = {}

    # hooks must return None: a non-None return value replaces the module's input or output
    def grab_embeds(
        module: torch.nn.Module,
        hook_args: tuple[torch.Tensor, ...],
        hook_kwargs: dict[str, torch.Tensor],
    ) -> None:
        captured["embeds"] = hook_kwargs["inputs_embeds"]
        return None

    def grab_norm(
        module: torch.nn.Module,
        hook_args: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured["pre"] = hook_args[0]
        captured["post"] = output
        return None

    hooks = [
        backbone.register_forward_pre_hook(grab_embeds, with_kwargs=True),
        backbone.norm_f.register_forward_hook(grab_norm),
    ]

    utterances = load_utterances(args.data, args.limit, args.librispeech_parquet)
    depth_max = args.k_max
    per_utterance: list[dict[str, object]] = []
    aggregate = {
        source: {"runs": [], "match": [], "valid": []}
        for source in ("post_norm", "pre_norm")
    }
    references: list[str] = []
    hypotheses: list[str] = []
    consistent_tokens = generated_tokens = 0
    for index, utterance in enumerate(utterances):
        waveform = resample_to_16k(utterance.waveform, utterance.sample_rate)
        inputs = processor.apply_transcription_request(waveform, prompt=PROMPT).to(
            input_device
        )
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(dtype)
        else:
            pass
        prompt_len = inputs["input_ids"].shape[1]
        captured.clear()
        generated_ids = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
        hypothesis = processor.decode(
            generated_ids[0, prompt_len:], skip_special_tokens=True
        )
        teacher_forced = dict(inputs)
        teacher_forced["input_ids"] = generated_ids
        if "attention_mask" in teacher_forced:
            teacher_forced["attention_mask"] = torch.ones_like(generated_ids)
        else:
            pass
        captured.clear()
        model(**teacher_forced, use_cache=False)
        token_embeds, post_norm_states, pre_norm_states = (
            captured[name][0].to(head_device) for name in ("embeds", "post", "pre")
        )
        target_tokens = vocab.logits(post_norm_states).argmax(-1)
        seq_len = generated_ids.shape[1]
        # the teacher-forced argmax should reproduce the greedy transcript
        teacher_forced_tokens = target_tokens[prompt_len - 1 : seq_len - 1]
        consistent_tokens += int(
            (
                teacher_forced_tokens == generated_ids[0, prompt_len:].to(head_device)
            ).sum()
        )
        generated_tokens += seq_len - prompt_len
        row: dict[str, object] = {
            "id": utterance.utterance_id,
            "prompt_len": prompt_len,
            "gen_len": seq_len - prompt_len,
            "hyp": hypothesis,
            "ref": utterance.reference_text,
        }
        for source, states in (
            ("post_norm", post_norm_states),
            ("pre_norm", pre_norm_states),
        ):
            acceptance = chain_acceptance(
                mtp_head,
                vocab,
                token_embeds,
                states,
                target_tokens,
                prompt_len,
                depth_max,
            )
            if acceptance is None:
                runs: list[int] = []
            else:
                aggregate[source]["runs"].append(acceptance.runs)
                aggregate[source]["match"].append(acceptance.match)
                aggregate[source]["valid"].append(acceptance.valid)
                runs = acceptance.runs.tolist()
            row[source] = {
                "runs": runs,
                "steps": {
                    depth: verification_steps(runs, seq_len - prompt_len - 1, depth)
                    for depth in REPORTED_DEPTHS
                    if depth <= depth_max
                },
            }
        per_utterance.append(row)
        references.append(normalize_text(utterance.reference_text))
        hypotheses.append(normalize_text(hypothesis))
        if index % 10 == 0:
            print(
                f"[{index + 1}/{len(utterances)}] prompt={prompt_len} gen={seq_len - prompt_len} "
                f"runs(post)={row['post_norm']['runs'][:12]} hyp={hypothesis[:70]!r}",
                flush=True,
            )
        else:
            pass
    for hook in hooks:
        hook.remove()

    summary: dict[str, object] = {
        "model": MODEL_ID,
        "data": args.data,
        "n_utts": len(utterances),
        "k_max": depth_max,
        "wer": jiwer.wer(references, hypotheses),
        "teacher_forced_greedy_agreement": consistent_tokens / max(generated_tokens, 1),
        "gen_tokens": generated_tokens,
        "expert_transpose": mtp_head.expert_transpose,
    }
    for source in ("post_norm", "pre_norm"):
        runs_all = torch.cat(aggregate[source]["runs"]).float()
        match = torch.cat(aggregate[source]["match"], 1)
        valid = torch.cat(aggregate[source]["valid"], 1)
        source_summary: dict[str, object] = {"n_starts": int(runs_all.numel())}
        # conditional acceptance at depth s: P(d_s correct | d_1..d_{s-1} correct)
        conditional: list[float] = []
        previous_hit = torch.ones(match.shape[1], dtype=torch.bool)
        for depth_index in range(depth_max):
            eligible = previous_hit & valid[depth_index]
            conditional.append(
                float((match[depth_index] & eligible).sum() / eligible.sum())
                if eligible.sum()
                else math.nan
            )
            previous_hit = previous_hit & match[depth_index]
        source_summary["conditional_acceptance_by_depth"] = [
            round(value, 4) for value in conditional
        ]
        source_summary["marginal_acceptance_by_depth"] = [
            round(
                float(
                    (match[depth_index] & valid[depth_index]).sum()
                    / valid[depth_index].sum()
                ),
                4,
            )
            for depth_index in range(depth_max)
        ]
        for depth in range(1, depth_max + 1):
            accepted = runs_all.clamp(max=depth)
            if depth in REPORTED_DEPTHS:
                steps = sum(row[source]["steps"][depth] for row in per_utterance)
            else:
                steps = 0
            source_summary[f"mtp{depth}"] = {
                "mean_accepted_drafts": round(float(accepted.mean()), 4),
                "acceptance_length": round(
                    1 + float(accepted.mean()), 4
                ),  # tokens per verification step, per start
                "draft_acceptance_rate": round(float(accepted.mean()) / depth, 4),
                "walk_tokens_per_step": (
                    round((generated_tokens - len(per_utterance)) / steps, 4)
                    if steps
                    else None
                ),
            }
        summary[source] = source_summary
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps({"summary": summary, "utterances": per_utterance}, indent=2)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
