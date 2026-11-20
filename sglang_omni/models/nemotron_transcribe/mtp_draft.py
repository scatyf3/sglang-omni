# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's native MTP layer as an SGLang draft for Nemotron Transcribe."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    MTP_CHECKPOINT_PREFIX,
    classify_checkpoint_name,
)
from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
)

MTP_LAYER_PARAMETER_PREFIX = "model.layers."
FUSED_PARAMETER_CHECKPOINT_NAMES = {
    ".mixer.qkv_proj.weight": (
        ".mixer.q_proj.weight",
        ".mixer.k_proj.weight",
        ".mixer.v_proj.weight",
    ),
    ".mixer.experts.w13_weight": (".mixer.experts.gate_and_up_projs",),
    ".mixer.experts.w2_weight": (".mixer.experts.down_projs",),
}
MAX_REPORTED_CHECKPOINT_NAMES = 8


def expected_mtp_checkpoint_names(parameter_names: Iterable[str]) -> set[str]:
    """Return the llm.mtp. tensors that fill every MTP layer parameter of the draft."""
    expected_names: set[str] = set()
    for parameter_name in parameter_names:
        if not parameter_name.startswith(MTP_LAYER_PARAMETER_PREFIX):
            # note (scatyf3): the draft shares embed_tokens and lm_head with the target.
            continue
        else:
            pass
        checkpoint_name = MTP_CHECKPOINT_PREFIX + parameter_name[len("model.") :]
        fused_suffix = next(
            (
                parameter_suffix
                for parameter_suffix in FUSED_PARAMETER_CHECKPOINT_NAMES
                if checkpoint_name.endswith(parameter_suffix)
            ),
            None,
        )
        if fused_suffix is None:
            expected_names.add(checkpoint_name)
        else:
            layer_name = checkpoint_name[: -len(fused_suffix)]
            expected_names.update(
                layer_name + checkpoint_suffix
                for checkpoint_suffix in FUSED_PARAMETER_CHECKPOINT_NAMES[fused_suffix]
            )
    return expected_names


class NemotronTranscribeMTPDraft(NemotronHForCausalLMMTP):

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        """Load the llm.mtp. layer and fail unless it covers every draft parameter."""
        del is_mtp
        expected_names = expected_mtp_checkpoint_names(
            name for name, _ in self.named_parameters()
        )
        loaded_names: set[str] = set()
        unmapped_names: set[str] = set()

        def inventoried_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                kind, _ = classify_checkpoint_name(name, is_mtp_draft=True)
                if kind == "skip":
                    continue
                elif name in expected_names:
                    loaded_names.add(name)
                    yield name, tensor
                else:
                    unmapped_names.add(name)

        # note (scatyf3): names are checked while streaming because buffering the whole target checkpoint would double host memory.
        super().load_weights(
            checkpoint_to_language_model_weights(
                inventoried_weights(), is_mtp_draft=True
            ),
            is_mtp=True,
        )
        missing_names = expected_names - loaded_names
        if missing_names or unmapped_names:
            raise ValueError(
                "Nemotron Transcribe checkpoint does not match the native MTP draft "
                f"layout ({self.config.mtp_hybrid_override_pattern}): missing "
                f"{len(missing_names)} tensors "
                f"{sorted(missing_names)[:MAX_REPORTED_CHECKPOINT_NAMES]}, unmapped "
                f"{len(unmapped_names)} tensors "
                f"{sorted(unmapped_names)[:MAX_REPORTED_CHECKPOINT_NAMES]}; serve a "
                "checkpoint with its complete llm.mtp. layer or turn off "
                "speculative decoding"
            )
        else:
            pass


EntryClass = NemotronTranscribeMTPDraft
