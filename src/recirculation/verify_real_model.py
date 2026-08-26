"""Real-checkpoint verification of equivalence, injection, and causality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM

import recirculation.gemma3_recirculation as recirculation_module

from .constants import (
    ALPHA,
    ATTENTION_IMPLEMENTATION,
    BETA,
    DESTINATION_LAYER,
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)
from .data import load_manifest
from .gemma3_recirculation import Gemma3ForCausalLM


def load(model_class):
    return (
        model_class.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation=ATTENTION_IMPLEMENTATION,
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .to("mps")
    )


def run(manifest_path: Path) -> dict:
    ids = torch.tensor(
        [load_manifest(manifest_path)["windows"][0]["token_ids"][:530]],
        dtype=torch.long,
        device="mps",
    )
    mask = torch.ones_like(ids)
    official = load(HFGemma3ForCausalLM)
    adapter = load(Gemma3ForCausalLM)
    versions = tuple(parameter._version for parameter in adapter.parameters())

    with torch.inference_mode():
        official_logits = official(ids, attention_mask=mask, use_cache=False).logits
        adapter_logits = adapter(ids, attention_mask=mask, use_cache=False).logits
    baseline_max_abs = float((official_logits - adapter_logits).abs().max().item())
    baseline_exact = bool(torch.equal(official_logits, adapter_logits))
    del official_logits, adapter_logits, official

    adapter.set_recirculation_args(
        target_layer=DESTINATION_LAYER,
        source_layer=SOURCE_LAYER,
        target_layer_weight=BETA,
        source_layer_weight=ALPHA,
        num_recurrence_steps=NUM_RECURRENCE_STEPS,
        normalization=NORMALIZATION,
        ramp_steps=RAMP_STEPS,
    )

    shapes = []
    captured = []
    real_mix = recirculation_module.fixed_recirculation_mix

    def observing_mix(destination, source, **kwargs):
        shapes.append([list(destination.shape), list(source.shape)])
        if len(shapes) == RAMP_STEPS + 1:
            captured.append(
                (destination.detach().clone(), source.detach().clone(), kwargs.copy())
            )
        return real_mix(destination, source, **kwargs)

    with (
        patch.object(
            recirculation_module, "fixed_recirculation_mix", side_effect=observing_mix
        ),
        torch.inference_mode(),
    ):
        recirculated_output = adapter(ids, attention_mask=mask, use_cache=False)

    destination, source, mix_kwargs = captured[0]
    scaled_source = source / (source.norm(dim=-1, keepdim=True) + 1e-8)
    scaled_source = scaled_source * destination.norm(dim=-1, keepdim=True)
    destination_norm = destination.norm(dim=-1)
    norm_error = (scaled_source.norm(dim=-1) - destination_norm).abs()
    norm_match_error = float(norm_error.max().item())
    norm_match_relative_error = float(
        (norm_error / destination_norm.clamp_min(1e-8)).max().item()
    )

    changed = ids.clone()
    future_change_position = 514
    changed[:, future_change_position:] = torch.flip(
        changed[:, future_change_position:], dims=(1,)
    )
    with torch.inference_mode():
        changed_output = adapter(
            changed, attention_mask=torch.ones_like(changed), use_cache=False
        )
    causal_prefix_max_abs = float(
        (
            recirculated_output.logits[:, :future_change_position]
            - changed_output.logits[:, :future_change_position]
        )
        .abs()
        .max()
        .item()
    )
    cache_lengths = [
        int(layer.keys.shape[-2])
        for layer in recirculated_output.past_key_values.layers
    ]
    expected_cache_lengths = [
        min(ids.shape[1] - NUM_RECURRENCE_STEPS, adapter.config.sliding_window - 1)
        if layer_type == "sliding_attention"
        else ids.shape[1] - NUM_RECURRENCE_STEPS
        for layer_type in adapter.config.layer_types
    ]

    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "fixture_tokens": ids.shape[1],
        "official_vs_public_adapter_baseline_bitwise_equal": baseline_exact,
        "official_vs_public_adapter_baseline_max_abs_logit_difference": baseline_max_abs,
        "recirculation_mix_call_count": len(shapes),
        "recirculation_unique_destination_source_shapes": sorted(
            {json.dumps(shape) for shape in shapes}
        ),
        "configured_destination_layer_zero_based": DESTINATION_LAYER,
        "configured_source_layer_zero_based": SOURCE_LAYER,
        "captured_full_ramp_source_weight": mix_kwargs["source_weight"],
        "captured_full_ramp_destination_weight": mix_kwargs["destination_weight"],
        "max_norm_match_absolute_error": norm_match_error,
        "max_norm_match_relative_error": norm_match_relative_error,
        "captured_destination_norm_min": float(destination_norm.min().item()),
        "captured_destination_norm_max": float(destination_norm.max().item()),
        "sliding_window": adapter.config.sliding_window,
        "returned_cache_layer_lengths": cache_lengths,
        "expected_cache_layer_lengths": expected_cache_lengths,
        "cache_layer_lengths_match": cache_lengths == expected_cache_lengths,
        "future_changed_from_position": future_change_position,
        "causal_unchanged_prefix_max_abs_logit_difference": causal_prefix_max_abs,
        "parameter_version_counters_unchanged": versions
        == tuple(parameter._version for parameter in adapter.parameters()),
        "all_weights_frozen": not any(
            parameter.requires_grad for parameter in adapter.parameters()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("experiments/pg19_windows.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/verification.json")
    )
    args = parser.parse_args()
    result = run(args.manifest)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
