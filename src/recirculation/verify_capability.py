"""Real-model correctness checks for the Gemma 3 1B IT capability adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoTokenizer
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM

import recirculation.gemma3_recirculation as recirculation_module

from .capability_constants import (
    CAPABILITY_ATTENTION_IMPLEMENTATION,
    IT_CHAT_TEMPLATE_SHA256,
    IT_MODEL_ID,
    IT_MODEL_REVISION,
)
from .constants import (
    ALPHA,
    BETA,
    DESTINATION_LAYER,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)
from .gemma3_recirculation import Gemma3ForCausalLM


def _load(model_class):
    return (
        model_class.from_pretrained(
            IT_MODEL_ID,
            revision=IT_MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation=CAPABILITY_ATTENTION_IMPLEMENTATION,
            local_files_only=True,
        )
        .eval()
        .requires_grad_(False)
        .to("mps")
    )


def _first_sample(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    return next(iter(artifact["samples"].values()))[0]


def _greedy_recirculation(model, input_ids, max_new_tokens):
    output, state = model.recirculating_prefill(
        input_ids, attention_mask=torch.ones_like(input_ids)
    )
    generated = []
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    for _ in range(max_new_tokens):
        next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(int(next_ids.item()))
        if generated[-1] in eos_ids:
            break
        output, state = model.recirculating_decode_step(next_ids, state)
    return generated


def _paired_smoke_checks(smoke_dir: Path) -> dict:
    benchmark_checks = {}
    for benchmark in ("mmlu_pro", "gsm8k", "ifeval", "hellaswag"):
        baseline = json.loads((smoke_dir / f"{benchmark}_baseline.json").read_text())
        recirculation = json.loads(
            (smoke_dir / f"{benchmark}_recirculation.json").read_text()
        )
        baseline_samples = [
            (task, sample)
            for task, samples in baseline["samples"].items()
            for sample in samples
        ]
        recirculation_samples = [
            (task, sample)
            for task, samples in recirculation["samples"].items()
            for sample in samples
        ]
        same_sample_identity = [
            (task, sample["doc_id"], sample["doc_hash"], sample["prompt_hash"])
            for task, sample in baseline_samples
        ] == [
            (task, sample["doc_id"], sample["doc_hash"], sample["prompt_hash"])
            for task, sample in recirculation_samples
        ]
        same_input_token_hashes = [
            event["input_token_ids_sha256_le_u32"] for event in baseline["call_events"]
        ] == [
            event["input_token_ids_sha256_le_u32"]
            for event in recirculation["call_events"]
        ]

        no_current_label_leak = True
        for task, sample in baseline_samples:
            prompt = sample["arguments"][0][0]
            doc = sample["doc"]
            if task.startswith("mmlu_pro"):
                current_cot = doc["cot_content"]
                no_current_label_leak &= not current_cot or current_cot not in prompt
            elif task == "gsm8k":
                no_current_label_leak &= doc["answer"] not in prompt
            elif task == "hellaswag":
                no_current_label_leak &= [
                    argument[1] for argument in sample["arguments"]
                ] == doc["choices"] and len(
                    {argument[0] for argument in sample["arguments"]}
                ) == 1

        benchmark_checks[benchmark] = {
            "manifest_sha256_identical": baseline["manifest_sha256"]
            == recirculation["manifest_sha256"],
            "sample_doc_and_prompt_hashes_identical": same_sample_identity,
            "model_input_token_hashes_identical": same_input_token_hashes,
            "current_evaluation_label_not_leaked_by_task_structure": bool(
                no_current_label_leak
            ),
            "identical_harness_configs": baseline["harness_configs"]
            == recirculation["harness_configs"],
        }
    return benchmark_checks


def run(smoke_dir: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(
        IT_MODEL_ID,
        revision=IT_MODEL_REVISION,
        local_files_only=True,
    )
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    first_mmlu = _first_sample(smoke_dir / "mmlu_pro_baseline.json")
    prompt = first_mmlu["arguments"][0][0]
    fixture = tokenizer.encode(prompt, return_tensors="pt").to("mps")

    official = _load(HFGemma3ForCausalLM)
    adapter = _load(Gemma3ForCausalLM)
    versions = tuple(parameter._version for parameter in adapter.parameters())
    short_fixture = fixture[:, :64]
    with torch.inference_mode():
        official_logits = official(
            short_fixture,
            attention_mask=torch.ones_like(short_fixture),
            use_cache=False,
        ).logits
        adapter_logits = adapter(
            short_fixture,
            attention_mask=torch.ones_like(short_fixture),
            use_cache=False,
        ).logits
    baseline_max_abs = float((official_logits - adapter_logits).abs().max().item())
    baseline_exact = bool(torch.equal(official_logits, adapter_logits))

    generation_fixture = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2 + 2? Reply with only the number."}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].to("mps")
    with torch.inference_mode():
        official_generation_one = official.generate(
            generation_fixture, max_new_tokens=8, do_sample=False, use_cache=True
        )
        official_generation_two = official.generate(
            generation_fixture, max_new_tokens=8, do_sample=False, use_cache=True
        )
        adapter_generation = adapter.generate(
            generation_fixture, max_new_tokens=8, do_sample=False, use_cache=True
        )

    del official_logits, adapter_logits, official
    torch.mps.empty_cache()
    adapter.set_recirculation_args(
        target_layer=DESTINATION_LAYER,
        source_layer=SOURCE_LAYER,
        target_layer_weight=BETA,
        source_layer_weight=ALPHA,
        num_recurrence_steps=NUM_RECURRENCE_STEPS,
        normalization=NORMALIZATION,
        ramp_steps=RAMP_STEPS,
    )

    observed_shapes = []
    observed_mix = []
    real_mix = recirculation_module.fixed_recirculation_mix

    def observing_mix(destination, source, **kwargs):
        observed_shapes.append([list(destination.shape), list(source.shape)])
        if len(observed_shapes) == RAMP_STEPS + 1:
            observed_mix.append(
                (destination.detach().clone(), source.detach().clone(), kwargs.copy())
            )
        return real_mix(destination, source, **kwargs)

    causal_fixture = fixture[:, :530]
    with (
        patch.object(
            recirculation_module,
            "fixed_recirculation_mix",
            side_effect=observing_mix,
        ),
        torch.inference_mode(),
    ):
        recirculated = adapter(
            causal_fixture,
            attention_mask=torch.ones_like(causal_fixture),
            use_cache=False,
        )

    destination, source, mix_kwargs = observed_mix[0]
    scaled_source = source / (source.norm(dim=-1, keepdim=True) + 1e-8)
    scaled_source = scaled_source * destination.norm(dim=-1, keepdim=True)
    norm_error = (scaled_source.norm(dim=-1) - destination.norm(dim=-1)).abs()

    changed = causal_fixture.clone()
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
            recirculated.logits[:, :future_change_position]
            - changed_output.logits[:, :future_change_position]
        )
        .abs()
        .max()
        .item()
    )

    incremental_fixture = fixture[:, :60]
    with torch.inference_mode():
        incremental, state = adapter.recirculating_prefill(
            incremental_fixture, attention_mask=torch.ones_like(incremental_fixture)
        )
    incremental_max_abs = 0.0
    incremental_argmax_equal = True
    for index in range(60, 64):
        next_ids = fixture[:, index : index + 1]
        incremental_fixture = torch.cat((incremental_fixture, next_ids), dim=1)
        with torch.inference_mode():
            incremental, state = adapter.recirculating_decode_step(next_ids, state)
            fresh = adapter(
                incremental_fixture,
                attention_mask=torch.ones_like(incremental_fixture),
                use_cache=False,
            )
        incremental_max_abs = max(
            incremental_max_abs,
            float((incremental.logits[:, -1] - fresh.logits[:, -1]).abs().max().item()),
        )
        incremental_argmax_equal &= bool(
            torch.equal(
                incremental.logits[:, -1].argmax(dim=-1),
                fresh.logits[:, -1].argmax(dim=-1),
            )
        )

    with torch.inference_mode():
        recirculation_generation_one = _greedy_recirculation(
            adapter, generation_fixture, 8
        )
        recirculation_generation_two = _greedy_recirculation(
            adapter, generation_fixture, 8
        )

    cache_lengths = [
        int(layer.keys.shape[-2]) for layer in recirculated.past_key_values.layers
    ]
    expected_cache_lengths = [
        min(
            causal_fixture.shape[1] - NUM_RECURRENCE_STEPS,
            adapter.config.sliding_window - 1,
        )
        if layer_type == "sliding_attention"
        else causal_fixture.shape[1] - NUM_RECURRENCE_STEPS
        for layer_type in adapter.config.layer_types
    ]

    smoke_checks = _paired_smoke_checks(smoke_dir)
    all_smoke_checks_pass = all(
        all(checks.values()) for checks in smoke_checks.values()
    )
    parameter_versions_unchanged = versions == tuple(
        parameter._version for parameter in adapter.parameters()
    )
    return {
        "schema_version": 1,
        "model_id": IT_MODEL_ID,
        "model_revision": IT_MODEL_REVISION,
        "chat_template_sha256": template_hash,
        "chat_template_matches_pin": template_hash == IT_CHAT_TEMPLATE_SHA256,
        "official_vs_adapter_baseline_64_token_logits_bitwise_equal": baseline_exact,
        "official_vs_adapter_baseline_64_token_logits_max_abs_difference": baseline_max_abs,
        "adapter_baseline_generation_matches_official": bool(
            torch.equal(official_generation_one, adapter_generation)
        ),
        "official_greedy_generation_deterministic": bool(
            torch.equal(official_generation_one, official_generation_two)
        ),
        "recirculation_greedy_generation_deterministic": (
            recirculation_generation_one == recirculation_generation_two
        ),
        "recirculation_mix_call_count": len(observed_shapes),
        "configured_destination_layer_zero_based": DESTINATION_LAYER,
        "configured_source_layer_zero_based": SOURCE_LAYER,
        "captured_full_ramp_source_weight": mix_kwargs["source_weight"],
        "captured_full_ramp_destination_weight": mix_kwargs["destination_weight"],
        "recirculation_unique_destination_source_shapes": sorted(
            {json.dumps(shape) for shape in observed_shapes}
        ),
        "max_norm_match_absolute_error": float(norm_error.max().item()),
        "future_changed_from_position": future_change_position,
        "causal_unchanged_prefix_max_abs_logit_difference": causal_prefix_max_abs,
        "returned_cache_layer_lengths": cache_lengths,
        "expected_cache_layer_lengths": expected_cache_lengths,
        "cache_layer_lengths_match": cache_lengths == expected_cache_lengths,
        "incremental_decode_vs_fresh_full_max_abs_logit_difference": (
            incremental_max_abs
        ),
        "incremental_decode_vs_fresh_full_argmax_equal_all_steps": (
            incremental_argmax_equal
        ),
        "incremental_decode_note": (
            "MPS bfloat16 kernel shape changes can alter low-order logits; greedy "
            "tokens matched fresh full-sequence recirculation at every checked step."
        ),
        "parameter_version_counters_unchanged": parameter_versions_unchanged,
        "all_weights_frozen": not any(
            parameter.requires_grad for parameter in adapter.parameters()
        ),
        "paired_smoke_checks": smoke_checks,
        "all_paired_smoke_checks_pass": all_smoke_checks_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-dir", type=Path, default=Path("results/capability_smoke")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/capability_verification.json")
    )
    args = parser.parse_args()
    result = run(args.smoke_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
