import hashlib
import json
from pathlib import Path

import torch

from recirculation.capability_model import token_ids_sha256
from recirculation.run_capability import _json_safe

ROOT = Path(__file__).parents[1]
SMOKE = ROOT / "results" / "capability_smoke"


def test_capability_token_hash_has_explicit_little_endian_u32_encoding() -> None:
    expected = hashlib.sha256(bytes.fromhex("020000000100000000010000")).hexdigest()
    assert token_ids_sha256([2, 1, 256]) == expected
    assert token_ids_sha256(torch.tensor([[2, 1, 256]])) == expected


def test_harness_config_serialization_removes_process_specific_addresses() -> None:
    value = {"callable": "functools.partial(<function f at 0x123ABC>, x=1)"}
    assert _json_safe(value) == {
        "callable": "functools.partial(<function f at 0xADDR>, x=1)"
    }


def test_smoke_lanes_use_identical_inputs_and_are_non_evidentiary() -> None:
    for benchmark in ("mmlu_pro", "gsm8k", "ifeval", "hellaswag"):
        baseline = json.loads((SMOKE / f"{benchmark}_baseline.json").read_text())
        recirculation = json.loads(
            (SMOKE / f"{benchmark}_recirculation.json").read_text()
        )
        assert baseline["phase"] == "plumbing_smoke_non_evidentiary"
        assert recirculation["phase"] == "plumbing_smoke_non_evidentiary"
        assert baseline["manifest_sha256"] == recirculation["manifest_sha256"]
        assert baseline["harness_configs"] == recirculation["harness_configs"]
        assert [
            event["input_token_ids_sha256_le_u32"] for event in baseline["call_events"]
        ] == [
            event["input_token_ids_sha256_le_u32"]
            for event in recirculation["call_events"]
        ]


def test_real_capability_verification_passes_every_required_check() -> None:
    verification = json.loads(
        (ROOT / "results" / "capability_verification.json").read_text()
    )
    required = (
        "chat_template_matches_pin",
        "official_vs_adapter_baseline_64_token_logits_bitwise_equal",
        "adapter_baseline_generation_matches_official",
        "official_greedy_generation_deterministic",
        "recirculation_greedy_generation_deterministic",
        "cache_layer_lengths_match",
        "incremental_decode_vs_fresh_full_argmax_equal_all_steps",
        "parameter_version_counters_unchanged",
        "all_weights_frozen",
        "all_paired_smoke_checks_pass",
    )
    assert all(verification[key] for key in required)
    assert verification["causal_unchanged_prefix_max_abs_logit_difference"] == 0
    assert verification["configured_destination_layer_zero_based"] == 4
    assert verification["configured_source_layer_zero_based"] == 11
