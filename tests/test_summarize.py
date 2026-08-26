import json
from pathlib import Path

import pytest

from recirculation.constants import (
    ALPHA,
    ATTENTION_IMPLEMENTATION,
    BETA,
    DESTINATION_LAYER,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)
from recirculation.summarize import summarize


def condition(label: str, nll: float) -> dict:
    windows = [
        {"token_ids_sha256_le_u32": str(index), "predicted_tokens": 2, "nll_sum": nll}
        for index in range(10)
    ]
    return {
        "condition": label,
        "recirculation": None
        if label == "baseline"
        else {
            "destination_layer_zero_based": DESTINATION_LAYER,
            "source_layer_zero_based": SOURCE_LAYER,
            "alpha": ALPHA,
            "beta": BETA,
            "num_recurrence_steps": NUM_RECURRENCE_STEPS,
            "normalization": NORMALIZATION,
            "ramp_steps": RAMP_STEPS,
        },
        "model_id": "model",
        "model_revision": "revision",
        "weights_sha256": "weights",
        "parameter_count": 1,
        "dtype": "bfloat16",
        "device": "mps",
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "torch_version": "torch",
        "transformers_version": "transformers",
        "python_version": "python",
        "dataset_id": "dataset",
        "dataset_revision": "data-revision",
        "evaluated_predicted_tokens": 20,
        "perplexity": 1.0,
        "evaluation_seconds": 1.0,
        "tokens_per_second": 20.0,
        "max_observed_mps_driver_allocated_bytes_after_sync": 1,
        "windows": windows,
    }


def write(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def test_swapped_inputs_are_rejected(tmp_path: Path) -> None:
    baseline = write(tmp_path, "baseline.json", condition("baseline", 2.0))
    recirculation = write(
        tmp_path, "recirculation.json", condition("recirculation", 1.0)
    )
    with pytest.raises(ValueError, match="baseline input"):
        summarize(recirculation, baseline)


def test_negative_result_cannot_be_certified_as_pass(tmp_path: Path) -> None:
    baseline_value = condition("baseline", 1.0)
    recirculation_value = condition("recirculation", 2.0)
    baseline_value["perplexity"] = 2.0
    recirculation_value["perplexity"] = 3.0
    result = summarize(
        write(tmp_path, "baseline.json", baseline_value),
        write(tmp_path, "recirculation.json", recirculation_value),
    )
    assert result["reproduction_status"] == "FAIL"
    assert "did not" in result["conclusion"]
