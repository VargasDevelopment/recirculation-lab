import json
import math
from pathlib import Path

import pytest

from recirculation.constants import (
    ALPHA,
    BETA,
    DESTINATION_LAYER,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)
from recirculation.data import load_manifest
from recirculation.summarize_validation import classify, summarize_validation


MANIFEST = Path("experiments/pg19_validation_books_2_9.json")
EXPLORATORY_MANIFEST = Path("experiments/pg19_windows.json")
EXPLORATORY_COMPARISON = Path("results/comparison.json")


def condition(label: str, mean_nll: float) -> dict:
    manifest = load_manifest(MANIFEST)
    windows = [
        {
            "document_index": window["document_index"],
            "window_index_within_document": window["window_index_within_document"],
            "token_ids_sha256_le_u32": window["token_ids_sha256_le_u32"],
            "predicted_tokens": 1023,
            "nll_sum": mean_nll * 1023,
        }
        for window in manifest["windows"]
    ]
    total_nll = sum(window["nll_sum"] for window in windows)
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
        "weights_frozen": True,
        "parameter_version_counters_unchanged": True,
        "dtype": "bfloat16",
        "device": "mps",
        "attention_implementation": "eager",
        "torch_version": "torch",
        "transformers_version": "transformers",
        "python_version": "python",
        "dataset_id": "dataset",
        "dataset_revision": "dataset-revision",
        "manifest_path": str(MANIFEST),
        "manifest_total_predicted_tokens": 40_920,
        "evaluated_predicted_tokens": 40_920,
        "total_nll": total_nll,
        "perplexity": math.exp(total_nll / 40_920),
        "evaluation_seconds": 1.0,
        "windows": windows,
    }


def write(tmp_path: Path, name: str, value: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def summarize(tmp_path: Path, baseline: dict, recurrent: dict) -> dict:
    return summarize_validation(
        write(tmp_path, "baseline.json", baseline),
        write(tmp_path, "recirculation.json", recurrent),
        MANIFEST,
        EXPLORATORY_MANIFEST,
        EXPLORATORY_COMPARISON,
    )


def test_uniform_positive_result_is_confirmed(tmp_path: Path) -> None:
    result = summarize(tmp_path, condition("baseline", 3.0), condition("recirculation", 2.9))
    assert result["classification"] == "CONFIRMED"
    assert result["consistency"]["books_improved"] == 8
    assert result["consistency"]["windows_improved"] == 40


def test_uniform_negative_result_is_not_confirmed(tmp_path: Path) -> None:
    result = summarize(tmp_path, condition("baseline", 3.0), condition("recirculation", 3.1))
    assert result["classification"] == "NOT CONFIRMED"
    assert result["aggregate"]["percent_perplexity_reduction"] < 0


def test_locked_mechanism_drift_is_rejected(tmp_path: Path) -> None:
    baseline = condition("baseline", 3.0)
    recurrent = condition("recirculation", 2.9)
    recurrent["recirculation"]["alpha"] = 0.2
    with pytest.raises(ValueError, match="locked configuration"):
        summarize(tmp_path, baseline, recurrent)


def test_classification_rule_has_a_mixed_middle() -> None:
    assert classify(5.0, books_improved=4, windows_improved=25) == "MIXED"
