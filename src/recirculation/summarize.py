"""Validate matched result files and create the final comparison summary."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from .constants import (
    ALPHA,
    BETA,
    DESTINATION_LAYER,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)


def summarize(baseline_path: Path, recirculation_path: Path) -> dict:
    baseline = json.loads(baseline_path.read_text())
    recirculation = json.loads(recirculation_path.read_text())
    if baseline.get("condition") != "baseline":
        raise ValueError("baseline input is not labeled as the baseline condition")
    if recirculation.get("condition") != "recirculation":
        raise ValueError(
            "recirculation input is not labeled as the recirculation condition"
        )
    if baseline.get("recirculation") is not None:
        raise ValueError("baseline unexpectedly contains recirculation parameters")
    expected_recirculation = {
        "destination_layer_zero_based": DESTINATION_LAYER,
        "source_layer_zero_based": SOURCE_LAYER,
        "alpha": ALPHA,
        "beta": BETA,
        "num_recurrence_steps": NUM_RECURRENCE_STEPS,
        "normalization": NORMALIZATION,
        "ramp_steps": RAMP_STEPS,
    }
    if recirculation.get("recirculation") != expected_recirculation:
        raise ValueError("recirculation parameters differ from the locked experiment")
    exact_match_fields = [
        "model_id",
        "model_revision",
        "weights_sha256",
        "parameter_count",
        "dtype",
        "device",
        "attention_implementation",
        "torch_version",
        "transformers_version",
        "python_version",
        "dataset_id",
        "dataset_revision",
        "evaluated_predicted_tokens",
    ]
    for field in exact_match_fields:
        if baseline[field] != recirculation[field]:
            raise ValueError(f"conditions differ in {field}")
    baseline_hashes = [
        window["token_ids_sha256_le_u32"] for window in baseline["windows"]
    ]
    recirculation_hashes = [
        window["token_ids_sha256_le_u32"] for window in recirculation["windows"]
    ]
    if baseline_hashes != recirculation_hashes:
        raise ValueError("conditions evaluated different token windows")

    baseline_ppl = baseline["perplexity"]
    recirculation_ppl = recirculation["perplexity"]
    absolute = baseline_ppl - recirculation_ppl
    percent = absolute / baseline_ppl * 100

    # Descriptive paired-window bootstrap. Windows within a book are dependent,
    # so this is sensitivity evidence, not a population-confidence interval.
    rng = random.Random(260817981)
    count = len(baseline["windows"])
    tokens_per_window = baseline["windows"][0]["predicted_tokens"]
    bootstrap = []
    for _ in range(20_000):
        sample = [rng.randrange(count) for _ in range(count)]
        baseline_nll = sum(baseline["windows"][index]["nll_sum"] for index in sample)
        recirculation_nll = sum(
            recirculation["windows"][index]["nll_sum"] for index in sample
        )
        denominator = tokens_per_window * count
        baseline_sample_ppl = math.exp(baseline_nll / denominator)
        recirculation_sample_ppl = math.exp(recirculation_nll / denominator)
        bootstrap.append(
            (baseline_sample_ppl - recirculation_sample_ppl) / baseline_sample_ppl * 100
        )
    bootstrap.sort()
    bootstrap_interval = [bootstrap[500], bootstrap[19_499]]
    pass_threshold_met = percent > 0 and bootstrap_interval[0] > 0
    status = "PASS" if pass_threshold_met else "FAIL"
    conclusion = (
        "fixed recirculation measurably reduced perplexity on the locked local subset"
        if pass_threshold_met
        else "fixed recirculation did not measurably reduce perplexity on the locked local subset"
    )

    return {
        "baseline_perplexity": baseline_ppl,
        "recirculation_perplexity": recirculation_ppl,
        "absolute_perplexity_reduction": absolute,
        "percent_perplexity_reduction": percent,
        "paper_pg19_percent_reduction_reference": 14.41,
        "difference_from_paper_percentage_points": percent - 14.41,
        "evaluated_predicted_tokens": baseline["evaluated_predicted_tokens"],
        "windows": count,
        "windows_with_lower_nll": sum(
            recirculated["nll_sum"] < ordinary["nll_sum"]
            for ordinary, recirculated in zip(
                baseline["windows"], recirculation["windows"]
            )
        ),
        "paired_window_bootstrap_percent_reduction_95pct_interval_descriptive": [
            *bootstrap_interval,
        ],
        "baseline_evaluation_seconds": baseline["evaluation_seconds"],
        "recirculation_evaluation_seconds": recirculation["evaluation_seconds"],
        "runtime_ratio": recirculation["evaluation_seconds"]
        / baseline["evaluation_seconds"],
        "baseline_tokens_per_second": baseline["tokens_per_second"],
        "recirculation_tokens_per_second": recirculation["tokens_per_second"],
        "baseline_max_observed_mps_driver_allocated_bytes_after_sync": baseline[
            "max_observed_mps_driver_allocated_bytes_after_sync"
        ],
        "recirculation_max_observed_mps_driver_allocated_bytes_after_sync": recirculation[
            "max_observed_mps_driver_allocated_bytes_after_sync"
        ],
        "observed_mps_driver_allocated_delta_bytes": recirculation[
            "max_observed_mps_driver_allocated_bytes_after_sync"
        ]
        - baseline["max_observed_mps_driver_allocated_bytes_after_sync"],
        "identical_condition_fields_verified": exact_match_fields,
        "identical_ordered_window_hashes_verified": True,
        "pass_rule": "positive aggregate reduction and positive descriptive paired-window bootstrap lower bound",
        "conclusion": conclusion,
        "reproduction_status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.json"))
    parser.add_argument(
        "--recirculation", type=Path, default=Path("results/recirculation.json")
    )
    parser.add_argument("--output", type=Path, default=Path("results/comparison.json"))
    args = parser.parse_args()
    result = summarize(args.baseline, args.recirculation)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
