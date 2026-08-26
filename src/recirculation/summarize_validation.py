"""Summarize the locked unseen-book confirmatory experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .constants import (
    ALPHA,
    BETA,
    CONFIRMATORY_SELECTION_ID,
    DESTINATION_LAYER,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)
from .data import assert_manifests_disjoint, load_manifest

TIE_MEAN_NLL_TOLERANCE = 1e-6
EXPECTED_BOOKS = 8
EXPECTED_WINDOWS = 40
EXPECTED_TOKENS = 40_920


def perplexity(nll_sum: float, predicted_tokens: int) -> float:
    return math.exp(nll_sum / predicted_tokens)


def percent_reduction(baseline_ppl: float, recirculation_ppl: float) -> float:
    return (baseline_ppl - recirculation_ppl) / baseline_ppl * 100


def paired_direction(
    baseline_nll: float,
    recirculation_nll: float,
    predicted_tokens: int,
) -> str:
    mean_delta = (baseline_nll - recirculation_nll) / predicted_tokens
    if abs(mean_delta) <= TIE_MEAN_NLL_TOLERANCE:
        return "tied"
    return "improved" if mean_delta > 0 else "worsened"


def validate_conditions(
    baseline: dict, recirculation: dict, manifest: dict
) -> list[str]:
    if (
        baseline.get("condition") != "baseline"
        or baseline.get("recirculation") is not None
    ):
        raise ValueError("baseline input is not the ordinary condition")
    expected_recirculation = {
        "destination_layer_zero_based": DESTINATION_LAYER,
        "source_layer_zero_based": SOURCE_LAYER,
        "alpha": ALPHA,
        "beta": BETA,
        "num_recurrence_steps": NUM_RECURRENCE_STEPS,
        "normalization": NORMALIZATION,
        "ramp_steps": RAMP_STEPS,
    }
    if recirculation.get("condition") != "recirculation":
        raise ValueError("recirculation input is mislabeled")
    if recirculation.get("recirculation") != expected_recirculation:
        raise ValueError(
            "recirculation mechanism differs from the locked configuration"
        )

    exact_fields = [
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
        "manifest_path",
        "manifest_total_predicted_tokens",
        "evaluated_predicted_tokens",
    ]
    for field in exact_fields:
        if baseline.get(field) != recirculation.get(field):
            raise ValueError(f"conditions differ in {field}")
    if (
        baseline.get("weights_frozen") is not True
        or recirculation.get("weights_frozen") is not True
    ):
        raise ValueError("a condition did not report frozen weights")
    if baseline.get("parameter_version_counters_unchanged") is not True:
        raise ValueError("baseline parameter versions changed")
    if recirculation.get("parameter_version_counters_unchanged") is not True:
        raise ValueError("recirculation parameter versions changed")
    if baseline.get("evaluated_predicted_tokens") != EXPECTED_TOKENS:
        raise ValueError("confirmatory condition did not score exactly 40,920 targets")

    expected_identity = [
        (
            window["document_index"],
            window["window_index_within_document"],
            window["token_ids_sha256_le_u32"],
        )
        for window in manifest["windows"]
    ]
    for name, result in (("baseline", baseline), ("recirculation", recirculation)):
        actual_identity = [
            (
                window["document_index"],
                window["window_index_within_document"],
                window["token_ids_sha256_le_u32"],
            )
            for window in result.get("windows", [])
        ]
        if actual_identity != expected_identity:
            raise ValueError(f"{name} windows differ from the locked manifest")
        if any(window.get("predicted_tokens") != 1023 for window in result["windows"]):
            raise ValueError(f"{name} has an incorrect per-window target count")
    return exact_fields


def classify(
    aggregate_reduction: float, books_improved: int, windows_improved: int
) -> str:
    if aggregate_reduction > 0 and books_improved >= 5 and windows_improved >= 21:
        return "CONFIRMED"
    if aggregate_reduction <= 0 and books_improved <= 4 and windows_improved <= 20:
        return "NOT CONFIRMED"
    return "MIXED"


def summarize_validation(
    baseline_path: Path,
    recirculation_path: Path,
    manifest_path: Path,
    exploratory_manifest_path: Path,
    exploratory_comparison_path: Path,
) -> dict:
    manifest = load_manifest(manifest_path)
    if manifest.get("selection_id") != CONFIRMATORY_SELECTION_ID:
        raise ValueError("manifest is not the locked confirmatory selection")
    if len(manifest["windows"]) != EXPECTED_WINDOWS:
        raise ValueError("confirmatory manifest must contain 40 windows")
    assert_manifests_disjoint(exploratory_manifest_path, manifest_path)

    baseline = json.loads(baseline_path.read_text())
    recirculation = json.loads(recirculation_path.read_text())
    exact_fields = validate_conditions(baseline, recirculation, manifest)
    exploratory = json.loads(exploratory_comparison_path.read_text())

    baseline_by_identity = {
        (window["document_index"], window["window_index_within_document"]): window
        for window in baseline["windows"]
    }
    recirculation_by_identity = {
        (window["document_index"], window["window_index_within_document"]): window
        for window in recirculation["windows"]
    }

    per_window = []
    grouped: dict[int, list[dict]] = defaultdict(list)
    for manifest_window in manifest["windows"]:
        identity = (
            manifest_window["document_index"],
            manifest_window["window_index_within_document"],
        )
        ordinary = baseline_by_identity[identity]
        recurrent = recirculation_by_identity[identity]
        tokens = ordinary["predicted_tokens"]
        baseline_ppl = perplexity(ordinary["nll_sum"], tokens)
        recurrent_ppl = perplexity(recurrent["nll_sum"], tokens)
        row = {
            "document_index": identity[0],
            "short_book_title": manifest_window["short_book_title"],
            "window_index_within_document": identity[1],
            "token_ids_sha256_le_u32": manifest_window["token_ids_sha256_le_u32"],
            "predicted_tokens": tokens,
            "baseline_nll_sum": ordinary["nll_sum"],
            "recirculation_nll_sum": recurrent["nll_sum"],
            "baseline_perplexity": baseline_ppl,
            "recirculation_perplexity": recurrent_ppl,
            "percent_perplexity_reduction": percent_reduction(
                baseline_ppl, recurrent_ppl
            ),
            "paired_direction": paired_direction(
                ordinary["nll_sum"], recurrent["nll_sum"], tokens
            ),
        }
        per_window.append(row)
        grouped[identity[0]].append(row)

    per_book = []
    for document_index in sorted(grouped):
        rows = grouped[document_index]
        tokens = sum(row["predicted_tokens"] for row in rows)
        baseline_nll = sum(row["baseline_nll_sum"] for row in rows)
        recurrent_nll = sum(row["recirculation_nll_sum"] for row in rows)
        baseline_ppl = perplexity(baseline_nll, tokens)
        recurrent_ppl = perplexity(recurrent_nll, tokens)
        per_book.append(
            {
                "document_index": document_index,
                "short_book_title": rows[0]["short_book_title"],
                "windows": len(rows),
                "predicted_tokens": tokens,
                "baseline_nll_sum": baseline_nll,
                "recirculation_nll_sum": recurrent_nll,
                "baseline_perplexity": baseline_ppl,
                "recirculation_perplexity": recurrent_ppl,
                "percent_perplexity_reduction": percent_reduction(
                    baseline_ppl, recurrent_ppl
                ),
                "paired_direction": paired_direction(
                    baseline_nll, recurrent_nll, tokens
                ),
                "improving_windows": sum(
                    row["paired_direction"] == "improved" for row in rows
                ),
                "worsening_windows": sum(
                    row["paired_direction"] == "worsened" for row in rows
                ),
                "tied_windows": sum(row["paired_direction"] == "tied" for row in rows),
            }
        )

    baseline_ppl = baseline["perplexity"]
    recurrent_ppl = recirculation["perplexity"]
    aggregate_reduction = percent_reduction(baseline_ppl, recurrent_ppl)
    windows_improved = sum(row["paired_direction"] == "improved" for row in per_window)
    windows_worsened = sum(row["paired_direction"] == "worsened" for row in per_window)
    windows_tied = sum(row["paired_direction"] == "tied" for row in per_window)
    books_improved = sum(row["paired_direction"] == "improved" for row in per_book)
    books_worsened = sum(row["paired_direction"] == "worsened" for row in per_book)
    books_tied = sum(row["paired_direction"] == "tied" for row in per_book)

    # Resample whole books, keeping each five-window cluster intact. This is a
    # descriptive stability check for these eight sequential records, not a
    # confidence interval for a random sample from all books.
    rng = random.Random(260817982)
    bootstrap = []
    for _ in range(20_000):
        sample = [rng.randrange(EXPECTED_BOOKS) for _ in range(EXPECTED_BOOKS)]
        ordinary_nll = sum(per_book[index]["baseline_nll_sum"] for index in sample)
        recurrent_nll = sum(
            per_book[index]["recirculation_nll_sum"] for index in sample
        )
        tokens = sum(per_book[index]["predicted_tokens"] for index in sample)
        bootstrap.append(
            percent_reduction(
                perplexity(ordinary_nll, tokens), perplexity(recurrent_nll, tokens)
            )
        )
    bootstrap.sort()

    positive_book_nll_gains = sorted(
        (
            max(row["baseline_nll_sum"] - row["recirculation_nll_sum"], 0.0)
            for row in per_book
        ),
        reverse=True,
    )
    total_positive_gain = sum(positive_book_nll_gains)
    top_two_share = (
        sum(positive_book_nll_gains[:2]) / total_positive_gain
        if total_positive_gain > 0
        else None
    )
    distribution = (
        "reasonably_consistent"
        if books_improved >= 5 and top_two_share is not None and top_two_share <= 0.75
        else "highly_concentrated_or_mixed"
    )
    status = classify(aggregate_reduction, books_improved, windows_improved)

    return {
        "schema_version": 1,
        "experiment_id": CONFIRMATORY_SELECTION_ID,
        "classification": status,
        "classification_rule_locked_before_evaluation": (
            "CONFIRMED requires lower aggregate perplexity, at least 5 of 8 books "
            "improved, and at least 21 of 40 windows improved; NOT CONFIRMED "
            "requires no aggregate improvement and no majority at either level; "
            "all other outcomes are MIXED"
        ),
        "tie_rule": ("absolute paired mean-NLL difference <= 1e-6 per predicted token"),
        "aggregate": {
            "baseline_perplexity": baseline_ppl,
            "recirculation_perplexity": recurrent_ppl,
            "absolute_perplexity_reduction": baseline_ppl - recurrent_ppl,
            "percent_perplexity_reduction": aggregate_reduction,
            "evaluated_predicted_tokens": baseline["evaluated_predicted_tokens"],
            "windows": len(per_window),
            "books": len(per_book),
            "baseline_evaluation_seconds": baseline["evaluation_seconds"],
            "recirculation_evaluation_seconds": recirculation["evaluation_seconds"],
            "runtime_ratio": recirculation["evaluation_seconds"]
            / baseline["evaluation_seconds"],
        },
        "consistency": {
            "books_improved": books_improved,
            "books_worsened": books_worsened,
            "books_tied": books_tied,
            "windows_improved": windows_improved,
            "windows_worsened": windows_worsened,
            "windows_tied": windows_tied,
            "effect_distribution": distribution,
            "top_two_books_share_of_positive_nll_gain": top_two_share,
        },
        "book_block_bootstrap_percent_reduction_95pct_interval_descriptive": [
            bootstrap[500],
            bootstrap[19_499],
        ],
        "bootstrap_interpretation": (
            "Whole books were resampled with their five windows intact. Because the "
            "eight records are sequential rather than a random book sample, this is "
            "a descriptive stability interval, not population-level confidence."
        ),
        "comparisons": {
            "exploratory_percent_reduction": exploratory[
                "percent_perplexity_reduction"
            ],
            "difference_from_exploratory_percentage_points": aggregate_reduction
            - exploratory["percent_perplexity_reduction"],
            "paper_pg19_percent_reduction_reference": 14.41,
            "difference_from_paper_percentage_points": aggregate_reduction - 14.41,
        },
        "verification": {
            "identical_condition_fields": exact_fields,
            "identical_ordered_window_hashes": True,
            "locked_recirculation_configuration": True,
            "weights_frozen_and_version_counters_unchanged": True,
            "exploratory_document_overlap": False,
            "exploratory_token_hash_overlap": False,
        },
        "per_book": per_book,
        "per_window": per_window,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--recirculation", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--exploratory-manifest",
        type=Path,
        default=Path("experiments/pg19_windows.json"),
    )
    parser.add_argument(
        "--exploratory-comparison",
        type=Path,
        default=Path("results/comparison.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_validation(
        args.baseline,
        args.recirculation,
        args.manifest,
        args.exploratory_manifest,
        args.exploratory_comparison,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
