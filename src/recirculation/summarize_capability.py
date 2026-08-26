"""Validate and summarize the locked paired capability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

PRIMARY_METRICS = {
    "mmlu_pro": "exact_match,custom-extract",
    "gsm8k": "exact_match,flexible-extract",
    "ifeval": "prompt_level_strict_acc,none",
    "hellaswag": "acc_norm,none",
}

METRICS = {
    "mmlu_pro": ["exact_match,custom-extract"],
    "gsm8k": ["exact_match,flexible-extract", "exact_match,strict-match"],
    "ifeval": [
        "prompt_level_strict_acc,none",
        "inst_level_strict_acc,none",
        "prompt_level_loose_acc,none",
        "inst_level_loose_acc,none",
    ],
    "hellaswag": ["acc_norm,none", "acc,none"],
}

BOOTSTRAP_SEED = 260817982
BOOTSTRAP_ITERATIONS = 20_000


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sample_key(task: str, sample: dict[str, Any]) -> tuple[str, int, str]:
    return task, int(sample["doc_id"]), str(sample.get("filter", "none"))


def _samples(result: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    flattened = {}
    for task, task_samples in result["samples"].items():
        for sample in task_samples:
            key = _sample_key(task, sample)
            if key in flattened:
                raise ValueError(f"duplicate sample identity: {key}")
            flattened[key] = sample
    return flattened


def _metric_parts(metric: str) -> tuple[str, str]:
    name, _, filter_name = metric.partition(",")
    return name, filter_name or "none"


def _metric_rows(
    result: dict[str, Any], metric: str
) -> dict[tuple[str, int], tuple[dict[str, Any], float | list[bool]]]:
    metric_name, filter_name = _metric_parts(metric)
    rows = {}
    for (task, doc_id, sample_filter), sample in _samples(result).items():
        if sample_filter != filter_name or metric_name not in sample:
            continue
        value = sample[metric_name]
        rows[(task, doc_id)] = (
            sample,
            [bool(item) for item in value] if isinstance(value, list) else float(value),
        )
    if not rows:
        raise ValueError(f"no sample rows found for metric {metric}")
    return rows


def _mcnemar_exact(favorable: int, unfavorable: int) -> float:
    discordant = favorable + unfavorable
    if discordant == 0:
        return 1.0
    smaller = min(favorable, unfavorable)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _values(value: float | list[bool]) -> list[float]:
    return [float(item) for item in value] if isinstance(value, list) else [value]


def _score(values: list[float | list[bool]]) -> float:
    flattened = [item for value in values for item in _values(value)]
    return sum(flattened) / len(flattened)


def _bootstrap_delta(
    pairs: list[tuple[float | list[bool], float | list[bool]]],
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> dict[str, Any]:
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    count = len(pairs)
    for _ in range(iterations):
        sample = [pairs[rng.randrange(count)] for _ in range(count)]
        deltas.append(
            100.0
            * (
                _score([pair[1] for pair in sample])
                - _score([pair[0] for pair in sample])
            )
        )
    deltas.sort()
    return {
        "unit": "prompt/document (instruction vectors stay clustered by prompt)",
        "iterations": iterations,
        "seed": BOOTSTRAP_SEED,
        "delta_percentage_points_95_percentile_interval": [
            deltas[int(0.025 * iterations)],
            deltas[int(0.975 * iterations) - 1],
        ],
        "interpretation": (
            "Descriptive stability interval for this locked deterministic subset; "
            "not a population confidence interval for the full benchmark."
        ),
    }


def _classify_metric(delta_pp: float, favorable: int, unfavorable: int) -> str:
    if delta_pp > 0 and favorable > unfavorable:
        return "IMPROVED"
    if delta_pp < 0 and unfavorable > favorable:
        return "REGRESSED"
    if delta_pp == 0 and favorable == 0 and unfavorable == 0:
        return "UNCHANGED"
    return "MIXED / INCONCLUSIVE"


def _validate_pair(
    benchmark: str,
    baseline: dict[str, Any],
    recirculation: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    if baseline["condition"] != "baseline":
        raise ValueError(f"{benchmark}: ordinary lane is mislabeled")
    if recirculation["condition"] != "recirculation":
        raise ValueError(f"{benchmark}: recirculation lane is mislabeled")
    expected_manifest_hash = _sha256_file(manifest_path)
    expected = {
        "benchmark": benchmark,
        "phase": manifest["phase"],
        "manifest_sha256": expected_manifest_hash,
        "harness_revision": manifest["harness"]["revision"],
    }
    for lane_name, lane in (("baseline", baseline), ("recirculation", recirculation)):
        for field, value in expected.items():
            if lane.get(field) != value:
                raise ValueError(f"{benchmark} {lane_name}: mismatch in {field}")
        if not lane["verification"]["all_weights_frozen"]:
            raise ValueError(f"{benchmark} {lane_name}: weights were not frozen")
        if not lane["verification"]["parameter_version_counters_unchanged"]:
            raise ValueError(f"{benchmark} {lane_name}: parameters changed")

    equality_fields = [
        "model_info",
        "python_version",
        "platform",
        "torch_version",
        "transformers_version",
        "harness_revision",
        "harness_configs",
        "harness_versions",
        "harness_n_shot",
    ]
    for field in equality_fields:
        if baseline.get(field) != recirculation.get(field):
            raise ValueError(f"{benchmark}: lanes differ in {field}")

    baseline_samples = _samples(baseline)
    recirculation_samples = _samples(recirculation)
    if baseline_samples.keys() != recirculation_samples.keys():
        raise ValueError(f"{benchmark}: lane sample identities differ")
    identity_fields = ["doc_hash", "prompt_hash", "target_hash", "arguments", "target"]
    for key in baseline_samples:
        for field in identity_fields:
            if baseline_samples[key].get(field) != recirculation_samples[key].get(
                field
            ):
                raise ValueError(f"{benchmark} {key}: lanes differ in {field}")

    baseline_events = baseline["call_events"]
    recirculation_events = recirculation["call_events"]
    baseline_inputs = [
        (
            event["operation"],
            event["input_tokens"],
            event["input_token_ids_sha256_le_u32"],
        )
        for event in baseline_events
    ]
    recirculation_inputs = [
        (
            event["operation"],
            event["input_tokens"],
            event["input_token_ids_sha256_le_u32"],
        )
        for event in recirculation_events
    ]
    if baseline_inputs != recirculation_inputs:
        raise ValueError(f"{benchmark}: paired model-call token streams differ")

    expected_count = int(manifest["benchmarks"][benchmark]["sample_count"])
    unique_docs = {(task, doc_id) for task, doc_id, _ in baseline_samples}
    if len(unique_docs) != expected_count:
        raise ValueError(
            f"{benchmark}: expected {expected_count} documents, found {len(unique_docs)}"
        )
    return {
        "passed": True,
        "sample_count": expected_count,
        "sample_identities_equal": True,
        "prompt_hashes_equal": True,
        "rendered_arguments_equal": True,
        "model_call_token_hashes_equal": True,
        "model_and_software_settings_equal": True,
        "weights_frozen_and_unchanged": True,
    }


def _summarize_metric(
    baseline: dict[str, Any], recirculation: dict[str, Any], metric: str
) -> dict[str, Any]:
    ordinary = _metric_rows(baseline, metric)
    recurrent = _metric_rows(recirculation, metric)
    if ordinary.keys() != recurrent.keys():
        raise ValueError(f"paired rows differ for metric {metric}")
    pairs = [(ordinary[key][1], recurrent[key][1]) for key in ordinary]
    baseline_score = _score([pair[0] for pair in pairs])
    recirculation_score = _score([pair[1] for pair in pairs])
    paired_values = [
        (baseline_value, recirculation_value)
        for baseline, recirculation in pairs
        for baseline_value, recirculation_value in zip(
            _values(baseline), _values(recirculation), strict=True
        )
    ]
    favorable = sum(a < b for a, b in paired_values)
    unfavorable = sum(a > b for a, b in paired_values)
    tied = len(paired_values) - favorable - unfavorable
    delta_pp = 100.0 * (recirculation_score - baseline_score)
    return {
        "metric": metric,
        "paired_prompt_or_document_clusters": len(pairs),
        "paired_scored_units": len(paired_values),
        "baseline_score": baseline_score,
        "recirculation_score": recirculation_score,
        "absolute_delta_percentage_points": delta_pp,
        "relative_delta_percent": (
            None
            if baseline_score == 0
            else 100.0 * (recirculation_score - baseline_score) / baseline_score
        ),
        "favorable_flips": favorable,
        "unfavorable_flips": unfavorable,
        "tied_outcomes": tied,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(favorable, unfavorable),
        "paired_bootstrap": _bootstrap_delta(pairs),
        "classification": _classify_metric(delta_pp, favorable, unfavorable),
    }


def _changed_examples(
    benchmark: str,
    baseline: dict[str, Any],
    recirculation: dict[str, Any],
    metric: str,
) -> list[dict[str, Any]]:
    ordinary = _metric_rows(baseline, metric)
    recurrent = _metric_rows(recirculation, metric)
    changed = []
    for key in sorted(ordinary):
        baseline_sample, baseline_value = ordinary[key]
        recirculation_sample, recirculation_value = recurrent[key]
        if baseline_value == recirculation_value:
            continue
        changed.append(
            {
                "task": key[0],
                "doc_id": key[1],
                "doc_hash": baseline_sample["doc_hash"],
                "prompt_hash": baseline_sample["prompt_hash"],
                "target_hash": baseline_sample["target_hash"],
                "direction": (
                    "baseline_wrong_to_recirculation_right"
                    if baseline_value < recirculation_value
                    else "baseline_right_to_recirculation_wrong"
                ),
                "baseline_metric_value": baseline_value,
                "recirculation_metric_value": recirculation_value,
                "baseline_filtered_response": baseline_sample["filtered_resps"],
                "recirculation_filtered_response": recirculation_sample[
                    "filtered_resps"
                ],
                "baseline_raw_response": baseline_sample["resps"],
                "recirculation_raw_response": recirculation_sample["resps"],
            }
        )
    return changed


def _timing(result: dict[str, Any]) -> dict[str, Any]:
    events = result["call_events"]
    return {
        "model_load_seconds": result["model_load_seconds"],
        "evaluator_wall_seconds": result["evaluation_seconds"],
        "model_call_seconds": sum(event["runtime_seconds"] for event in events),
        "prefill_seconds": sum(event.get("prefill_seconds", 0.0) for event in events),
        "decode_seconds": sum(event.get("decode_seconds", 0.0) for event in events),
        "call_count": len(events),
        "mean_model_call_seconds": (
            sum(event["runtime_seconds"] for event in events) / len(events)
        ),
        "peak_process_rss_bytes": result["verification"]["process_peak_rss_bytes"],
        "max_observed_mps_driver_allocated_bytes": result["verification"][
            "max_observed_mps_driver_allocated_bytes_after_sync"
        ],
        "max_observed_mps_tensor_allocated_bytes": result["verification"][
            "max_observed_mps_tensor_allocated_bytes_after_sync"
        ],
    }


def summarize(manifest_path: Path, results_dir: Path) -> dict[str, Any]:
    manifest = _load(manifest_path)
    benchmark_summaries = {}
    all_verification_passed = True
    for benchmark, primary_metric in PRIMARY_METRICS.items():
        baseline = _load(results_dir / f"{benchmark}_baseline.json")
        recirculation = _load(results_dir / f"{benchmark}_recirculation.json")
        verification = _validate_pair(
            benchmark, baseline, recirculation, manifest, manifest_path
        )
        all_verification_passed &= verification["passed"]
        metrics = {
            metric: _summarize_metric(baseline, recirculation, metric)
            for metric in METRICS[benchmark]
        }
        benchmark_summaries[benchmark] = {
            "primary_metric": primary_metric,
            "classification": metrics[primary_metric]["classification"],
            "metrics": metrics,
            "changed_examples_primary_metric": _changed_examples(
                benchmark, baseline, recirculation, primary_metric
            ),
            "timing": {
                "baseline": _timing(baseline),
                "recirculation": _timing(recirculation),
            },
            "verification": verification,
        }

    classifications = [item["classification"] for item in benchmark_summaries.values()]
    improved = classifications.count("IMPROVED")
    regressed = classifications.count("REGRESSED")
    if improved >= 2 and regressed == 0:
        overall = "CAPABILITY SIGNAL"
    elif improved == 0:
        overall = "NO TRANSFER"
    else:
        overall = "MIXED"

    total_evaluator_seconds = sum(
        lane["evaluator_wall_seconds"]
        for benchmark in benchmark_summaries.values()
        for lane in benchmark["timing"].values()
    )
    return {
        "schema_version": 1,
        "experiment": "locked_gemma_3_1b_it_capability_transfer_v1",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "overall_classification": overall,
        "benchmark_classification_counts": {
            "improved": improved,
            "regressed": regressed,
            "unchanged": classifications.count("UNCHANGED"),
            "mixed_or_inconclusive": classifications.count("MIXED / INCONCLUSIVE"),
        },
        "verification": {
            "all_benchmark_pairs_passed": all_verification_passed,
            "real_model_preflight_artifact": "results/capability_verification.json",
        },
        "benchmarks": benchmark_summaries,
        "performance": {
            "sum_of_evaluator_wall_seconds_all_eight_lanes": total_evaluator_seconds,
            "sum_of_evaluator_wall_hours_all_eight_lanes": total_evaluator_seconds
            / 3600.0,
        },
        "statistical_scope": (
            "Paired tests and bootstraps describe the locked deterministic subsets. "
            "They do not turn these subsets into random samples of full benchmarks, "
            "and IFEval instructions remain clustered within prompts."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.manifest, args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
