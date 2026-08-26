import json
from pathlib import Path

import pytest

from recirculation.summarize_capability import (
    _classify_metric,
    _mcnemar_exact,
    _summarize_metric,
    _validate_pair,
)


def _result(condition: str, values: list[float]) -> dict:
    samples = []
    events = []
    for doc_id, value in enumerate(values):
        samples.append(
            {
                "doc_id": doc_id,
                "filter": "none",
                "metrics": ["acc"],
                "acc": value,
                "doc_hash": f"doc-{doc_id}",
                "prompt_hash": f"prompt-{doc_id}",
                "target_hash": f"target-{doc_id}",
                "arguments": [f"argument-{doc_id}"],
                "target": doc_id,
                "filtered_resps": [str(value)],
                "resps": [[str(value)]],
            }
        )
        events.append(
            {
                "operation": "loglikelihood",
                "input_tokens": 10,
                "input_token_ids_sha256_le_u32": f"tokens-{doc_id}",
                "runtime_seconds": 1.0,
            }
        )
    return {
        "condition": condition,
        "benchmark": "fixture",
        "phase": "locked",
        "manifest_sha256": "filled-by-test",
        "harness_revision": "harness",
        "model_info": {"model": "fixture"},
        "python_version": "python",
        "platform": "platform",
        "torch_version": "torch",
        "transformers_version": "transformers",
        "harness_configs": {},
        "harness_versions": {},
        "harness_n_shot": {},
        "harness_results": {
            "fixture_task": {
                "sample_len": len(values),
                "acc,none": sum(values) / len(values),
            }
        },
        "samples": {"fixture_task": samples},
        "call_events": events,
        "verification": {
            "all_weights_frozen": True,
            "parameter_version_counters_unchanged": True,
        },
    }


def test_paired_metric_counts_flips_and_delta() -> None:
    baseline = _result("baseline", [0, 1, 0, 1])
    recurrent = _result("recirculation", [1, 0, 1, 1])
    result = _summarize_metric(baseline, recurrent, "acc,none")
    assert result["baseline_score"] == 0.5
    assert result["recirculation_score"] == 0.75
    assert result["paired_prompt_or_document_clusters"] == 4
    assert result["paired_scored_units"] == 4
    assert result["absolute_delta_percentage_points"] == 25.0
    assert result["favorable_flips"] == 2
    assert result["unfavorable_flips"] == 1
    assert result["classification"] == "IMPROVED"


def test_exact_mcnemar_handles_balanced_and_one_sided_flips() -> None:
    assert _mcnemar_exact(0, 0) == 1.0
    assert _mcnemar_exact(1, 1) == 1.0
    assert _mcnemar_exact(6, 0) == pytest.approx(0.03125)


def test_classification_preserves_tied_but_changed_as_inconclusive() -> None:
    assert _classify_metric(0.0, 0, 0) == "UNCHANGED"
    assert _classify_metric(0.0, 2, 2) == "MIXED / INCONCLUSIVE"


def test_pair_validation_rejects_prompt_mismatch(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "phase": "locked",
        "harness": {"revision": "harness"},
        "benchmarks": {"fixture": {"sample_count": 2}},
    }
    manifest_path.write_text(json.dumps(manifest))
    baseline = _result("baseline", [0, 1])
    recurrent = _result("recirculation", [1, 1])
    digest = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    baseline["manifest_sha256"] = digest
    recurrent["manifest_sha256"] = digest
    recurrent["samples"]["fixture_task"][0]["prompt_hash"] = "different"
    with pytest.raises(ValueError, match="prompt_hash"):
        _validate_pair("fixture", baseline, recurrent, manifest, manifest_path)
