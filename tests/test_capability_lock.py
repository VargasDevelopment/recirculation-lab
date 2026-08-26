import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _hashes(manifest: dict, benchmark: str) -> set[str]:
    return {
        sample["doc_sha256_canonical_json"]
        for task in manifest["benchmarks"][benchmark]["tasks"].values()
        for sample in task["selected_samples"]
    }


def test_locked_capability_manifest_is_disjoint_and_complete() -> None:
    smoke = json.loads(
        (ROOT / "experiments" / "capability_smoke_manifest.json").read_text()
    )
    locked = json.loads(
        (ROOT / "experiments" / "capability_locked_manifest.json").read_text()
    )
    assert locked["locked_before_substantive_scoring"] is True
    assert {
        benchmark: metadata["sample_count"]
        for benchmark, metadata in locked["benchmarks"].items()
    } == {"mmlu_pro": 42, "gsm8k": 50, "ifeval": 50, "hellaswag": 100}
    for benchmark in locked["benchmarks"]:
        assert not (_hashes(smoke, benchmark) & _hashes(locked, benchmark))


def test_locked_capability_mechanism_matches_confirmed_pt_configuration() -> None:
    locked = json.loads(
        (ROOT / "experiments" / "capability_locked_manifest.json").read_text()
    )
    assert locked["model"]["id"] == "google/gemma-3-1b-it"
    assert locked["model"]["frozen"] is True
    assert locked["recirculation"] == {
        "destination_layer_zero_based": 4,
        "source_layer_zero_based": 11,
        "alpha": 0.15,
        "beta": 0.85,
        "alpha_ramp_tokens": 10,
        "normalization": "norm_rep",
        "additional_recurrence_iterations": 1,
    }
