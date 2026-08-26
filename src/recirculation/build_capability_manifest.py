"""Resolve deterministic harness indices into an auditable locked manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from lm_eval.tasks import TaskManager

from .capability_constants import (
    BENCHMARKS,
    CAPABILITY_ATTENTION_IMPLEMENTATION,
    CAPABILITY_DEVICE,
    CAPABILITY_DTYPE,
    CAPABILITY_MAX_LENGTH,
    CAPABILITY_SEED,
    HARNESS_REPOSITORY,
    HARNESS_REVISION,
    IT_CHAT_TEMPLATE_SHA256,
    IT_MODEL_ID,
    IT_MODEL_REVISION,
    IT_TOKENIZER_REVISION,
    IT_WEIGHTS_SHA256,
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
from .harness_pins import pinned_benchmark_datasets


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _doc_identifier(task_name: str, index: int, doc: dict) -> str:
    for field in ("key", "ind", "source_id", "id"):
        if field in doc:
            return f"{task_name}:{field}={doc[field]}"
    return f"{task_name}:eval_doc_index={index}"


def _indices(specification: list[int] | dict[str, int]) -> list[int]:
    if isinstance(specification, list):
        return specification
    if not {"start", "stop"} <= set(specification):
        raise ValueError("range selection requires start and stop")
    return list(
        range(
            specification["start"],
            specification["stop"],
            specification.get("step", 1),
        )
    )


def build(selection_path: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    with pinned_benchmark_datasets():
        manager = TaskManager()
        resolved_benchmarks = {}
        for benchmark, benchmark_selection in selection["benchmarks"].items():
            task_names = list(benchmark_selection["samples"])
            loaded = manager.load(task_names)["tasks"]
            resolved_tasks = {}
            for task_name, specification in benchmark_selection["samples"].items():
                indices = _indices(specification)
                task = loaded[task_name]
                docs = list(task.eval_docs)
                resolved_samples = []
                for index in indices:
                    doc = docs[index]
                    resolved_samples.append(
                        {
                            "eval_doc_index": index,
                            "stable_id": _doc_identifier(task_name, index, doc),
                            "doc_sha256_canonical_json": hashlib.sha256(
                                _canonical_json(doc)
                            ).hexdigest(),
                        }
                    )
                resolved_tasks[task_name] = {
                    "task_version": str(task.VERSION),
                    "output_type": task.OUTPUT_TYPE,
                    "available_eval_docs": len(docs),
                    "selected_samples": resolved_samples,
                }
            benchmark_metadata = BENCHMARKS[benchmark]
            resolved_benchmarks[benchmark] = {
                **benchmark_metadata,
                "harness_tasks": benchmark_selection["harness_tasks"],
                "selection_rule": selection["selection_rule"],
                "sample_count": sum(
                    len(_indices(specification))
                    for specification in benchmark_selection["samples"].values()
                ),
                "tasks": resolved_tasks,
            }

    manifest = {
        "schema_version": 1,
        "phase": selection["phase"],
        "locked_before_substantive_scoring": selection["phase"]
        != "plumbing_smoke_non_evidentiary",
        "model": {
            "id": IT_MODEL_ID,
            "revision": IT_MODEL_REVISION,
            "tokenizer_revision": IT_TOKENIZER_REVISION,
            "weights_sha256": IT_WEIGHTS_SHA256,
            "chat_template_sha256": IT_CHAT_TEMPLATE_SHA256,
            "frozen": True,
            "dtype": CAPABILITY_DTYPE,
            "device": CAPABILITY_DEVICE,
            "attention_implementation": CAPABILITY_ATTENTION_IMPLEMENTATION,
            "max_length": CAPABILITY_MAX_LENGTH,
        },
        "recirculation": {
            "destination_layer_zero_based": DESTINATION_LAYER,
            "source_layer_zero_based": SOURCE_LAYER,
            "alpha": ALPHA,
            "beta": BETA,
            "alpha_ramp_tokens": RAMP_STEPS,
            "normalization": NORMALIZATION,
            "additional_recurrence_iterations": NUM_RECURRENCE_STEPS,
        },
        "harness": {
            "repository": HARNESS_REPOSITORY,
            "revision": HARNESS_REVISION,
            "batch_size": 1,
            "apply_chat_template": True,
            "fewshot_as_multiturn": True,
            "task_default_fewshot_and_generation_settings": True,
            "random_seed": CAPABILITY_SEED,
            "numpy_random_seed": CAPABILITY_SEED,
            "torch_random_seed": CAPABILITY_SEED,
            "fewshot_random_seed": CAPABILITY_SEED,
        },
        "selection_rule": selection["selection_rule"],
        "benchmarks": resolved_benchmarks,
    }
    manifest["manifest_sha256_without_self_hash"] = hashlib.sha256(
        _canonical_json(manifest)
    ).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
