"""Run one pinned capability benchmark lane through lm-evaluation-harness."""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import torch
import transformers
from lm_eval import evaluator

from .capability_constants import CAPABILITY_SEED, HARNESS_REVISION
from .capability_model import CapabilityHFLM
from .harness_pins import pinned_benchmark_datasets


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_indices(benchmark: dict) -> dict[str, list[int]]:
    return {
        task_name: [sample["eval_doc_index"] for sample in task["selected_samples"]]
        for task_name, task in benchmark["tasks"].items()
    }


def _json_safe(value):
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return re.sub(r"0x[0-9a-fA-F]+", "0xADDR", value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, functools.partial):
        return {
            "callable": f"{value.func.__module__}.{value.func.__qualname__}",
            "args": _json_safe(value.args),
            "keywords": _json_safe(value.keywords),
        }
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def run(condition: str, benchmark_name: str, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    benchmark = manifest["benchmarks"][benchmark_name]
    model = CapabilityHFLM(condition)
    started = time.perf_counter()
    with pinned_benchmark_datasets():
        result = evaluator.simple_evaluate(
            model=model,
            tasks=benchmark["harness_tasks"],
            samples=_selected_indices(benchmark),
            batch_size=1,
            apply_chat_template=True,
            fewshot_as_multiturn=True,
            log_samples=True,
            bootstrap_iters=0,
            random_seed=CAPABILITY_SEED,
            numpy_random_seed=CAPABILITY_SEED,
            torch_random_seed=CAPABILITY_SEED,
            fewshot_random_seed=CAPABILITY_SEED,
        )
    torch.mps.synchronize()
    evaluation_seconds = time.perf_counter() - started
    verification = model.verify_unchanged()
    if not all(
        verification[key]
        for key in ("all_weights_frozen", "parameter_version_counters_unchanged")
    ):
        raise AssertionError("model immutability verification failed")
    if result is None:
        raise RuntimeError("lm-evaluation-harness returned no result")

    return {
        "schema_version": 1,
        "phase": manifest["phase"],
        "condition": condition,
        "benchmark": benchmark_name,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "harness_revision": HARNESS_REVISION,
        "model_info": model.get_model_info(),
        "model_load_seconds": model.model_load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "harness_results": _json_safe(result.get("results", {})),
        "harness_groups": _json_safe(result.get("groups", {})),
        "harness_configs": _json_safe(result.get("configs", {})),
        "harness_versions": _json_safe(result.get("versions", {})),
        "harness_n_shot": _json_safe(result.get("n-shot", {})),
        "samples": _json_safe(result.get("samples", {})),
        "call_events": model.call_events,
        "verification": verification,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition", choices=["baseline", "recirculation"], required=True
    )
    parser.add_argument(
        "--benchmark",
        choices=["mmlu_pro", "gsm8k", "ifeval", "hellaswag"],
        required=True,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.condition, args.benchmark, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
