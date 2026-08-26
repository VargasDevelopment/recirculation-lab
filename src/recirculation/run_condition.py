"""Run one pinned baseline or fixed-recirculation condition on MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import psutil
import torch
import transformers

from .author_recurrent_gemma3 import Gemma3ForCausalLM
from .constants import (
    ALPHA,
    ATTENTION_IMPLEMENTATION,
    BETA,
    DATASET_ID,
    DATASET_REVISION,
    DESTINATION_LAYER,
    DEVICE_NAME,
    DTYPE_NAME,
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZATION,
    NUM_RECURRENCE_STEPS,
    RAMP_STEPS,
    SOURCE_LAYER,
)
from .data import load_manifest
from .perplexity import perplexity, shifted_nll


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rss_peak_bytes() -> int:
    # Darwin reports bytes; Linux reports KiB.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def model_artifact() -> tuple[Path, str]:
    cache_root = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))
    snapshot = (
        cache_root
        / "models--google--gemma-3-1b-pt"
        / "snapshots"
        / MODEL_REVISION
    )
    weights = snapshot / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"missing pinned weights: {weights}")
    return weights, sha256_file(weights)


def run(condition: str, manifest_path: Path) -> dict:
    if condition not in {"baseline", "recirculation"}:
        raise ValueError(condition)
    if DEVICE_NAME != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the pinned experiment requires available MPS")

    torch.manual_seed(20260818)
    manifest = load_manifest(manifest_path)
    weights_path, weights_sha256 = model_artifact()

    load_start = time.perf_counter()
    model = Gemma3ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation=ATTENTION_IMPLEMENTATION,
        local_files_only=True,
    )
    model.eval().requires_grad_(False).to(DEVICE_NAME)
    torch.mps.synchronize()
    load_seconds = time.perf_counter() - load_start

    if condition == "recirculation":
        model.set_recirculation_args(
            target_layer=DESTINATION_LAYER,
            source_layer=SOURCE_LAYER,
            target_layer_weight=BETA,
            source_layer_weight=ALPHA,
            num_recurrence_steps=NUM_RECURRENCE_STEPS,
            normalization=NORMALIZATION,
            ramp_steps=RAMP_STEPS,
        )
    elif model.model.num_recurrence_steps is not None:
        raise AssertionError("baseline did not select the ordinary forward path")

    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("base-model weights are not frozen")
    parameter_versions = tuple(parameter._version for parameter in model.parameters())

    per_window = []
    total_nll = 0.0
    total_tokens = 0
    evaluation_start = time.perf_counter()
    observed_driver_bytes = torch.mps.driver_allocated_memory()
    observed_tensor_bytes = torch.mps.current_allocated_memory()

    with torch.inference_mode():
        for ordinal, window in enumerate(manifest["windows"], start=1):
            input_ids = torch.tensor(
                [window["token_ids"]], dtype=torch.long, device=DEVICE_NAME
            )
            # A full, unpadded 2D mask lets Transformers build distinct causal
            # masks for Gemma's 512-token sliding and full-attention layers.
            attention_mask = torch.ones_like(input_ids)
            window_start = time.perf_counter()
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            nll_tensor, predicted_tokens = shifted_nll(output.logits, input_ids)
            nll = float(nll_tensor.item())
            torch.mps.synchronize()
            window_seconds = time.perf_counter() - window_start
            observed_driver_bytes = max(observed_driver_bytes, torch.mps.driver_allocated_memory())
            observed_tensor_bytes = max(observed_tensor_bytes, torch.mps.current_allocated_memory())
            total_nll += nll
            total_tokens += predicted_tokens
            per_window.append(
                {
                    "document_index": window["document_index"],
                    "window_index_within_document": window["window_index_within_document"],
                    "token_ids_sha256_le_u32": window["token_ids_sha256_le_u32"],
                    "predicted_tokens": predicted_tokens,
                    "nll_sum": nll,
                    "mean_nll": nll / predicted_tokens,
                    "perplexity": perplexity(nll, predicted_tokens),
                    "runtime_seconds": window_seconds,
                }
            )
            print(
                f"{condition}: window {ordinal}/{len(manifest['windows'])} "
                f"document={window['document_index']} "
                f"window={window['window_index_within_document']} "
                f"runtime={window_seconds:.3f}s",
                file=sys.stderr,
                flush=True,
            )
            del output, nll_tensor, input_ids, attention_mask

    evaluation_seconds = time.perf_counter() - evaluation_start
    if parameter_versions != tuple(parameter._version for parameter in model.parameters()):
        raise AssertionError("a model parameter was modified in-place during evaluation")

    return {
        "schema_version": 1,
        "condition": condition,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "weights_file": weights_path.name,
        "weights_sha256": weights_sha256,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "weights_frozen": True,
        "parameter_version_counters_unchanged": True,
        "dtype": DTYPE_NAME,
        "device": DEVICE_NAME,
        "attention_implementation": ATTENTION_IMPLEMENTATION,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "manifest_path": str(manifest_path),
        "manifest_total_predicted_tokens": manifest["total_predicted_tokens"],
        "evaluated_predicted_tokens": total_tokens,
        "total_nll": total_nll,
        "mean_nll": total_nll / total_tokens,
        "perplexity": perplexity(total_nll, total_tokens),
        "model_load_seconds": load_seconds,
        "evaluation_seconds": evaluation_seconds,
        "tokens_per_second": total_tokens / evaluation_seconds,
        "process_peak_rss_bytes": rss_peak_bytes(),
        "max_observed_mps_driver_allocated_bytes_after_sync": observed_driver_bytes,
        "max_observed_mps_tensor_allocated_bytes_after_sync": observed_tensor_bytes,
        "recirculation": None
        if condition == "baseline"
        else {
            "destination_layer_zero_based": DESTINATION_LAYER,
            "source_layer_zero_based": SOURCE_LAYER,
            "alpha": ALPHA,
            "beta": BETA,
            "num_recurrence_steps": NUM_RECURRENCE_STEPS,
            "normalization": NORMALIZATION,
            "ramp_steps": RAMP_STEPS,
        },
        "windows": per_window,
        "process_rss_bytes_at_end": psutil.Process().memory_info().rss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["baseline", "recirculation"], required=True)
    parser.add_argument("--manifest", type=Path, default=Path("experiments/pg19_windows.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.condition, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
