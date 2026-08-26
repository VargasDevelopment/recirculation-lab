"""Pinned lm-evaluation-harness adapter for paired Gemma capability runs."""

from __future__ import annotations

import hashlib
import os
import resource
import sys
import time
from pathlib import Path

import psutil
import torch
import transformers
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer

from .capability_constants import (
    CAPABILITY_ATTENTION_IMPLEMENTATION,
    CAPABILITY_DEVICE,
    CAPABILITY_DTYPE,
    CAPABILITY_MAX_LENGTH,
    IT_CHAT_TEMPLATE_SHA256,
    IT_MODEL_ID,
    IT_MODEL_REVISION,
    IT_TOKENIZER_JSON_SHA256,
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
from .gemma3_recirculation import Gemma3ForCausalLM


def token_ids_sha256(token_ids: torch.Tensor | list[int]) -> str:
    """Hash token IDs as stable little-endian unsigned 32-bit integers."""
    tensor = torch.as_tensor(token_ids, dtype=torch.int64).detach().cpu()
    payload = tensor.numpy().astype("<u4", copy=False).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_path() -> Path:
    cache_root = Path(
        os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub")
    )
    return (
        cache_root / "models--google--gemma-3-1b-it" / "snapshots" / IT_MODEL_REVISION
    )


def _rss_peak_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


class CapabilityHFLM(HFLM):
    """Harness model whose only lane difference is fixed recirculation."""

    def __init__(self, condition: str):
        if condition not in {"baseline", "recirculation"}:
            raise ValueError(f"unknown condition: {condition}")
        if not torch.backends.mps.is_available():
            raise RuntimeError("the pinned capability experiment requires MPS")

        snapshot = _snapshot_path()
        weights_path = snapshot / "model.safetensors"
        tokenizer_path = snapshot / "tokenizer.json"
        if _sha256_file(weights_path) != IT_WEIGHTS_SHA256:
            raise RuntimeError("pinned IT weight artifact hash mismatch")
        if _sha256_file(tokenizer_path) != IT_TOKENIZER_JSON_SHA256:
            raise RuntimeError("pinned IT tokenizer artifact hash mismatch")

        tokenizer = AutoTokenizer.from_pretrained(
            IT_MODEL_ID,
            revision=IT_TOKENIZER_REVISION,
            local_files_only=True,
        )
        if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != (
            IT_CHAT_TEMPLATE_SHA256
        ):
            raise RuntimeError("pinned IT chat template hash mismatch")

        load_start = time.perf_counter()
        model = (
            Gemma3ForCausalLM.from_pretrained(
                IT_MODEL_ID,
                revision=IT_MODEL_REVISION,
                dtype=torch.bfloat16,
                attn_implementation=CAPABILITY_ATTENTION_IMPLEMENTATION,
                local_files_only=True,
            )
            .eval()
            .requires_grad_(False)
            .to(CAPABILITY_DEVICE)
        )
        torch.mps.synchronize()
        self.model_load_seconds = time.perf_counter() - load_start
        self.condition = condition
        self.call_events: list[dict] = []

        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            revision=IT_MODEL_REVISION,
            device=CAPABILITY_DEVICE,
            dtype=CAPABILITY_DTYPE,
            batch_size=1,
            max_length=CAPABILITY_MAX_LENGTH,
            logits_cache=False,
        )

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
            raise AssertionError("model weights are not frozen")
        self._parameter_versions = tuple(
            parameter._version for parameter in model.parameters()
        )
        self._initial_driver_bytes = torch.mps.driver_allocated_memory()
        self._initial_tensor_bytes = torch.mps.current_allocated_memory()

    def _record_event(
        self,
        *,
        operation: str,
        input_ids: torch.Tensor,
        runtime_seconds: float,
        **fields,
    ) -> None:
        self.call_events.append(
            {
                "operation": operation,
                "input_tokens": int(input_ids.shape[1]),
                "input_token_ids_sha256_le_u32": token_ids_sha256(input_ids),
                "runtime_seconds": runtime_seconds,
                "mps_driver_allocated_bytes_after_sync": (
                    torch.mps.driver_allocated_memory()
                ),
                "mps_tensor_allocated_bytes_after_sync": (
                    torch.mps.current_allocated_memory()
                ),
                **fields,
            }
        )

    def _model_call(
        self,
        inps: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inps.shape[0] != 1:
            raise ValueError("capability evaluation is deliberately batch-size one")
        if labels is not None:
            raise ValueError("the pinned tasks require causal, not seq2seq, scoring")
        if attn_mask is not None and not torch.all(attn_mask == 1):
            raise ValueError("padded likelihood batches are not supported")
        attention_mask = torch.ones_like(inps)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(
                inps, attention_mask=attention_mask, use_cache=False
            ).logits
        torch.mps.synchronize()
        self._record_event(
            operation="loglikelihood",
            input_ids=inps,
            runtime_seconds=time.perf_counter() - start,
        )
        return logits

    def _model_generate(
        self,
        context: torch.Tensor,
        max_length: int,
        stop: list[str],
        attention_mask: torch.Tensor | None = None,
        **generation_kwargs,
    ) -> torch.Tensor:
        if context.shape[0] != 1:
            raise ValueError("capability evaluation is deliberately batch-size one")
        if attention_mask is not None and not torch.all(attention_mask == 1):
            raise ValueError("padded generation batches are not supported")
        if generation_kwargs.get("do_sample", False):
            raise ValueError(
                "the locked capability experiment requires greedy decoding"
            )
        unsupported = set(generation_kwargs) - {"do_sample", "temperature"}
        if unsupported:
            raise ValueError(f"unsupported generation settings: {sorted(unsupported)}")

        started = time.perf_counter()
        with torch.inference_mode():
            if self.condition == "baseline":
                sequences, prefill_seconds, decode_seconds = self._baseline_greedy(
                    context, max_length, stop
                )
            else:
                sequences, prefill_seconds, decode_seconds = self._recirculation_greedy(
                    context, max_length, stop
                )
        torch.mps.synchronize()
        self._record_event(
            operation="generation",
            input_ids=context,
            runtime_seconds=time.perf_counter() - started,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            generated_tokens=int(sequences.shape[1] - context.shape[1]),
            generated_token_ids_sha256_le_u32=token_ids_sha256(
                sequences[:, context.shape[1] :]
            ),
        )
        return sequences

    def _should_stop(self, generated: list[int], stop: list[str]) -> bool:
        eos_ids = self.model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        if generated[-1] in eos_ids:
            return True
        rendered = self.tokenizer.decode(generated, skip_special_tokens=False)
        return any(sequence and sequence in rendered for sequence in stop)

    def _baseline_greedy(
        self, context: torch.Tensor, max_length: int, stop: list[str]
    ) -> tuple[torch.Tensor, float, float]:
        prefill_start = time.perf_counter()
        output = self.model(
            context, attention_mask=torch.ones_like(context), use_cache=True
        )
        torch.mps.synchronize()
        prefill_seconds = time.perf_counter() - prefill_start
        sequences = context.clone()
        generated: list[int] = []
        decode_seconds = 0.0
        while sequences.shape[1] < max_length:
            next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(next_ids.item()))
            sequences = torch.cat((sequences, next_ids), dim=1)
            if self._should_stop(generated, stop):
                break
            decode_start = time.perf_counter()
            output = self.model(
                next_ids,
                attention_mask=torch.ones_like(sequences),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            torch.mps.synchronize()
            decode_seconds += time.perf_counter() - decode_start
        return sequences, prefill_seconds, decode_seconds

    def _recirculation_greedy(
        self, context: torch.Tensor, max_length: int, stop: list[str]
    ) -> tuple[torch.Tensor, float, float]:
        prefill_start = time.perf_counter()
        output, state = self.model.recirculating_prefill(
            context, attention_mask=torch.ones_like(context)
        )
        torch.mps.synchronize()
        prefill_seconds = time.perf_counter() - prefill_start
        sequences = context.clone()
        generated: list[int] = []
        decode_seconds = 0.0
        while sequences.shape[1] < max_length:
            next_ids = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(int(next_ids.item()))
            sequences = torch.cat((sequences, next_ids), dim=1)
            if self._should_stop(generated, stop):
                break
            decode_start = time.perf_counter()
            output, state = self.model.recirculating_decode_step(next_ids, state)
            torch.mps.synchronize()
            decode_seconds += time.perf_counter() - decode_start
        return sequences, prefill_seconds, decode_seconds

    def verify_unchanged(self) -> dict:
        versions_unchanged = self._parameter_versions == tuple(
            parameter._version for parameter in self.model.parameters()
        )
        return {
            "all_weights_frozen": not any(
                parameter.requires_grad for parameter in self.model.parameters()
            ),
            "parameter_version_counters_unchanged": versions_unchanged,
            "process_peak_rss_bytes": _rss_peak_bytes(),
            "process_rss_bytes": psutil.Process().memory_info().rss,
            "initial_mps_driver_allocated_bytes": self._initial_driver_bytes,
            "initial_mps_tensor_allocated_bytes": self._initial_tensor_bytes,
            "max_observed_mps_driver_allocated_bytes_after_sync": max(
                [self._initial_driver_bytes]
                + [
                    event["mps_driver_allocated_bytes_after_sync"]
                    for event in self.call_events
                ]
            ),
            "max_observed_mps_tensor_allocated_bytes_after_sync": max(
                [self._initial_tensor_bytes]
                + [
                    event["mps_tensor_allocated_bytes_after_sync"]
                    for event in self.call_events
                ]
            ),
        }

    def get_model_info(self) -> dict:
        return {
            "model_id": IT_MODEL_ID,
            "model_revision": IT_MODEL_REVISION,
            "tokenizer_revision": IT_TOKENIZER_REVISION,
            "dtype": CAPABILITY_DTYPE,
            "device": CAPABILITY_DEVICE,
            "parameter_count": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
            "transformers_version": transformers.__version__,
        }
