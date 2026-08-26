# Copyright 2025 Google Inc. and Hugging Face Inc. team
# Copyright 2026 Joseph Vargas
# SPDX-License-Identifier: Apache-2.0
"""Fixed recirculation for the Transformers Gemma 3 text model.

This module is an independent, compact implementation of the algorithm in
Mozer et al., *Recirculation* (arXiv:2608.17981). The cache timing was checked
against Siddiqui's author reference gist, but no gist source is vendored here.

Ordinary execution delegates directly to Transformers. Recirculation reuses the
official model's layers and weights while controlling only the rolling query
stack, layer-to-layer residual injection, and cache commit timing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM as TransformersGemma3ForCausalLM,
)
from transformers.models.gemma3.modeling_gemma3 import (
    _bidirectional_window_overlay,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


@dataclass(frozen=True)
class FixedRecirculationConfig:
    destination_layer: int
    source_layer: int
    destination_weight: float
    source_weight: float
    recurrence_steps: int
    normalization: str | None
    ramp_steps: int


def fixed_recirculation_mix(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    destination_weight: float,
    source_weight: float,
    normalization: str | None,
) -> torch.Tensor:
    """Apply the paper's tokenwise norm match and convex residual mix."""
    if normalization == "norm_rep":
        source_norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
        destination_norm = torch.linalg.vector_norm(destination, dim=-1, keepdim=True)
        source = source / (source_norm + 1e-8) * destination_norm
    elif normalization is not None:
        raise ValueError(f"unknown normalization: {normalization}")
    return destination_weight * destination + source_weight * source


def ramped_alpha(alpha: float, token_index: int, ramp_steps: int) -> float:
    """Return Appendix B.3's alpha_t = min(t / ramp_steps, 1) * alpha."""
    if ramp_steps < 0:
        raise ValueError("ramp_steps must be non-negative")
    return alpha if ramp_steps == 0 else min(token_index / ramp_steps, 1.0) * alpha


def _attention_with_cache_policy(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache,
    cache_position: torch.LongTensor,
    cache_policy: str,
    **kwargs,
) -> torch.Tensor:
    """Run one official Gemma attention module with an explicit cache write policy."""
    if cache_policy not in {"all", "first", "none"}:
        raise ValueError(f"invalid cache policy: {cache_policy}")

    input_shape = hidden_states.shape[:-1]
    projected_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(hidden_states).view(projected_shape).transpose(1, 2)
    key = attention.k_proj(hidden_states).view(projected_shape).transpose(1, 2)
    value = attention.v_proj(hidden_states).view(projected_shape).transpose(1, 2)
    query = attention.q_norm(query)
    key = attention.k_norm(key)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    if cache_policy == "all":
        key, value = past_key_values.update(
            key, value, attention.layer_idx, cache_kwargs
        )
    elif cache_policy == "first":
        first_cache_kwargs = {
            "sin": sin[:, :1, :],
            "cos": cos[:, :1, :],
            "cache_position": cache_position[:1],
        }
        cached_key, cached_value = past_key_values.update(
            key[:, :, :1, :],
            value[:, :, :1, :],
            attention.layer_idx,
            first_cache_kwargs,
        )
        key = torch.cat((cached_key, key[:, :, 1:, :]), dim=2)
        value = torch.cat((cached_value, value[:, :, 1:, :]), dim=2)
    elif past_key_values.layers[attention.layer_idx].is_initialized:
        layer_cache = past_key_values.layers[attention.layer_idx]
        key = torch.cat((layer_cache.keys, key), dim=2)
        value = torch.cat((layer_cache.values, value), dim=2)

    attention_fn: Callable = eager_attention_forward
    if attention.config._attn_implementation != "eager":
        attention_fn = ALL_ATTENTION_FUNCTIONS[attention.config._attn_implementation]
    attended, _ = attention_fn(
        attention,
        query,
        key,
        value,
        attention_mask,
        dropout=attention.attention_dropout if attention.training else 0.0,
        scaling=attention.scaling,
        sliding_window=attention.sliding_window,
        **kwargs,
    )
    return attention.o_proj(attended.reshape(*input_shape, -1).contiguous())


def _decoder_layer_with_cache_policy(
    layer: nn.Module,
    hidden_states: torch.Tensor,
    *,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache,
    cache_position: torch.LongTensor,
    cache_policy: str,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = layer.input_layernorm(hidden_states)
    hidden_states = _attention_with_cache_policy(
        layer.self_attn,
        hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        cache_policy=cache_policy,
        **kwargs,
    )
    hidden_states = residual + layer.post_attention_layernorm(hidden_states)
    residual = hidden_states
    hidden_states = layer.pre_feedforward_layernorm(hidden_states)
    hidden_states = layer.mlp(hidden_states)
    hidden_states = layer.post_feedforward_layernorm(hidden_states)
    return residual + hidden_states


def _build_masks_and_positions(
    text_model: nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | dict | None,
    position_ids: torch.LongTensor,
    cache_position: torch.LongTensor,
    cache: Cache,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    if isinstance(attention_mask, dict):
        masks = attention_mask
    else:
        mask_kwargs = {
            "config": text_model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": cache,
            "position_ids": position_ids,
        }
        sliding_kwargs = mask_kwargs.copy()
        if text_model.config.use_bidirectional_attention:
            mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(
                True, dtype=torch.bool
            )
            sliding_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                text_model.config.sliding_window
            )
        masks = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**sliding_kwargs),
        }
    positions = {
        layer_type: text_model.rotary_emb(inputs_embeds, position_ids, layer_type)
        for layer_type in text_model.config.layer_types
    }
    return masks, positions


def _recirculating_text_forward(
    text_model: nn.Module,
    recirculation: FixedRecirculationConfig,
    *,
    input_ids: torch.LongTensor | None,
    attention_mask: torch.Tensor | dict | None,
    position_ids: torch.LongTensor | None,
    past_key_values: Cache | None,
    inputs_embeds: torch.FloatTensor | None,
    cache_position: torch.LongTensor | None,
    **kwargs,
) -> BaseModelOutputWithPast:
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("specify exactly one of input_ids or inputs_embeds")
    if past_key_values is not None:
        raise ValueError("recirculation requires an empty per-sequence cache")
    if inputs_embeds is None:
        inputs_embeds = text_model.embed_tokens(input_ids)

    steps = recirculation.recurrence_steps
    sequence_length = inputs_embeds.shape[1]
    if steps < 0 or steps >= text_model.config.sliding_window:
        raise ValueError("recurrence_steps must fit inside the sliding window")
    if sequence_length < steps:
        raise ValueError("sequence is shorter than recurrence_steps")

    cache = DynamicCache(config=text_model.config)
    if cache_position is None:
        cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    masks, positions = _build_masks_and_positions(
        text_model,
        inputs_embeds,
        attention_mask,
        position_ids,
        cache_position,
        cache,
    )

    batch, _, width = inputs_embeds.shape
    feedback = torch.zeros(
        (batch, steps + 1, width),
        dtype=inputs_embeds.dtype,
        device=inputs_embeds.device,
    )
    padded_inputs = torch.nn.functional.pad(inputs_embeds, (0, 0, steps, 0, 0, 0))
    padded_positions = {
        name: tuple(
            torch.nn.functional.pad(value, (0, 0, steps, 0, 0, 0)) for value in pair
        )
        for name, pair in positions.items()
    }
    padded_cache_position = torch.nn.functional.pad(cache_position, (steps, 0))

    blocked = torch.min(next(iter(masks.values())))
    warmup_mask = torch.full(
        (1, 1, steps + 1, sequence_length),
        blocked,
        dtype=blocked.dtype,
        device=blocked.device,
    )
    for slot in range(steps + 1):
        warmup_mask[:, :, slot, slot] = 0

    outputs = []
    for token_index in range(sequence_length):
        stop = token_index + steps + 1
        layer_input = padded_inputs[:, token_index:stop, :]
        if token_index < steps:
            warmup_mask[:, :, -1 - token_index :, steps - token_index] = 0
            layer_mask = warmup_mask
        else:
            oldest_sliding_key = max(
                0,
                token_index - steps - text_model.config.sliding_window + 1,
            )
            query_start = token_index - steps
            query_stop = token_index + 1

        for layer_index, layer in enumerate(
            text_model.layers[: text_model.config.num_hidden_layers]
        ):
            if token_index >= steps:
                key_start = (
                    oldest_sliding_key
                    if layer.attention_type == "sliding_attention"
                    else 0
                )
                layer_mask = masks[layer.attention_type][
                    :, :, query_start:query_stop, key_start:
                ]
            layer_positions = tuple(
                value[:, token_index:stop, :]
                for value in padded_positions[layer.attention_type]
            )
            layer_output = _decoder_layer_with_cache_policy(
                layer,
                layer_input,
                position_embeddings=layer_positions,
                attention_mask=layer_mask,
                past_key_values=cache,
                cache_position=padded_cache_position[token_index:stop],
                cache_policy="none" if token_index < steps else "first",
                **kwargs,
            )

            if layer_index == recirculation.destination_layer:
                feedback = torch.cat(
                    (feedback[:, 1:, :], layer_output[:, -1:, :]), dim=1
                )
                layer_input = feedback
            elif layer_index == recirculation.source_layer:
                alpha = ramped_alpha(
                    recirculation.source_weight,
                    token_index,
                    recirculation.ramp_steps,
                )
                beta = (
                    1.0 - alpha
                    if recirculation.ramp_steps
                    else recirculation.destination_weight
                )
                feedback = fixed_recirculation_mix(
                    feedback,
                    layer_output,
                    destination_weight=beta,
                    source_weight=alpha,
                    normalization=recirculation.normalization,
                )
                layer_input = layer_output
            else:
                layer_input = layer_output
        outputs.append(layer_output[:, -1:, :])

    return BaseModelOutputWithPast(
        last_hidden_state=text_model.norm(torch.cat(outputs, dim=1)),
        past_key_values=cache,
    )


class Gemma3ForCausalLM(TransformersGemma3ForCausalLM):
    """Official Gemma 3 LM with an opt-in fixed-recirculation forward path."""

    def __init__(self, config):
        super().__init__(config)
        self._fixed_recirculation: FixedRecirculationConfig | None = None
        # Retained as a small compatibility/introspection flag for experiment checks.
        self.model.num_recurrence_steps = None

    def set_recirculation_args(
        self,
        target_layer: int,
        source_layer: int,
        target_layer_weight: float,
        source_layer_weight: float,
        num_recurrence_steps: int = 0,
        normalization: str | None = None,
        ramp_steps: int = 0,
    ) -> None:
        if not 0 <= target_layer < source_layer < len(self.model.layers):
            raise ValueError(
                "layers must satisfy 0 <= destination < source < layer count"
            )
        if target_layer_weight < 0 or source_layer_weight < 0:
            raise ValueError("mixing weights must be non-negative")
        if abs(target_layer_weight + source_layer_weight - 1.0) > 1e-9:
            raise ValueError("mixing weights must sum to one")
        if normalization not in {None, "norm_rep"}:
            raise ValueError("unsupported normalization")
        if num_recurrence_steps < 0 or ramp_steps < 0:
            raise ValueError("recurrence and ramp steps must be non-negative")
        self._fixed_recirculation = FixedRecirculationConfig(
            destination_layer=target_layer,
            source_layer=source_layer,
            destination_weight=target_layer_weight,
            source_weight=source_layer_weight,
            recurrence_steps=num_recurrence_steps,
            normalization=normalization,
            ramp_steps=ramp_steps,
        )
        self.model.num_recurrence_steps = num_recurrence_steps

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self._fixed_recirculation is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        outputs = _recirculating_text_forward(
            self.model,
            self._fixed_recirculation,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        selected = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, selected, :])
        if self.config.final_logit_softcapping is not None:
            logits = torch.tanh(logits / self.config.final_logit_softcapping)
            logits = logits * self.config.final_logit_softcapping
        loss = (
            self.loss_function(logits, labels, self.vocab_size, **kwargs)
            if labels is not None
            else None
        )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
        )
