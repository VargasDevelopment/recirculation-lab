"""Token-weighted causal-language-model perplexity."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def shifted_nll(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return summed float32 NLL and number of next-token targets."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("expected logits [B,L,V] and input_ids [B,L]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids sequence shapes differ")
    shifted_logits = logits[:, :-1, :].float().contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    count = shifted_labels.numel()
    nll = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        shifted_labels.view(-1),
        reduction="sum",
    )
    return nll, count


def perplexity(total_nll: float, predicted_tokens: int) -> float:
    if predicted_tokens <= 0:
        raise ValueError("predicted_tokens must be positive")
    return math.exp(total_nll / predicted_tokens)
