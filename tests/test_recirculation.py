from unittest.mock import patch

import pytest
import torch
from transformers import Gemma3ForCausalLM as HFGemma3ForCausalLM
from transformers import Gemma3TextConfig

import recirculation.gemma3_recirculation as recirculation_module
from recirculation.gemma3_recirculation import (
    Gemma3ForCausalLM,
    fixed_recirculation_mix,
    ramped_alpha,
)


def tiny_config() -> Gemma3TextConfig:
    config = Gemma3TextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=16,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    config._attn_implementation = "eager"
    return config


def paired_models() -> tuple[HFGemma3ForCausalLM, Gemma3ForCausalLM]:
    torch.manual_seed(7)
    baseline = HFGemma3ForCausalLM(tiny_config()).eval()
    author = Gemma3ForCausalLM(tiny_config()).eval()
    result = author.load_state_dict(baseline.state_dict(), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    return baseline, author


def test_author_baseline_is_exactly_hugging_face_forward() -> None:
    baseline, author = paired_models()
    input_ids = torch.tensor([[2, 10, 11, 12, 13, 14]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        expected = baseline(
            input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        actual = author(
            input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
    assert torch.equal(actual, expected)


def test_norm_matched_convex_mix() -> None:
    destination = torch.tensor([[[3.0, 4.0], [0.0, 2.0]]])
    source = torch.tensor([[[0.0, 10.0], [3.0, 4.0]]])
    mixed = fixed_recirculation_mix(
        destination,
        source,
        destination_weight=0.85,
        source_weight=0.15,
        normalization="norm_rep",
    )
    scaled_source = (
        source
        / source.norm(dim=-1, keepdim=True)
        * destination.norm(dim=-1, keepdim=True)
    )
    assert torch.allclose(mixed, 0.85 * destination + 0.15 * scaled_source)
    assert torch.allclose(
        scaled_source.norm(dim=-1), destination.norm(dim=-1), atol=1e-6
    )


@pytest.mark.parametrize(
    ("token_index", "expected"),
    [(0, 0.0), (1, 0.015), (5, 0.075), (10, 0.15), (99, 0.15)],
)
def test_paper_ramp(token_index: int, expected: float) -> None:
    assert ramped_alpha(0.15, token_index, 10) == pytest.approx(expected)


def test_recirculation_hits_source_once_per_step_and_preserves_weights() -> None:
    _, model = paired_models()
    model.requires_grad_(False)
    model.set_recirculation_args(
        target_layer=0,
        source_layer=2,
        target_layer_weight=0.85,
        source_layer_weight=0.15,
        num_recurrence_steps=1,
        normalization="norm_rep",
        ramp_steps=0,
    )
    input_ids = torch.tensor([[2, 10, 11, 12, 13, 14]])
    attention_mask = torch.ones_like(input_ids)
    before = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    observed_shapes = []
    real_mix = recirculation_module.fixed_recirculation_mix

    def observing_mix(destination, source, **kwargs):
        observed_shapes.append((destination.shape, source.shape))
        return real_mix(destination, source, **kwargs)

    with (
        patch.object(
            recirculation_module, "fixed_recirculation_mix", side_effect=observing_mix
        ),
        torch.inference_mode(),
    ):
        output = model(input_ids, attention_mask=attention_mask, use_cache=False)

    assert output.logits.shape == (1, input_ids.shape[1], tiny_config().vocab_size)
    assert output.past_key_values.get_seq_length() == input_ids.shape[1] - 1
    assert len(observed_shapes) == input_ids.shape[1]
    assert set(observed_shapes) == {(torch.Size([1, 2, 32]), torch.Size([1, 2, 32]))}
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_first_readout_is_ordinary_and_later_readouts_change() -> None:
    _, model = paired_models()
    input_ids = torch.tensor([[2, 10, 11, 12, 13, 14]])
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        baseline = model(
            input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
    model.set_recirculation_args(
        target_layer=0,
        source_layer=2,
        target_layer_weight=0.85,
        source_layer_weight=0.15,
        num_recurrence_steps=1,
        normalization="norm_rep",
        ramp_steps=0,
    )
    with torch.inference_mode():
        recirculated = model(
            input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
    assert torch.allclose(recirculated[:, 0], baseline[:, 0], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(recirculated[:, 2:], baseline[:, 2:])


def test_future_tokens_cannot_change_earlier_recirculated_logits() -> None:
    _, model = paired_models()
    model.set_recirculation_args(
        target_layer=0,
        source_layer=2,
        target_layer_weight=0.85,
        source_layer_weight=0.15,
        num_recurrence_steps=1,
        normalization="norm_rep",
        ramp_steps=0,
    )
    first = torch.tensor([[2, 10, 11, 12, 13, 14]])
    second = torch.tensor([[2, 10, 11, 99, 98, 97]])
    with torch.inference_mode():
        first_logits = model(
            first, attention_mask=torch.ones_like(first), use_cache=False
        ).logits
        second_logits = model(
            second, attention_mask=torch.ones_like(second), use_cache=False
        ).logits
    assert torch.allclose(
        first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-6
    )


def test_incremental_decode_matches_fresh_full_sequence_recirculation() -> None:
    _, model = paired_models()
    model.set_recirculation_args(
        target_layer=0,
        source_layer=2,
        target_layer_weight=0.85,
        source_layer_weight=0.15,
        num_recurrence_steps=1,
        normalization="norm_rep",
        ramp_steps=10,
    )
    sequence = torch.tensor([[2, 10, 11, 12, 13]])
    with torch.inference_mode():
        prefill, state = model.recirculating_prefill(
            sequence, attention_mask=torch.ones_like(sequence)
        )
        full_prefill = model(
            sequence, attention_mask=torch.ones_like(sequence), use_cache=False
        )
    torch.testing.assert_close(
        prefill.logits[:, -1], full_prefill.logits[:, -1], atol=1e-6, rtol=1e-6
    )
    assert torch.equal(
        prefill.logits[:, -1].argmax(dim=-1),
        full_prefill.logits[:, -1].argmax(dim=-1),
    )

    for new_token in (14, 15, 16):
        new_input = torch.tensor([[new_token]])
        sequence = torch.cat((sequence, new_input), dim=1)
        with torch.inference_mode():
            step, state = model.recirculating_decode_step(new_input, state)
            fresh = model(
                sequence, attention_mask=torch.ones_like(sequence), use_cache=False
            )
        assert torch.allclose(
            step.logits[:, -1], fresh.logits[:, -1], atol=1e-6, rtol=1e-6
        )
        assert state.sequence_length == sequence.shape[1]
        assert state.past_key_values.get_seq_length() == sequence.shape[1] - 1
