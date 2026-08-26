# Gemma 3 1B fixed-recirculation reproduction

## Outcome

**PASS.** On this Apple M3 Max, fixed recirculation reduced perplexity on the
locked local PG-19 subset while preserving the exact frozen Gemma 3 1B PT
weights and evaluation tokens.

| Measurement | Ordinary forward | Fixed recirculation |
|---|---:|---:|
| Perplexity | 33.45225 | 28.87729 |
| Total NLL | 35,908.5173 | 34,404.0566 |
| Evaluation runtime | 2.713 s | 309.158 s |
| Throughput | 3,770.08 tok/s | 33.09 tok/s |
| Max observed MPS driver allocation after sync | 6.137 GB | 8.478 GB |

- Absolute perplexity reduction: **4.57496**
- Relative perplexity reduction: **13.676%**
- Paper's full-PG-19 reference reduction: **14.41%** (22.27 to 19.06)
- Windows with lower NLL: **8 of 10**
- Evaluated next-token targets: **10,230**
- Runtime cost in this serial prefill implementation: **113.9x**
- Additional observed MPS driver allocation: **2.340 GB**

The percentage improvement is close to the paper, but the absolute perplexities
must not be compared directly: this run evaluates only 10 windows from two books,
whereas the paper evaluates about 10.8 million PG-19 tokens.

Browse the shareable, self-contained [HTML field report](report/index.html) for a
visual summary of the method, controls, result, and caveats.

## Exact experiment

- Hardware: Apple M3 Max, 64 GB unified memory
- Device/dtype: PyTorch MPS, bfloat16
- Model: `google/gemma-3-1b-pt`
- Model/tokenizer revision: `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29`
- Weight artifact SHA-256:
  `ee5250f6eb1aa7cfb729dfd4dc8d9964fd772324776c6d00bf2bc674c069cb27`
- Parameters: 999,885,952, all frozen
- Attention: eager; native Gemma layer masks (512-token sliding attention plus
  full-attention layers)
- Software: Python 3.12.6, PyTorch 2.10.0, Transformers 5.0.0
- Data: `emozilla/pg19-test`, test split, revision
  `c5e39bf32e33f9111323aa68d7d9000d22722035`
- Selection: tokenize each full document with special tokens, then take the first
  five non-overlapping 1024-token windows from each of the first two records
- Scoring: positions 1–1023 in every independent window; summed float32
  cross-entropy, then `exp(total_nll / 10230)`

The ordered token IDs and their per-window SHA-256 values are committed in
[`experiments/pg19_windows.json`](experiments/pg19_windows.json). Both conditions
validate and consume that same manifest.

## Fixed mechanism

The implementation vendors the author's public PyTorch/Hugging Face reference
gist and pins the compatible Transformers release. For Gemma 3 1B it uses:

- zero-based source layer 11 and destination layer 4
- one additional recurrence iteration (a rolling two-token stack)
- tokenwise hidden-dimension L2 source normalization to destination magnitude
- convex mixing with source weight alpha=0.15 and destination weight beta=0.85
- the paper's 10-position ramp, `alpha_t = min(t / 10, 1) * 0.15`, with
  `beta_t = 1 - alpha_t`
- empty per-window KV cache; warmup writes no token, then each layer commits only
  the rolling window's leftmost token
- ordinary first-pass readout for the newest token

The upstream gist needed two small paper-alignment changes: its stale assertion
rejected beta below 0.9 even though the paper specifies 0.85, and it did not
include the final paper's first-ten-token ramp. Both changes are marked
`REPRODUCTION` in the source.

## Verification

`uv run pytest` exercises loss shifting, exact baseline equivalence, source and
destination tensor shapes, norm matching, coefficient ramping, recurrence call
counts, returned cache length, future-token causality, and weight immutability.

The real-checkpoint verification additionally records:

- official Transformers versus vendored baseline logits: bitwise equal, max
  absolute difference 0
- one source mix per input step; shape `[1, 2, 1152]` for both tensors
- unchanged logits through position 513 after changing positions 514–529, crossing
  the model's 512-token sliding-attention boundary
- native returned cache lengths on the 530-token fixture: 511 positions for
  sliding-attention layers and 529 for full-attention layers
- unchanged parameter version counters and no trainable parameters

See [`results/verification.json`](results/verification.json) and the 19 tests in
[`tests/`](tests/).

## Reproduce

The model is gated; authenticate with Hugging Face and accept its license first.

```bash
uv sync --frozen
uv run python -m recirculation.data --output experiments/pg19_windows.json
uv run pytest -q
uv run python -m recirculation.run_condition \
  --condition baseline --output results/baseline.json
uv run python -m recirculation.run_condition \
  --condition recirculation --output results/recirculation.json
uv run python -m recirculation.verify_real_model \
  --output results/verification.json
uv run python -m recirculation.summarize --output results/comparison.json
```

`HF_HUB_DISABLE_PROGRESS_BARS=1` may be prefixed to model commands for quiet
logs. Model weights stay in the ordinary Hugging Face cache; dependencies remain
inside this project's `.venv`.

## Caveats and audit trail

- The paper does not pin model, tokenizer, data, dtype, device, or software
  revisions. This repository does.
- The paper's exact PG-19 artifact and preprocessing are unspecified. This run
  uses a pinned parquet mirror and only 0.095% as many predicted tokens.
- The paper says ramping was introduced for 1B but does not explicitly say that
  Table 1 includes it. This run includes it as the closest reading of the final
  method.
- The initial two-window pilot (the first chunk from each book) worsened PPL by
  0.396%. Before any tuning, the sample was extended once by the content-blind
  rule above and then stopped at 10 windows regardless of outcome. The eight
  added windows alone show a 16.874% reduction. The pilot is preserved in
  [`experiments/pilot_two_window.json`](experiments/pilot_two_window.json).
- A diagnostic initially supplied the same full-context mask to sliding and full
  layers. It produced an invalid baseline PPL of 75.636 and was rejected after
  an ordinary-forward mask equivalence check. Correct native masks yield 19.201
  on that same two-window pilot. Invalid numbers are not present in final result
  files.
- The descriptive paired-window bootstrap interval is 7.69%–19.55%, but windows
  from the same book are dependent; it is sensitivity evidence, not a
  population-level confidence interval.
- Runtime here measures serial prefill. The paper's near-zero generation-latency
  claim concerns parallel recurrent/new-token stacks and is not tested by the
  author's full-sequence reference implementation.
- MPS memory values are the maximum sampled allocator values after synchronized
  windows, not a hardware peak-memory counter.

## Most interesting next experiment (not run)

Repeat the identical locked configuration on a held-out, content-blind sample:
the first five complete windows from the next eight PG-19 test books. That would
test whether the effect survives broader document variation without changing any
mechanism hyperparameter.
