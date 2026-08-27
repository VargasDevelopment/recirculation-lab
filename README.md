# Gemma 3 1B fixed-recirculation reproduction

Independent Apple Silicon / PyTorch reproduction of fixed recirculation from
[Mozer et al., *Recirculation*](https://arxiv.org/abs/2608.17981). The core
perplexity experiments use frozen
[`google/gemma-3-1b-pt`](https://huggingface.co/google/gemma-3-1b-pt) weights; a
separate locked capability-transfer milestone uses frozen
[`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) weights.

Normally, information moves through a transformer once from early layers to
late layers. Fixed recirculation takes a residual representation from a deeper
layer, rescales it to match an earlier representation's magnitude, mixes the two,
and sends that state through the intervening layers one additional time. The
weights never change; only the inference computation does.

**Measured outcomes:** perplexity on pinned PG-19 windows and accuracy on four
small, locked downstream benchmark subsets. Neither is a percentage measure of
general model quality or intelligence.

## Confirmatory outcome

**CONFIRMED.** With the mechanism frozen before evaluation, fixed recirculation
reduced aggregate perplexity on all eight previously unseen PG-19 books. The
sample, token hashes, stopping rule, and classification rule were committed
before either new condition was evaluated and remain in the
[locked protocol](experiments/confirmatory_protocol.md).

| Measurement | Ordinary forward | Fixed recirculation |
|---|---:|---:|
| Perplexity | 27.23190 | 23.55482 |
| Total NLL | 135,215.5967 | 129,279.7966 |
| Evaluation runtime | 10.585 s | 1,240.263 s |
| Throughput | 3,865.71 tok/s | 32.99 tok/s |
| Max observed MPS driver allocation after sync | 6.137 GB | 8.478 GB |

- Absolute perplexity reduction: **3.67708**
- Relative perplexity reduction: **13.503%**
- Books improved: **8 of 8**
- Windows improved/worsened/tied: **32 / 8 / 0**
- Evaluated next-token targets: **40,920** across 40 windows
- Runtime cost in this serial prefill implementation: **117.2x**
- Additional observed MPS driver allocation: **2.340 GB**
- Descriptive whole-book bootstrap interval: **10.70%–16.12%**
- Two largest book gains' share of positive NLL gain: **34.3%**

The effect is broad rather than dominated by a few books: every book improved,
and individual book reductions ranged from 7.49% to 19.25%. A systematic
boundary pattern remains important: window 0 of every book worsened by
1.46%–3.97%, while windows 1–4 improved in every book. Nothing was removed or
retuned in response.

The confirmatory magnitude is 0.173 percentage points below the original
13.676% exploratory result and 0.907 points below the paper's 14.41% PG-19
reference.

Machine-readable artifacts:

- [`results/validation_comparison.json`](results/validation_comparison.json):
  aggregate, per-book, and all 40 paired per-window results
- [`results/validation_baseline.json`](results/validation_baseline.json) and
  [`results/validation_recirculation.json`](results/validation_recirculation.json):
  raw condition outputs
- [`experiments/pg19_validation_books_2_9.json`](experiments/pg19_validation_books_2_9.json):
  locked token IDs, record metadata, and hashes
- [`experiments/confirmatory_protocol.md`](experiments/confirmatory_protocol.md):
  pre-outcome selection, stopping, and classification rules

Browse the [HTML field report](https://vargasdevelopment.github.io/recirculation-lab/)
for the visual methodology, verification evidence, and caveats.

## Capability-transfer outcome (Gemma 3 1B IT)

**MIXED.** With the same mechanism unchanged, two primary metrics moved up and
two moved down. The confirmed perplexity reduction did not transfer uniformly
into downstream task performance.

| Locked primary metric | N | Baseline | Recirculation | Delta | Favorable / unfavorable flips | Classification |
|---|---:|---:|---:|---:|---:|---|
| MMLU-Pro exact match | 42 | 9.524% | 11.905% | +2.381 pp | 2 / 1 | IMPROVED |
| GSM8K flexible exact match | 50 | 40.000% | 30.000% | −10.000 pp | 3 / 8 | REGRESSED |
| IFEval prompt-level strict | 50 | 58.000% | 54.000% | −4.000 pp | 3 / 5 | REGRESSED |
| HellaSwag normalized accuracy | 100 | 41.000% | 43.000% | +2.000 pp | 6 / 4 | IMPROVED |

None of the paired deltas is conventionally statistically significant on these
subsets. HellaSwag raw accuracy was unchanged at 43%; GSM8K strict exact match
fell from 16% to 8%. The most informative behavioral pattern was increased
verbosity: recirculation produced longer output on 91 of 142 generative
examples. That sometimes repaired required formatting, but also introduced
extra answers, forbidden capitalization, and arithmetic detours.

See the [full capability report](results/capability/REPORT.md),
[machine-readable comparison](results/capability/comparison.json), and
[pre-outcome locked protocol](experiments/capability_locked_protocol.md). The
original smoke artifacts remain separately labeled non-evidentiary.

### Per-book results

| Record | Book | Baseline PPL | Recirculation PPL | Reduction | Improving windows |
|---:|---|---:|---:|---:|---:|
| 2 | Travels in Morocco, Vol. 2 | 39.6202 | 33.0576 | 16.564% | 4/5 |
| 3 | Impressions of Theophrastus Such | 51.0193 | 42.4333 | 16.829% | 4/5 |
| 4 | Odd Craft, Part 4 | 19.8284 | 18.3424 | 7.494% | 4/5 |
| 5 | The S. W. F. Club | 19.3801 | 17.7407 | 8.459% | 4/5 |
| 6 | Frank Merriwell Down South | 24.8161 | 21.4131 | 13.713% | 4/5 |
| 7 | Critical Miscellanies, Vol. 2 | 31.0531 | 25.0763 | 19.247% | 4/5 |
| 8 | From Sand Hill to Pine | 36.0620 | 32.2304 | 10.625% | 4/5 |
| 9 | Child's Health Primer | 14.0101 | 11.9957 | 14.378% | 4/5 |

## Exploratory outcome (preserved)

The original two-book experiment remains unchanged: baseline PPL **33.45225**,
recirculation PPL **28.87729**, a **13.676%** reduction, with 8 of 10 windows
improving over 10,230 targets. Its original condition and comparison JSON files
remain under `results/`, and its sample remains
[`experiments/pg19_windows.json`](experiments/pg19_windows.json).

## Exact experiment configuration

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
- Exploratory selection: first five complete windows from records 0–1
- Confirmatory selection: first five complete windows from unseen records 2–9
- Scoring: positions 1–1023 in every independent window; summed float32
  cross-entropy, then `exp(total_nll / evaluated_targets)`

The confirmatory generator and result summarizer enforce selection identity,
ordered hashes, exactly 40,920 targets, frozen mechanism fields, matched software
and checkpoint fields, and zero document/hash overlap with the exploratory
manifest.

The capability milestone changes only the checkpoint to
`google/gemma-3-1b-it` at model/tokenizer revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752` (weight SHA-256
`3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6`).
It uses the native Gemma chat template and EleutherAI `lm-evaluation-harness`
commit `dd417662e5bda6a247489ec28d2ff46a45d1c42c`; the mechanism and numerical
backend remain unchanged.

## Fixed mechanism

The public implementation is a compact adapter over the Apache-2.0 Transformers
Gemma 3 classes. It was independently written from the paper's algorithm after
using the author's reference gist to resolve cache timing. No unlicensed gist
source is redistributed. On the pinned Mac, its first 64-token logits and a full
1,024-token per-target loss vector are bitwise identical to the reference-derived
implementation that produced both recorded experiments; see
[`results/implementation_equivalence.json`](results/implementation_equivalence.json).

For Gemma 3 1B it uses:

- zero-based source layer 11 and destination layer 4
- one additional recurrence iteration (a rolling two-token stack)
- tokenwise hidden-dimension L2 source normalization to destination magnitude
- convex mixing with source weight alpha=0.15 and destination weight beta=0.85
- the paper's 10-position ramp, `alpha_t = min(t / 10, 1) * 0.15`, with
  `beta_t = 1 - alpha_t`
- empty per-window KV cache; warmup writes no token, then each layer commits only
  the rolling window's leftmost token
- ordinary first-pass readout for the newest token

The author gist predates the final paper's 1B ramp and contains a stale assertion
that rejects beta=0.85. The implementation follows the final paper for both
points and records that choice explicitly.

## Verification

`uv run pytest` exercises loss shifting, exact baseline equivalence, source and
destination tensor shapes, norm matching, coefficient ramping, recurrence call
counts, returned cache length, future-token causality, and weight immutability.

The real-checkpoint verification additionally records:

- official Transformers versus the public adapter's ordinary path: bitwise equal, max
  absolute difference 0
- one source mix per input step; shape `[1, 2, 1152]` for both tensors
- unchanged logits through position 513 after changing positions 514–529, crossing
  the model's 512-token sliding-attention boundary
- native returned cache lengths on the 530-token fixture: 511 positions for
  sliding-attention layers and 529 for full-attention layers
- unchanged parameter version counters and no trainable parameters

See [`results/verification.json`](results/verification.json), the IT
[`results/capability_verification.json`](results/capability_verification.json),
and the 42 weight-free tests in [`tests/`](tests/).

## What this does not establish

The PT experiments establish lower perplexity only on their specified PG-19
windows. The IT milestone measures four modest locked subsets and is mixed; it
does not establish broad improvements in reasoning, coding, instruction
following, agentic task completion, or general model "intelligence." In
particular, a percentage reduction in perplexity is not an equivalent percentage
increase in any downstream capability.

## Reproduce

Requirements: an Apple Silicon Mac with an MPS-capable PyTorch installation,
Python 3.12, [`uv`](https://docs.astral.sh/uv/), and enough unified memory for the
1B checkpoint. The model is gated: log in with `hf auth login` and accept the
Gemma terms on the model page before running the real-model commands.

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

# Confirmatory sample and protocol are already locked in experiments/.
uv run python -m recirculation.run_condition \
  --condition baseline \
  --manifest experiments/pg19_validation_books_2_9.json \
  --output results/validation_baseline.json
uv run python -m recirculation.run_condition \
  --condition recirculation \
  --manifest experiments/pg19_validation_books_2_9.json \
  --output results/validation_recirculation.json
uv run python -m recirculation.summarize_validation \
  --baseline results/validation_baseline.json \
  --recirculation results/validation_recirculation.json \
  --manifest experiments/pg19_validation_books_2_9.json \
  --output results/validation_comparison.json

# Gemma 3 1B IT preflight and complete locked capability suite.
uv run python -m recirculation.verify_capability \
  --smoke-dir results/capability_smoke \
  --output results/capability_verification.json
./scripts/run_locked_capability_suite.sh
```

`HF_HUB_DISABLE_PROGRESS_BARS=1` may be prefixed to model commands for quiet
logs. Model weights stay in the ordinary Hugging Face cache; dependencies remain
inside this project's `.venv`.

## Project structure

| Path | Purpose |
|---|---|
| `src/recirculation/` | Compact Gemma adapter, loss/data runners, summaries, and real-model checks |
| `experiments/` | Locked protocols, selected record metadata, token IDs, and hashes |
| `results/` | Raw lane outputs plus aggregate, per-book, and per-window comparisons |
| `tests/` | Weight-free unit, invariant, schema, and report-integrity checks |
| `report/` | Static GitHub Pages field report and social image |
| `sources/` | Paper, reference-code, model, dataset, and license provenance |

## Caveats and audit trail

- The paper does not pin model, tokenizer, data, dtype, device, or software
  revisions. This repository does.
- The paper's exact PG-19 artifact and preprocessing are unspecified. The
  confirmatory run uses a pinned parquet mirror and only about 0.379% as many
  predicted tokens as the paper.
- The paper says ramping was introduced for 1B but does not explicitly say that
  Table 1 includes it. Both runs include it as the closest reading of the final
  method.
- In the earlier exploratory run, the initial two-window pilot worsened PPL by
  0.396%. Before any tuning, the sample was extended once by the content-blind
  rule above and then stopped at 10 windows regardless of outcome. The eight
  added windows alone show a 16.874% reduction. The pilot is preserved in
  [`experiments/pilot_two_window.json`](experiments/pilot_two_window.json).
- During that earlier exploratory work, a diagnostic initially supplied the same
  full-context mask to sliding and full layers. It produced an invalid baseline
  PPL of 75.636 and was rejected after
  an ordinary-forward mask equivalence check. Correct native masks yield 19.201
  on that same two-window pilot. Invalid numbers are not present in final result
  files.
- The confirmatory descriptive interval resamples whole five-window books and is
  10.70%–16.12%. The records are eight sequential dataset entries, not a random
  book sample, so this is stability evidence rather than population-level
  confidence.
- All eight record-initial windows worsened while all 32 later windows improved.
  This exact boundary pattern is reported as observed; it was not used to alter
  the locked sample or mechanism.
- Runtime here measures serial prefill. The paper's near-zero generation-latency
  claim concerns parallel recurrent/new-token stacks and is not tested by this
  reference-style full-sequence evaluator.
- MPS memory values are the maximum sampled allocator values after synchronized
  windows, not a hardware peak-memory counter.

## Most interesting next experiment (not run)

Run the full HellaSwag validation split with the same frozen Gemma 3 1B IT
configuration. Likelihood scoring avoids generative format effects, and the
locked 100-item subset produced the cleanest positive capability direction
(normalized accuracy 41% → 43%) while raw accuracy was unchanged. No parameter,
prompt, or sample-dependent tuning should precede that run.

## Sources, licensing, and citation

Primary sources are the [paper](https://arxiv.org/abs/2608.17981) by Michael C.
Mozer, Shoaib Ahmed Siddiqui, Danny Sawyer, Sunny Sanyal, and Rosanne Liu, and
Siddiqui's [author reference implementation](https://gist.github.com/shoaibahmed/10702acc01cc5a169fdbc1719932438f).
See [`sources/SOURCES.md`](sources/SOURCES.md) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for exact revisions and
provenance.

Repository-authored source, tests, and documentation are released under
Apache-2.0. Gemma weights are **not** included and remain governed by Google's
Gemma terms. The PG-19 mirror is not redistributed; the committed experiment
manifests contain only the selected evaluation token IDs, metadata, and hashes.
