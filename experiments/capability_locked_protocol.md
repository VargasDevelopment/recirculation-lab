# Locked Gemma 3 1B IT capability-transfer protocol

Status: **locked before substantive scoring** on 2026-08-26.

This experiment asks whether the already-confirmed fixed-recirculation mechanism
transfers from lower PG-19 perplexity on Gemma 3 1B PT to downstream capability
on the frozen `google/gemma-3-1b-it` checkpoint. Smoke examples are plumbing
fixtures only and are excluded from every substantive sample.

## Frozen model and mechanism

- model/tokenizer: `google/gemma-3-1b-it`
- revision: `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- weights: bfloat16 on PyTorch MPS, eager attention, fully frozen
- source layer 11 -> destination layer 4 (zero-based)
- alpha 0.15, beta 0.85, 10-token alpha ramp
- hidden-axis L2 source norm matching
- one additional recurrence iteration
- no tuning, adapters, controller, quantization, or parameter updates

## Canonical evaluation implementation

EleutherAI lm-evaluation-harness commit
`dd417662e5bda6a247489ec28d2ff46a45d1c42c` is used with its built-in task
prompts, filters, scorers, few-shot counts, generation limits, and stopping
rules. All dataset Hub commits are pinned in the manifest. The Gemma tokenizer's
native chat template is applied with few-shot examples represented as multiple
turns. Inference is batch-size one and deterministic greedy decoding.

## Locked disjoint samples

The machine-readable source of truth is
[`capability_locked_manifest.json`](capability_locked_manifest.json).

| Benchmark | Canonical task version | Locked selection | N |
|---|---:|---|---:|
| MMLU-Pro | 3.1 | filtered indices 1-3 in every one of 14 categories | 42 |
| GSM8K | 3.0 | test indices 5-54 | 50 |
| IFEval | 4.0 | canonical evaluation split indices 5-54 | 50 |
| HellaSwag | 1.0 | validation indices 5-104 | 100 |

The non-evidentiary smoke set used index 0 in five MMLU-Pro categories and
indices 0-4 in the other tasks. The locked document hashes have zero overlap
with those smoke hashes. Sample sizes were chosen only from measured runtime:
approximately 52 seconds per MMLU-Pro item, 39 seconds per GSM8K item, 23 seconds
per IFEval item, and 6 seconds per HellaSwag item in the recirculation smoke
lane. Smoke accuracy was not used for selection.

## Metrics and paired interpretation

- MMLU-Pro: canonical extracted exact-match accuracy; overall and category
  results.
- GSM8K: canonical strict and flexible extraction exact match; flexible extract
  is the primary reported metric.
- IFEval: prompt- and instruction-level strict and loose accuracy; prompt-level
  strict is the primary metric.
- HellaSwag: canonical raw accuracy and length-normalized accuracy;
  length-normalized accuracy is primary.
- Every metric reports favorable and unfavorable paired flips.
- Accuracy deltas use an exact two-sided McNemar/binomial test on discordant
  pairs and a fixed-seed paired bootstrap interval. These are descriptive on
  the locked subsets, not claims about all benchmark examples.

No example may be replaced, discarded, added, or stopped early. Prompts,
mechanism settings, parsing, and generation settings may not change after any
substantive result is observed. All four complete paired lanes must be retained
regardless of direction. GPQA is not included because the four priority tasks
already consume the practical local runtime budget; it remains explicitly
non-blocking and is not substituted with a custom task.

## Classification rules

Each benchmark is classified from its primary metric:

- **IMPROVED**: positive delta with more favorable than unfavorable flips.
- **REGRESSED**: negative delta with more unfavorable than favorable flips.
- **UNCHANGED**: no score or paired-outcome difference.
- **MIXED / INCONCLUSIVE**: any other small-sample or metric-conflicted result.

Overall:

- **CAPABILITY SIGNAL**: multiple meaningful primary metrics improve with no
  comparable regression.
- **NO TRANSFER**: no meaningful downstream gain is observed across the suite.
- **MIXED**: improvements and regressions coexist, or evidence is otherwise
  inconsistent.
