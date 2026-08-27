# Locked Gemma 3 1B IT capability-transfer report

## Headline

**Overall classification: MIXED.** The unchanged fixed-recirculation mechanism
produced small positive directional results on MMLU-Pro and normalized
HellaSwag, but it regressed on GSM8K and IFEval. The confirmed PG-19 perplexity
effect therefore did **not** transfer uniformly into downstream capability on
this locked Gemma 3 1B IT suite.

| Primary metric | N | Baseline | Recirculation | Delta | Favorable / unfavorable flips | Exact paired p | Descriptive paired-bootstrap 95% interval | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| MMLU-Pro exact match | 42 | 9.524% | 11.905% | +2.381 pp | 2 / 1 | 1.0000 | −4.76 to +9.52 pp | IMPROVED |
| GSM8K flexible exact match | 50 | 40.000% | 30.000% | −10.000 pp | 3 / 8 | 0.2266 | −22.00 to +2.00 pp | REGRESSED |
| IFEval prompt-level strict | 50 | 58.000% | 54.000% | −4.000 pp | 3 / 5 | 0.7266 | −16.00 to +6.00 pp | REGRESSED |
| HellaSwag normalized accuracy | 100 | 41.000% | 43.000% | +2.000 pp | 6 / 4 | 0.7539 | −4.00 to +8.00 pp | IMPROVED |

No paired result is conventionally statistically significant. The intervals
are deterministic-subset stability descriptions, not population confidence
intervals for the full benchmarks.

Secondary canonical metrics agree with the mixed headline:

- GSM8K strict exact match: 16.000% → 8.000% (−8.000 pp; 0 favorable,
  4 unfavorable flips).
- IFEval instruction-level strict: 71.053% → 68.421% (−2.632 pp); prompt-level
  loose: 62.000% → 58.000% (−4.000 pp); instruction-level loose: 73.684% →
  72.368% (−1.316 pp).
- HellaSwag raw accuracy: 43.000% → 43.000% (2 favorable and 2 unfavorable
  flips). The positive HellaSwag classification is specifically for the
  canonical length-normalized primary metric.

## Locked configuration

- Model/tokenizer: `google/gemma-3-1b-it` at revision
  `dcc83ea841ab6100d6b47a070329e1ba4cf78752`
- Weight artifact SHA-256:
  `3d4ef8d71c14db7e448a09ebe891cfb6bf32c57a9b44499ae0d1c098e48516b6`
- Model: 999,885,952 parameters, bfloat16, PyTorch MPS/eager attention, frozen
- Mechanism: zero-based layer 11 → 4, alpha 0.15, beta 0.85, ten-token alpha
  ramp, hidden-axis L2 norm matching, one additional recurrence iteration
- Harness: EleutherAI `lm-evaluation-harness` commit
  `dd417662e5bda6a247489ec28d2ff46a45d1c42c`
- Inference: batch size one; native Gemma IT chat template; harness-default
  few-shot settings as multiturn chat; deterministic greedy generation
- Locked samples: MMLU-Pro 42 (3 × 14 categories), GSM8K 50, IFEval 50,
  HellaSwag 100; all disjoint from the non-evidentiary smoke set
- GPQA: predeclared omission because the four priority tasks filled the local
  runtime budget

Exact task/dataset versions, sample identifiers, document hashes, selections,
seeds, and prompt policy are in
[`../../experiments/capability_locked_manifest.json`](../../experiments/capability_locked_manifest.json)
and the pre-outcome
[`../../experiments/capability_locked_protocol.md`](../../experiments/capability_locked_protocol.md).

## Verification

All checks passed. The paired validator established identical sample identities,
rendered prompt arguments, prompt hashes, model-call token hashes, checkpoint,
tokenizer, software settings, and scoring configuration in every benchmark.
Every lane reported frozen weights and unchanged parameter version counters.

The real-model preflight additionally established ordinary-adapter equality to
Transformers, deterministic generation, intended layer routing and mixing,
norm matching, causality across the 512-token sliding-attention boundary,
expected cache lengths, and matching incremental-versus-fresh recirculation
greedy tokens. See [`../capability_verification.json`](../capability_verification.json).

## Representative paired flips

Examples below use the first locked flips in each direction, not a favorable
selection. Full outputs for every example remain in the raw lane JSON files;
all changed primary-metric examples are copied into
[`comparison.json`](comparison.json).

### MMLU-Pro

- Favorable, biology index 1: both answers identified option D, but baseline
  ended with bold `D` and the canonical extractor returned invalid;
  recirculation used the requested `(D)` format and scored correct. This gain
  is better format compliance, not new biological knowledge.
- Favorable, other index 3: on the newsletter writing-category question,
  baseline selected technical writing (E) and recirculation correctly selected
  promotional writing (G).
- Unfavorable, history index 2 (the only unfavorable flip): baseline selected
  the keyed Cahokia answer C; recirculation changed it to D.

### GSM8K

- Favorable, index 19: the trail-average problem changed from 4 mph to the
  correct 6 mph.
- Favorable, index 20: the spilled mixed-drink calculation changed from 14.67 L
  to the correct 15 L.
- Unfavorable, index 11: baseline correctly totaled three per-dozen prices as
  $694; recirculation multiplied the dozen prices by 12 again and returned
  $8,328.
- Unfavorable, index 14: baseline correctly found 60% hip-hop enrollment;
  recirculation subtracted the contemporary students twice and returned 30%.

### IFEval

- Favorable, index 5: recirculation satisfied the canonical exact-three-bullet
  check where baseline did not.
- Favorable, index 20: the response had to finish with an exact refund question;
  baseline appended a postamble after it, while recirculation ended at the
  required phrase.
- Unfavorable, index 13: baseline kept an entire movie review lowercase;
  recirculation introduced capitalized names including `Mark Zuckerberg`.
- Unfavorable, index 16: the prompt requested exactly two names separated by
  six asterisks; baseline gave two, while recirculation expanded to five names
  with repeated separators.

### HellaSwag (normalized)

- Favorable, index 21: lawn-mowing continuation changed from “runs from one
  side” to the keyed “walks back and forth as he mows.”
- Favorable, index 25: the face-care continuation changed from an unrelated
  painted-lips ending to the keyed oily-skin/spray continuation.
- Unfavorable, index 22: an ice-cream family scene changed from the keyed
  “enjoys eating the dessert together” to an unrelated hut/photograph ending.
- Unfavorable, index 27: a parent instructing children at a sink changed from
  the keyed teeth-brushing continuation to an unrelated lipstick ending.

## Behavioral and performance observations

Recirculation made generative responses longer more often than shorter:
30/42 MMLU-Pro responses, 30/50 GSM8K responses, and 31/50 IFEval responses
were longer, adding 5,868 generated tokens across those lanes. The IFEval
examples show why length can cut both ways: extra text sometimes completed a
required structure, but it also introduced forbidden capitalization, extra
answers, or text after an exact required ending. This is descriptive, not a
causal explanation.

| Benchmark | Baseline runtime | Recirculation runtime | Ratio | Baseline / recirculation seconds per example |
|---|---:|---:|---:|---:|
| MMLU-Pro | 359.41 s | 2,325.30 s | 6.47× | 8.56 / 55.36 |
| GSM8K | 351.45 s | 1,980.63 s | 5.64× | 7.03 / 39.61 |
| IFEval | 954.09 s | 1,192.47 s | 1.25× | 19.08 / 23.85 |
| HellaSwag | 25.45 s | 613.24 s | 24.10× | 0.25 / 6.13 |

Summed substantive evaluator time was 7,802.05 seconds (2 h 10 m 2 s):
1,690.40 seconds baseline and 6,111.64 seconds recirculation, a 3.62× aggregate
runtime ratio. This ratio mixes likelihood and generation tasks and is not a
production-latency benchmark. Peak measurements are process/allocator samples,
not hardware high-water counters; exact per-lane values are in
[`comparison.json`](comparison.json).

The serial locked run itself elapsed 2 h 11 m 27 s from the first substantive
lane log to the comparison artifact. The broader scientific phase—from the
first pinned IT checkpoint artifact fetch through adapter verification, smoke,
locking, all eight lanes, and the comparison artifact—elapsed approximately
2 h 53 m 43 s by filesystem timestamps. This second figure includes engineering
and setup time and is reported as an auditable wall-clock approximation, not
benchmark compute.

## Interpretation

The clean answer is **MIXED**, not capability signal. Two primary metrics moved
up, but both positive deltas were small and uncertain. Two moved down, including
a 10-point GSM8K regression. The existing fixed PT calibration demonstrably
changes IT behavior, but on this practical locked subset it does not make the
instruction-tuned model consistently better at useful tasks.

The single best next experiment is a **full HellaSwag validation run using the
same frozen IT configuration**, because likelihood scoring avoids generative
format effects and the locked 100-item normalized result was the cleanest
positive downstream direction. It is deliberately not run here.
