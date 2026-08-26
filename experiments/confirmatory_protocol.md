# Locked confirmatory protocol: unseen PG-19 books 2–9

This protocol was written before evaluating either condition on the confirmatory
sample. It extends commit `4f6760e` without changing its model, mechanism, or
scoring implementation.

## Locked sample

- Dataset and revision: unchanged from the exploratory experiment
- Records: dataset indices 2 through 9 inclusive, excluding exploratory records
  0 and 1
- Windows: the first five complete, non-overlapping 1,024-token windows from
  each record
- Targets: positions 1–1,023 in each window
- Planned total: 8 books, 40 windows, 40,920 predicted tokens
- No book replacement, window exclusion, extension, or early stopping

The committed manifest must contain ordered token IDs and SHA-256 hashes and
must pass both document-index and token-hash non-overlap checks against
`experiments/pg19_windows.json`.

## Frozen conditions

Both lanes use the exact checkpoint revision, tokenizer revision, bfloat16 MPS
backend, eager/native Gemma attention masks, loss implementation, and per-window
reset behavior from `4f6760e`. The recirculation lane remains source layer 11 to
destination layer 4 (zero-based), alpha 0.15, beta 0.85, ten-token ramp,
hidden-axis L2 norm matching, and one additional recurrence iteration. All
weights remain frozen.

## Locked interpretation

A window or book is effectively tied only when its absolute paired mean-NLL
difference is at most `1e-6` per predicted token.

- **CONFIRMED:** aggregate perplexity is lower, at least 5 of 8 books improve,
  and at least 21 of 40 windows improve.
- **NOT CONFIRMED:** aggregate perplexity is not lower, no more than 4 books
  improve, and no more than 20 windows improve.
- **MIXED:** every other outcome.

The report will include aggregate, book-level, and window-level paired results.
A 20,000-resample book-block bootstrap will keep each book's five windows
together. Because these are eight sequential records rather than a random book
sample, its interval is descriptive stability evidence, not population-level
confidence.

For concentration, the effect is labeled `reasonably_consistent` only if at
least five books improve and the two largest positive book-level NLL gains
account for no more than 75% of all positive book-level NLL gain. Otherwise it
is labeled `highly_concentrated_or_mixed`.
