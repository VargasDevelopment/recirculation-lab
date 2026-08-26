# Public-release audit

Audit date: 2026-08-25

## Security and artifact boundary

- Scanned the complete four-commit public history (63 unique blobs) for common
  Hugging Face, GitHub, OpenAI, AWS, and Slack token forms; private-key headers;
  and local `/Users/...` paths. No matches remained.
- Ran `detect-secrets` 1.5.0 over all publication candidates. Its only findings
  were expected high-entropy hexadecimal model revisions, artifact hashes, and
  token-window hashes; these were manually reviewed as non-secret provenance.
- Largest historical blob: 2,534,602 bytes (the earlier social card). No blob is
  larger than 5 MB.
- No model checkpoints, tokenizer bundles, Hugging Face caches, full dataset
  files, virtual environments, `.env` files, shell histories, debug logs, or
  editor metadata are published.
- The downloaded paper PDF and local environment remain gitignored.

## Provenance boundary

- The local experimental history originally included a substantial adapter
  derived from an author gist that states no license.
- Before any public remote was created, that path was removed from every public
  commit with `git-filter-repo`. The public implementation contains no source
  copied from the gist.
- The replacement adapter builds on Apache-2.0 Transformers code and retains the
  relevant Google/Hugging Face notice.
- Exact behavioral equivalence was established before removal and recorded in
  `results/implementation_equivalence.json`.
- Gemma weights remain under Google's Gemma terms and are downloaded by each
  user. The PG-19 parquet mirror is not included.

## Reproducibility and site checks

- Both result summaries regenerate byte-for-byte from their raw lane files.
- Weight-free test suite, Ruff format, and Ruff lint checks pass locally.
- Real-checkpoint checks verify ordinary-path bitwise identity, fixed-layer
  injection, coefficient/norm behavior, causal isolation across the sliding
  boundary, native cache lengths, frozen weights, and unchanged parameter
  version counters.
- The static report and OG image return HTTP 200 in local preview, and report
  integrity tests enforce distinct exploratory/confirmatory counts and public
  link targets.

GitHub Actions intentionally runs only weight-free checks. The gated real-model
commands remain documented manual reproductions.
