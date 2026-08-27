# Public-release audit

Audit date: 2026-08-26

## Security and artifact boundary

- Scanned the complete ten-commit public history (100 unique blobs) for common
  Hugging Face, GitHub, OpenAI, AWS, and Slack token forms and private-key
  headers. No credential-shaped matches were found. Capability smoke artifacts
  added after the original release audit contained an absolute local harness
  path; the current public tree replaces it with a portable task identifier.
- Ran `detect-secrets` 1.5.0 over all current publication candidates. Its only findings
  were expected high-entropy hexadecimal model revisions, artifact hashes, and
  token-window hashes; these were manually reviewed as non-secret provenance.
- Largest historical blob: 2,534,602 bytes (the earlier social card). No blob is
  larger than 5 MB.
- No model checkpoints, tokenizer bundles, Hugging Face caches, full dataset
  files, virtual environments, `.env` files, shell histories, debug logs, or
  editor metadata are published.
- Removed absolute harness-install paths from smoke and substantive capability
  artifacts; portable `lm_eval/tasks/...` identifiers retain the same provenance.
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
  user. The PG-19 parquet mirror is not included. MMLU-Pro, GSM8K, IFEval,
  HellaSwag, and lm-eval license notices are recorded in
  `THIRD_PARTY_NOTICES.md`.

## Reproducibility and site checks

- The perplexity and capability summaries regenerate from their raw lane files.
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
