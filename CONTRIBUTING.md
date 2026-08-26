# Contributing

Issues and focused pull requests are welcome. This is a research reproduction,
so experimental integrity takes priority over obtaining a positive result.

Before proposing a change:

1. Keep existing exploratory and confirmatory artifacts immutable.
2. Put new selection rules and mechanism settings in a committed protocol or
   structured config before evaluating outcomes.
3. Never commit model weights, full datasets, credentials, caches, or gated
   artifacts.
4. Run `uv sync --frozen` and `uv run pytest -q`.
5. Explain whether a change affects the ordinary path, recurrence semantics,
   token selection, scoring, or only documentation.

Large benchmark additions and new model families should be separate, locked
experiments rather than revisions of the recorded Gemma 3 results.
