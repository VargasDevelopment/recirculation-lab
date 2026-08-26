"""Narrow dataset-revision pinning for canonical harness task loaders."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import datasets

from .capability_constants import BENCHMARKS

DATASET_REVISIONS = {
    metadata["dataset_id"]: metadata["dataset_revision"]
    for metadata in BENCHMARKS.values()
}


@contextmanager
def pinned_benchmark_datasets() -> Iterator[None]:
    """Force built-in harness tasks onto the recorded Hub commits."""
    original = datasets.load_dataset

    def load_dataset(path, *args, **kwargs):
        revision = DATASET_REVISIONS.get(path)
        if revision is not None:
            supplied = kwargs.get("revision")
            if supplied is not None and supplied != revision:
                raise ValueError(
                    f"dataset revision mismatch for {path}: {supplied} != {revision}"
                )
            kwargs["revision"] = revision
        return original(path, *args, **kwargs)

    with patch.object(datasets, "load_dataset", side_effect=load_dataset):
        yield
