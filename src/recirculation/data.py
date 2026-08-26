"""Create and validate the immutable evaluation-window manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
from dataclasses import dataclass
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from .constants import (
    CONFIRMATORY_NUM_DOCUMENTS,
    CONFIRMATORY_SELECTION_ID,
    CONFIRMATORY_START_DOCUMENT,
    CONTEXT_LENGTH,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    EXPLORATORY_SELECTION_ID,
    EXPLORATORY_START_DOCUMENT,
    MODEL_ID,
    MODEL_REVISION,
    NUM_DOCUMENTS,
    WINDOWS_PER_DOCUMENT,
)


@dataclass(frozen=True)
class Selection:
    selection_id: str
    start_document: int
    document_count: int
    selection_rule: str


SELECTIONS = {
    EXPLORATORY_SELECTION_ID: Selection(
        selection_id=EXPLORATORY_SELECTION_ID,
        start_document=EXPLORATORY_START_DOCUMENT,
        document_count=NUM_DOCUMENTS,
        selection_rule=(
            "first five non-overlapping complete 1024-token windows from each "
            "of the first two PG-19 test documents in dataset order"
        ),
    ),
    CONFIRMATORY_SELECTION_ID: Selection(
        selection_id=CONFIRMATORY_SELECTION_ID,
        start_document=CONFIRMATORY_START_DOCUMENT,
        document_count=CONFIRMATORY_NUM_DOCUMENTS,
        selection_rule=(
            "first five non-overlapping complete 1024-token windows from each "
            "of PG-19 test documents 2 through 9 inclusive in dataset order"
        ),
    ),
}


def token_sha256(token_ids: list[int]) -> str:
    payload = b"".join(struct.pack("<I", token_id) for token_id in token_ids)
    return hashlib.sha256(payload).hexdigest()


def prepare_manifest(
    output: Path, selection_id: str = EXPLORATORY_SELECTION_ID
) -> dict:
    selection = SELECTIONS[selection_id]
    dataset = load_dataset(
        DATASET_ID,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )

    windows = []
    for document_index in range(
        selection.start_document,
        selection.start_document + selection.document_count,
    ):
        record = dataset[document_index]
        token_ids = tokenizer.encode(record["text"], add_special_tokens=True)
        required = CONTEXT_LENGTH * WINDOWS_PER_DOCUMENT
        if len(token_ids) < required:
            raise RuntimeError(
                f"document {document_index} has fewer than {required} tokens"
            )
        for window_index in range(WINDOWS_PER_DOCUMENT):
            start = window_index * CONTEXT_LENGTH
            window = token_ids[start : start + CONTEXT_LENGTH]
            windows.append(
                {
                    "document_index": document_index,
                    "short_book_title": record["short_book_title"],
                    "publication_date": record["publication_date"],
                    "url": record["url"],
                    "document_token_count": len(token_ids),
                    "window_index_within_document": window_index,
                    "token_count": len(window),
                    "predicted_token_count": len(window) - 1,
                    "token_ids_sha256_le_u32": token_sha256(window),
                    "token_ids": window,
                }
            )

    manifest = {
        "schema_version": 2,
        "selection_id": selection.selection_id,
        "selection_rule": selection.selection_rule,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_add_special_tokens": True,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "context_length": CONTEXT_LENGTH,
        "score_rule": "causal next-token NLL at positions 1..1023 per window",
        "total_predicted_tokens": sum(w["predicted_token_count"] for w in windows),
        "python": platform.python_version(),
        "windows": windows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    # The original schema predates selection IDs. Its missing value is accepted
    # only as the already-committed exploratory selection.
    selection_id = manifest.get("selection_id", EXPLORATORY_SELECTION_ID)
    if selection_id not in SELECTIONS:
        raise ValueError("manifest selection_id is not a locked experiment selection")
    selection = SELECTIONS[selection_id]
    if manifest.get("selection_rule") != selection.selection_rule:
        raise ValueError("manifest selection_rule differs from the locked selection")
    expected_top_level = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "tokenizer_add_special_tokens": True,
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "context_length": CONTEXT_LENGTH,
    }
    for field, expected in expected_top_level.items():
        if manifest.get(field) != expected:
            raise ValueError(f"manifest {field} differs from experiment pin")

    windows = manifest.get("windows", [])
    expected_count = selection.document_count * WINDOWS_PER_DOCUMENT
    if len(windows) != expected_count:
        raise ValueError(f"manifest must contain exactly {expected_count} windows")
    expected_order = [
        (document_index, window_index)
        for document_index in range(
            selection.start_document,
            selection.start_document + selection.document_count,
        )
        for window_index in range(WINDOWS_PER_DOCUMENT)
    ]
    actual_order = [
        (window.get("document_index"), window.get("window_index_within_document"))
        for window in windows
    ]
    if actual_order != expected_order:
        raise ValueError(
            "manifest document/window order differs from the locked selection"
        )

    predicted_total = 0
    for window in windows:
        if len(window["token_ids"]) != CONTEXT_LENGTH:
            raise ValueError("manifest contains a partial window")
        if window.get("token_count") != CONTEXT_LENGTH:
            raise ValueError("manifest window token_count is inconsistent")
        if window.get("predicted_token_count") != CONTEXT_LENGTH - 1:
            raise ValueError("manifest predicted_token_count is inconsistent")
        if token_sha256(window["token_ids"]) != window["token_ids_sha256_le_u32"]:
            raise ValueError("manifest token hash mismatch")
        predicted_total += window["predicted_token_count"]
    if manifest.get("total_predicted_tokens") != predicted_total:
        raise ValueError("manifest total_predicted_tokens is inconsistent")
    return manifest


def assert_manifests_disjoint(first_path: Path, second_path: Path) -> None:
    first = load_manifest(first_path)
    second = load_manifest(second_path)
    first_documents = {window["document_index"] for window in first["windows"]}
    second_documents = {window["document_index"] for window in second["windows"]}
    if overlap := first_documents & second_documents:
        raise ValueError(f"manifest document overlap: {sorted(overlap)}")
    first_hashes = {window["token_ids_sha256_le_u32"] for window in first["windows"]}
    second_hashes = {window["token_ids_sha256_le_u32"] for window in second["windows"]}
    if overlap := first_hashes & second_hashes:
        raise ValueError(f"manifest token-window hash overlap: {sorted(overlap)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/pg19_windows.json")
    )
    parser.add_argument(
        "--selection",
        choices=sorted(SELECTIONS),
        default=EXPLORATORY_SELECTION_ID,
    )
    args = parser.parse_args()
    manifest = prepare_manifest(args.output, args.selection)
    print(json.dumps({k: v for k, v in manifest.items() if k != "windows"}, indent=2))
    for window in manifest["windows"]:
        print(
            f"document={window['document_index']} "
            f"window={window['window_index_within_document']} "
            f"tokens={window['token_count']} "
            f"sha256={window['token_ids_sha256_le_u32']}"
        )


if __name__ == "__main__":
    main()
