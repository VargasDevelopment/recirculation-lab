import copy
import json
from pathlib import Path

import pytest

from recirculation.data import assert_manifests_disjoint, load_manifest


SOURCE = Path("experiments/pg19_windows.json")
CONFIRMATORY = Path("experiments/pg19_validation_books_2_9.json")


def write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_committed_manifest_satisfies_all_locked_invariants() -> None:
    manifest = load_manifest(SOURCE)
    assert manifest["total_predicted_tokens"] == 10_230


def test_confirmatory_manifest_is_locked_and_disjoint() -> None:
    manifest = load_manifest(CONFIRMATORY)
    assert manifest["selection_id"] == "confirmatory_unseen_books_2_9_v1"
    assert manifest["total_predicted_tokens"] == 40_920
    assert len(manifest["windows"]) == 40
    assert {window["document_index"] for window in manifest["windows"]} == set(
        range(2, 10)
    )
    assert_manifests_disjoint(SOURCE, CONFIRMATORY)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("model_id", "wrong/model"),
        lambda value: value.__setitem__("tokenizer_add_special_tokens", False),
        lambda value: value.__setitem__("total_predicted_tokens", 1),
        lambda value: value["windows"].__setitem__(
            slice(0, 2), list(reversed(value["windows"][:2]))
        ),
        lambda value: value["windows"][0].__setitem__("predicted_token_count", 1),
    ],
)
def test_manifest_drift_is_rejected(tmp_path: Path, mutation) -> None:
    manifest = json.loads(SOURCE.read_text())
    mutation(manifest)
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, manifest))


def test_confirmatory_selection_drift_is_rejected(tmp_path: Path) -> None:
    manifest = json.loads(CONFIRMATORY.read_text())
    manifest["selection_id"] = "exploratory_books_0_1_v1"
    with pytest.raises(ValueError):
        load_manifest(write_manifest(tmp_path, manifest))
