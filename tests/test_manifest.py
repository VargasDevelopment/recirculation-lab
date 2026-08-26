import copy
import json
from pathlib import Path

import pytest

from recirculation.data import load_manifest


SOURCE = Path("experiments/pg19_windows.json")


def write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_committed_manifest_satisfies_all_locked_invariants() -> None:
    manifest = load_manifest(SOURCE)
    assert manifest["total_predicted_tokens"] == 10_230


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
