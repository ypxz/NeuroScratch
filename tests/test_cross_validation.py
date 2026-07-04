from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")


def _load_test_accuracy(path: Path) -> float:
    """Read the final test accuracy from a metrics JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return float(payload["test_accuracy"])


def test_reference_and_from_scratch_agree_within_one_point() -> None:
    """The two implementations should remain within 1 percentage point."""
    repo_root = Path(__file__).resolve().parents[1]
    scratch_path = repo_root / "training" / "metrics_reference.json"
    reference_path = repo_root / "reference" / "metrics_reference.json"
    assert scratch_path.exists(), f"missing metrics file: {scratch_path}"
    assert reference_path.exists(), f"missing metrics file: {reference_path}"

    scratch_accuracy = _load_test_accuracy(scratch_path)
    reference_accuracy = _load_test_accuracy(reference_path)
    assert abs(scratch_accuracy - reference_accuracy) <= 0.01
