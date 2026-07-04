"""Parity regression tests for the exported NeuroScratch artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from neuroscratch.training.train import load_checkpoint


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable softmax for a single row."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def test_python_reproduces_exported_parity_fixture() -> None:
    """Ensure the canonical Python network matches the committed fixture."""
    fixtures_path = Path("web/parity/fixtures.json")
    with fixtures_path.open("r", encoding="utf-8") as handle:
        fixtures = json.load(handle)
    network = load_checkpoint("checkpoints/mnist_from_scratch.npz")
    for sample in fixtures["samples"]:
        x = np.array(sample["pixels"], dtype=np.float64) / 255.0
        logits = network.forward(x[None, :])
        probabilities = _softmax(logits)
        assert int(np.argmax(logits, axis=1)[0]) == sample["prediction"]
        assert np.allclose(probabilities[0], sample["probabilities"], atol=1e-6)


def test_node_parity_suite_if_available() -> None:
    """Run the browser parity test suite when Node is installed."""
    if shutil.which("node") is None:
        pytest.skip("node is not available")
    result = subprocess.run(
        ["node", "--test", "parity/parity.test.mjs"],
        cwd="web",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
