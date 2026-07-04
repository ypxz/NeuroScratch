"""Weight export helpers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from neuroscratch.training.train import load_checkpoint

Array = npt.NDArray[np.float64]


class ExportableNetwork(Protocol):
    """A network exposing learnable parameters in canonical order."""

    def params(self) -> list[Array]:
        """Return Dense parameters as `[W0, b0, W1, b1, W2, b2]`."""
        ...


def _validate_layer_shapes(layers: list[tuple[Array, Array]]) -> None:
    """Ensure the canonical Dense layer shapes are present."""
    expected = [((784, 256), (256,)), ((256, 128), (128,)), ((128, 10), (10,))]
    if len(layers) != len(expected):
        raise ValueError('canonical network must have exactly three Dense layers')
    for index, ((weights, bias), ((in_features, out_features), bias_shape)) in enumerate(
        zip(layers, expected, strict=True)
    ):
        if weights.shape != (in_features, out_features):
            raise ValueError(
                f"layer {index} weights must have shape "
                f"{(in_features, out_features)}, got {weights.shape}"
            )
        if bias.shape != bias_shape:
            raise ValueError(f'layer {index} bias must have shape {bias_shape}, got {bias.shape}')


def export_weights(
    network: ExportableNetwork,
    path: str | Path,
    *,
    test_accuracy: float = 0.0,
    trained_at: str | datetime | None = None,
) -> None:
    """Write the canonical weight export JSON schema to disk."""
    params = list(network.params())
    if len(params) != 6:
        raise ValueError('canonical network must expose exactly six parameter arrays')
    layers = [(params[index], params[index + 1]) for index in range(0, 6, 2)]
    _validate_layer_shapes(layers)
    timestamp = (
        datetime.now(timezone.utc).isoformat()
        if trained_at is None
        else trained_at.isoformat()
        if isinstance(trained_at, datetime)
        else trained_at
    )
    payload: dict[str, object] = {
        'format_version': 1,
        'input': {'shape': [28, 28], 'normalization': 'divide_by_255'},
        'architecture': [
            {'type': 'dense', 'in': 784, 'out': 256, 'activation': 'relu'},
            {'type': 'dense', 'in': 256, 'out': 128, 'activation': 'relu'},
            {'type': 'dense', 'in': 128, 'out': 10, 'activation': 'softmax'},
        ],
        'layers': [{'W': weights.tolist(), 'b': bias.tolist()} for weights, bias in layers],
        'meta': {
            'test_accuracy': float(test_accuracy),
            'trained_at': timestamp,
            'framework': 'neuroscratch-from-scratch',
        },
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'))
        handle.write('\n')


def _read_test_accuracy(metrics_path: Path) -> float:
    """Read the canonical test accuracy from a metrics reference file."""
    with metrics_path.open('r', encoding='utf-8') as handle:
        metrics = json.load(handle)
    return float(metrics['test_accuracy'])


def main() -> None:
    """Export the checkpointed network to `web/weights.json`."""
    parser = argparse.ArgumentParser(description='Export canonical NeuroScratch weights')
    parser.add_argument(
        '--checkpoint-path',
        default='checkpoints/mnist_from_scratch.npz',
        help='path to the canonical checkpoint',
    )
    parser.add_argument(
        '--metrics-path',
        default='training/metrics_reference.json',
        help='path to the metrics reference JSON',
    )
    parser.add_argument(
        '--output',
        default='web/weights.json',
        help='output path for the exported weights JSON',
    )
    args = parser.parse_args()
    network = load_checkpoint(args.checkpoint_path)
    test_accuracy = _read_test_accuracy(Path(args.metrics_path))
    export_weights(network, args.output, test_accuracy=test_accuracy)


if __name__ == '__main__':
    main()
