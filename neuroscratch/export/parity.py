"""Parity fixture generation for browser and Python inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from neuroscratch.training.data import load_mnist
from neuroscratch.training.train import load_checkpoint

Array = npt.NDArray[np.float64]


def _softmax(logits: Array) -> Array:
    """Compute a numerically stable softmax over a one-dimensional vector."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return cast(Array, exp_values / exp_values.sum())


def generate_parity_fixture(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    *,
    num_samples: int = 200,
) -> None:
    """Generate a deterministic parity fixture from the canonical checkpoint."""
    network = load_checkpoint(checkpoint_path)
    mnist = load_mnist(data_dir)
    x_test = mnist.x_test[:num_samples]
    y_test = mnist.y_test[:num_samples]
    pixels = np.rint(x_test * 255.0).astype(np.int64)
    samples: list[dict[str, object]] = []
    for pixel_row, label in zip(pixels, y_test, strict=True):
        x = pixel_row.astype(np.float64) / 255.0
        logits = network.forward(x[None, :])[0]
        probabilities = _softmax(logits)
        samples.append(
            {
                'pixels': pixel_row.tolist(),
                'label': int(label),
                'prediction': int(np.argmax(logits)),
                'probabilities': probabilities.tolist(),
            }
        )
    payload: dict[str, object] = {
        'num_samples': int(num_samples),
        'input_length': 784,
        'samples': samples,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def main() -> None:
    """Generate the canonical browser parity fixture."""
    parser = argparse.ArgumentParser(description='Generate NeuroScratch parity fixtures')
    parser.add_argument(
        '--checkpoint-path',
        default='checkpoints/mnist_from_scratch.npz',
        help='path to the canonical checkpoint',
    )
    parser.add_argument(
        '--data-dir',
        default='data',
        help='directory containing MNIST cache files',
    )
    parser.add_argument(
        '--output',
        default='web/parity/fixtures.json',
        help='output path for parity fixtures',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=200,
        help='number of test samples to export',
    )
    args = parser.parse_args()
    generate_parity_fixture(
        args.checkpoint_path,
        args.data_dir,
        args.output,
        num_samples=args.num_samples,
    )


if __name__ == '__main__':
    main()
