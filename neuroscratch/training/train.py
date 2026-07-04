"""Training entrypoint for the from-scratch MNIST model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import numpy.typing as npt

from neuroscratch.model import Adam, Dense, Sequential, SoftmaxCrossEntropy, build_mnist_network

from .data import load_mnist
from .gradient_check import max_relative_gradient_error

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


@dataclass(frozen=True)
class EpochMetrics:
    """Scalar training metrics for one epoch."""

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


@dataclass(frozen=True)
class TrainResult:
    """Return value containing the training history and final evaluation metrics."""

    history: list[EpochMetrics]
    test_accuracy: float
    gradient_check: float


def _timestamp_run_name() -> str:
    """Create a filesystem-safe UTC timestamp for run directories."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parameter_snapshot(params: list[Array]) -> list[Array]:
    """Copy parameters so the best checkpoint can be restored after training."""
    return [param.copy() for param in params]


def _restore_parameters(params: list[Array], snapshot: list[Array]) -> None:
    """Overwrite live parameter arrays with a saved snapshot elementwise."""
    for param, saved in zip(params, snapshot, strict=True):
        param[...] = saved


def _dense_layers(network: Sequential) -> list[Dense]:
    """Collect the canonical Dense layers in forward order."""
    return [layer for layer in network.layers if isinstance(layer, Dense)]


def _apply_checkpoint(network: Sequential, path: Path) -> None:
    """Load checkpoint arrays from disk into the canonical Dense layers."""
    with np.load(path, allow_pickle=False) as data:
        dense_layers = _dense_layers(network)
        if len(dense_layers) != 3:
            raise ValueError("canonical network must have exactly three Dense layers")
        for index, layer in enumerate(dense_layers):
            layer.W[...] = data[f"W{index}"]
            layer.b[...] = data[f"b{index}"]


def load_checkpoint(path: str | Path) -> Sequential:
    """Rebuild the canonical network and load Dense weights from an `.npz` checkpoint."""
    checkpoint_path = Path(path)
    network = build_mnist_network(seed=42)
    _apply_checkpoint(network, checkpoint_path)
    return network


def _save_checkpoint(
    path: Path,
    network: Sequential,
    *,
    epoch: int,
    val_accuracy: float,
) -> None:
    """Persist the three Dense layers so the best validation model can be reloaded."""
    path.parent.mkdir(parents=True, exist_ok=True)
    dense_layers = _dense_layers(network)
    if len(dense_layers) != 3:
        raise ValueError("canonical network must have exactly three Dense layers")
    np.savez(
        path,
        W0=dense_layers[0].W,
        b0=dense_layers[0].b,
        W1=dense_layers[1].W,
        b1=dense_layers[1].b,
        W2=dense_layers[2].W,
        b2=dense_layers[2].b,
        epoch=np.array(epoch, dtype=np.int64),
        val_accuracy=np.array(val_accuracy, dtype=np.float64),
    )


def _iterate_minibatches(indices: Labels, batch_size: int) -> Iterable[Labels]:
    """Yield contiguous slices of shuffled row indices for SGD-style minibatches."""
    for start in range(0, indices.shape[0], batch_size):
        yield indices[start : start + batch_size]


def _evaluate(
    network: Sequential,
    x: Array,
    y: Labels,
    *,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Compute mean cross-entropy and accuracy in batches without materializing all logits."""
    loss_fn = SoftmaxCrossEntropy()
    total_loss = 0.0
    total_correct = 0
    total_examples = x.shape[0]
    for start in range(0, total_examples, batch_size):
        stop = min(start + batch_size, total_examples)
        xb = x[start:stop]
        yb = y[start:stop]
        logits = network.forward(xb)
        batch_loss = loss_fn.forward(logits, yb)
        predictions = np.argmax(logits, axis=1)
        total_loss += batch_loss * xb.shape[0]
        total_correct += int(np.sum(predictions == yb))
    return total_loss / total_examples, total_correct / total_examples


def _write_metrics(
    path: Path,
    *,
    hyperparams: dict[str, float | int | str],
    history: list[EpochMetrics],
    test_accuracy: float,
    gradient_check: float,
) -> None:
    """Serialize the canonical metrics.json schema used by the shared contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "hyperparams": hyperparams,
        "history": [
            {
                "epoch": item.epoch,
                "train_loss": item.train_loss,
                "train_acc": item.train_acc,
                "val_loss": item.val_loss,
                "val_acc": item.val_acc,
            }
            for item in history
        ],
        "test_accuracy": test_accuracy,
        "gradient_check": {
            "max_relative_error": gradient_check,
            "tolerance": 1e-6,
        },
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def train(
    *,
    data_dir: str | Path = "data",
    checkpoint_path: str | Path = "checkpoints/mnist_from_scratch.npz",
    runs_dir: str | Path = "training/runs",
    run_name: str | None = None,
    seed: int = 42,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    write_artifacts: bool = True,
) -> TrainResult:
    """Train the canonical 784-256-128-10 network with Adam on the MNIST split.

    Each epoch shuffles the 54k-row training set with a deterministic RNG, updates the
    network using mini-batches, evaluates full train and validation metrics, and keeps
    the best-validation checkpoint for the final test-set report.
    """
    splits = load_mnist(data_dir)
    network = build_mnist_network(seed=seed)
    optimizer = Adam(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
    loss_fn = SoftmaxCrossEntropy()
    rng = np.random.default_rng(seed)
    best_snapshot = _parameter_snapshot(network.params())
    best_val_accuracy = -1.0
    best_epoch = 0
    checkpoint_file = Path(checkpoint_path)
    run_directory = Path(runs_dir) / (run_name if run_name is not None else _timestamp_run_name())
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        indices = rng.permutation(splits.x_train.shape[0])
        for batch_indices in _iterate_minibatches(indices, batch_size):
            xb = splits.x_train[batch_indices]
            yb = splits.y_train[batch_indices]
            logits = network.forward(xb)
            loss_fn.forward(logits, yb)
            dlogits = loss_fn.backward()
            network.backward(dlogits)
            optimizer.step(network.params(), network.grads())

        train_loss, train_acc = _evaluate(
            network,
            splits.x_train,
            splits.y_train,
            batch_size=batch_size,
        )
        val_loss, val_acc = _evaluate(
            network,
            splits.x_val,
            splits.y_val,
            batch_size=batch_size,
        )
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(metrics)
        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            best_snapshot = _parameter_snapshot(network.params())
            if write_artifacts:
                _save_checkpoint(checkpoint_file, network, epoch=epoch, val_accuracy=val_acc)

    if write_artifacts:
        best_network = load_checkpoint(checkpoint_file)
    else:
        best_network = build_mnist_network(seed=seed)
        _restore_parameters(best_network.params(), best_snapshot)

    _, test_accuracy = _evaluate(best_network, splits.x_test, splits.y_test, batch_size=batch_size)
    gradient_error = max_relative_gradient_error(seed=0)

    if write_artifacts:
        hyperparams: dict[str, float | int | str] = {
            "optimizer": "adam",
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "architecture": "784-256-128-10",
        }
        _write_metrics(
            run_directory / "metrics.json",
            hyperparams=hyperparams,
            history=history,
            test_accuracy=test_accuracy,
            gradient_check=gradient_error,
        )

    print(f"best_epoch={best_epoch} test_acc={test_accuracy:.4f} grad_check={gradient_error:.3e}")
    return TrainResult(history=history, test_accuracy=test_accuracy, gradient_check=gradient_error)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI parser exposing the canonical training hyperparameters."""
    parser = argparse.ArgumentParser(description="Train the NeuroScratch MNIST MLP.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-path", default="checkpoints/mnist_from_scratch.npz")
    parser.add_argument("--runs-dir", default="training/runs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="skip checkpoint and metrics.json writes",
    )
    return parser


def main() -> None:
    """Parse CLI flags and run canonical MNIST training."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        runs_dir=args.runs_dir,
        run_name=args.run_name,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        write_artifacts=not args.no_write_artifacts,
    )


if __name__ == "__main__":
    main()
