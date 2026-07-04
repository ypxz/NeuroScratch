"""PyTorch reference training entrypoint for the canonical MNIST MLP."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from neuroscratch.training.data import load_mnist

from .model import ReferenceMLP, build_reference_model

try:
    import torch
except ImportError as exc:  # pragma: no cover - import guard for non-reference envs
    raise ImportError(
        "neuroscratch.reference.train requires the optional reference dependencies"
    ) from exc


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
    best_epoch: int


def _iterate_minibatches(num_examples: int, batch_size: int) -> range:
    """Yield minibatch start indices for a contiguous slice-based loop."""
    return range(0, num_examples, batch_size)


def _evaluate(
    model: ReferenceMLP,
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Compute mean cross-entropy and accuracy for a split."""
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    total_loss = 0.0
    total_correct = 0
    total_examples = int(x.shape[0])
    model.eval()
    with torch.inference_mode():
        for start in _iterate_minibatches(total_examples, batch_size):
            stop = min(start + batch_size, total_examples)
            xb = torch.as_tensor(x[start:stop], dtype=torch.float32)
            yb = torch.as_tensor(y[start:stop], dtype=torch.long)
            logits = model(xb)
            loss = criterion(logits, yb)
            predictions = torch.argmax(logits, dim=1)
            total_loss += float(loss.detach().cpu()) * int(xb.shape[0])
            total_correct += int((predictions == yb).sum().item())
    return total_loss / total_examples, total_correct / total_examples


def _write_metrics(
    path: Path,
    *,
    hyperparams: dict[str, float | int | str],
    history: list[EpochMetrics],
    test_accuracy: float,
) -> None:
    """Serialize the reference metrics.json payload."""
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
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _save_checkpoint(path: Path, model: ReferenceMLP) -> None:
    """Persist the reference model's state dict for downstream analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def train(
    *,
    data_dir: str | Path = "data",
    checkpoint_path: str | Path = "checkpoints/mnist_reference.pt",
    metrics_path: str | Path = "reference/metrics_reference.json",
    seed: int = 42,
    batch_size: int = 64,
    epochs: int = 15,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> TrainResult:
    """Train the PyTorch reference model on the canonical MNIST split."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_mnist(data_dir)
    model = build_reference_model(seed=seed)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(beta1, beta2),
        eps=eps,
    )
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    rng = np.random.default_rng(seed)
    history: list[EpochMetrics] = []
    best_val_accuracy = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        model.train()
        permutation = rng.permutation(int(splits.x_train.shape[0]))
        for start in _iterate_minibatches(permutation.shape[0], batch_size):
            batch_indices = permutation[start : start + batch_size]
            xb = torch.as_tensor(splits.x_train[batch_indices], dtype=torch.float32)
            yb = torch.as_tensor(splits.y_train[batch_indices], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        train_loss, train_acc = _evaluate(
            model,
            splits.x_train,
            splits.y_train,
            batch_size=batch_size,
        )
        val_loss, val_acc = _evaluate(
            model,
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
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    _, test_acc = _evaluate(
        model,
        splits.x_test,
        splits.y_test,
        batch_size=batch_size,
    )
    _save_checkpoint(Path(checkpoint_path), model)
    _write_metrics(
        Path(metrics_path),
        hyperparams={
            "optimizer": "adam",
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "eps": eps,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "architecture": "784-256-128-10",
            "framework": "pytorch-reference",
        },
        history=history,
        test_accuracy=test_acc,
    )
    return TrainResult(history=history, test_accuracy=test_acc, best_epoch=best_epoch)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `python -m neuroscratch.reference.train`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-path", default="checkpoints/mnist_reference.pt")
    parser.add_argument("--metrics-path", default="reference/metrics_reference.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args(argv)
    train(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        metrics_path=args.metrics_path,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
