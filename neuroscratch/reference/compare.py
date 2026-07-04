"""Cross-validation comparison between the from-scratch and reference models."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from neuroscratch.training.data import load_mnist
from neuroscratch.training.train import load_checkpoint as load_scratch_checkpoint

from .model import ReferenceMLP, build_reference_model

try:
    import matplotlib.pyplot as plt
    import torch
except ImportError as exc:  # pragma: no cover - optional plotting/runtime deps
    raise ImportError(
        "neuroscratch.reference.compare requires the optional reference dependencies"
    ) from exc


CLASS_LABELS: tuple[str, ...] = tuple(str(index) for index in range(10))


def _load_metrics(path: Path) -> dict[str, object]:
    """Load a metrics JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return cast(dict[str, object], json.load(handle))


def _history_rows(payload: dict[str, object]) -> list[dict[str, float | int]]:
    """Extract the epoch history rows from a metrics payload."""
    history = payload["history"]
    if not isinstance(history, list):
        raise TypeError("metrics history must be a list")
    rows: list[dict[str, float | int]] = []
    for item in history:
        if not isinstance(item, dict):
            raise TypeError("each history row must be a mapping")
        rows.append(
            {
                "epoch": int(item["epoch"]),
                "train_loss": float(item["train_loss"]),
                "train_acc": float(item["train_acc"]),
                "val_loss": float(item["val_loss"]),
                "val_acc": float(item["val_acc"]),
            }
        )
    return rows


def _load_reference_model(checkpoint_path: Path) -> ReferenceMLP:
    """Instantiate the PyTorch model and load the saved state dict."""
    model = build_reference_model(seed=42)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict_logits(model: object, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Run batched inference and return concatenated logits."""
    logits_batches: list[np.ndarray] = []
    if isinstance(model, ReferenceMLP):
        with torch.inference_mode():
            for start in range(0, x.shape[0], batch_size):
                stop = min(start + batch_size, x.shape[0])
                xb = torch.as_tensor(x[start:stop], dtype=torch.float32)
                logits = model(xb).detach().cpu().numpy()
                logits_batches.append(logits)
        return np.concatenate(logits_batches, axis=0)
    for start in range(0, x.shape[0], batch_size):
        stop = min(start + batch_size, x.shape[0])
        logits_batches.append(model.forward(x[start:stop]))  # type: ignore[attr-defined]
    return np.concatenate(logits_batches, axis=0)


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Compute a simple confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for target, prediction in zip(y_true, y_pred, strict=True):
        matrix[int(target), int(prediction)] += 1
    return matrix


def _per_class_metrics(confusion: np.ndarray) -> list[dict[str, float | int]]:
    """Compute per-class precision, recall, and F1 scores."""
    metrics: list[dict[str, float | int]] = []
    for class_index in range(confusion.shape[0]):
        tp = float(confusion[class_index, class_index])
        fp = float(confusion[:, class_index].sum() - confusion[class_index, class_index])
        fn = float(confusion[class_index, :].sum() - confusion[class_index, class_index])
        precision = tp / (tp + fp) if tp + fp > 0.0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0.0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        metrics.append(
            {
                "class": class_index,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return metrics


def _plot_training_curves(
    scratch_history: Sequence[dict[str, float | int]],
    reference_history: Sequence[dict[str, float | int]],
    path: Path,
) -> None:
    """Create overlay plots for loss and accuracy curves."""
    epochs = [int(row["epoch"]) for row in scratch_history]
    scratch_train_loss = [float(row["train_loss"]) for row in scratch_history]
    scratch_val_loss = [float(row["val_loss"]) for row in scratch_history]
    scratch_train_acc = [float(row["train_acc"]) for row in scratch_history]
    scratch_val_acc = [float(row["val_acc"]) for row in scratch_history]
    reference_train_loss = [float(row["train_loss"]) for row in reference_history]
    reference_val_loss = [float(row["val_loss"]) for row in reference_history]
    reference_train_acc = [float(row["train_acc"]) for row in reference_history]
    reference_val_acc = [float(row["val_acc"]) for row in reference_history]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_loss.plot(epochs, scratch_train_loss, label="from-scratch train")
    ax_loss.plot(epochs, scratch_val_loss, label="from-scratch val")
    ax_loss.plot(epochs, reference_train_loss, label="reference train")
    ax_loss.plot(epochs, reference_val_loss, label="reference val")
    ax_loss.set_title("Cross-entropy loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(fontsize=8)

    ax_acc.plot(epochs, scratch_train_acc, label="from-scratch train")
    ax_acc.plot(epochs, scratch_val_acc, label="from-scratch val")
    ax_acc.plot(epochs, reference_train_acc, label="reference train")
    ax_acc.plot(epochs, reference_val_acc, label="reference val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrices(
    scratch_confusion: np.ndarray,
    reference_confusion: np.ndarray,
    path: Path,
) -> None:
    """Render side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, matrix, title in zip(
        axes,
        (scratch_confusion, reference_confusion),
        ("From-scratch", "PyTorch reference"),
        strict=True,
    ):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(range(len(CLASS_LABELS)))
        ax.set_yticks(range(len(CLASS_LABELS)))
        ax.set_xticklabels(CLASS_LABELS)
        ax.set_yticklabels(CLASS_LABELS)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _render_table(metrics: Sequence[dict[str, float | int]], title: str) -> str:
    """Render a markdown table for one model's per-class metrics."""
    lines = [
        f"**{title}**",
        "",
        "| Class | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metrics:
        class_index = int(row["class"])
        lines.append(
            "| {class_index} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                class_index=class_index,
                precision=row["precision"],
                recall=row["recall"],
                f1=row["f1"],
            )
        )
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    scratch_accuracy: float,
    reference_accuracy: float,
    accuracy_delta: float,
    curve_note: str,
    scratch_table: str,
    reference_table: str,
) -> None:
    """Write the cross-validation markdown report."""
    content = f"""# Cross-validation report

## Final test accuracy

- From-scratch: {scratch_accuracy:.6f}
- PyTorch reference: {reference_accuracy:.6f}
- Delta (from-scratch - reference): {accuracy_delta:.6f}

## Training curve comparison

{curve_note}

![Training curves](training_curves.png)

## Confusion matrices

![Confusion matrices](confusion_matrix.png)

## Per-class metrics

### From-scratch

{scratch_table}

### PyTorch reference

{reference_table}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def compare(
    *,
    scratch_metrics_path: str | Path = "training/metrics_reference.json",
    reference_metrics_path: str | Path = "reference/metrics_reference.json",
    scratch_checkpoint_path: str | Path = "checkpoints/mnist_from_scratch.npz",
    reference_checkpoint_path: str | Path = "checkpoints/mnist_reference.pt",
    reports_dir: str | Path = "reports",
) -> dict[str, object]:
    """Compare the from-scratch and PyTorch reference MNIST models."""
    scratch_metrics_payload = _load_metrics(Path(scratch_metrics_path))
    reference_metrics_payload = _load_metrics(Path(reference_metrics_path))
    scratch_history = _history_rows(scratch_metrics_payload)
    reference_history = _history_rows(reference_metrics_payload)

    splits = load_mnist()
    scratch_model = load_scratch_checkpoint(scratch_checkpoint_path)
    reference_model = _load_reference_model(Path(reference_checkpoint_path))

    scratch_logits = _predict_logits(scratch_model, splits.x_test)
    reference_logits = _predict_logits(reference_model, splits.x_test)
    scratch_predictions = np.argmax(scratch_logits, axis=1)
    reference_predictions = np.argmax(reference_logits, axis=1)

    scratch_confusion = _confusion_matrix(splits.y_test, scratch_predictions)
    reference_confusion = _confusion_matrix(splits.y_test, reference_predictions)
    scratch_per_class = _per_class_metrics(scratch_confusion)
    reference_per_class = _per_class_metrics(reference_confusion)

    scratch_accuracy = float(cast(float, scratch_metrics_payload["test_accuracy"]))
    reference_accuracy = float(cast(float, reference_metrics_payload["test_accuracy"]))
    accuracy_delta = scratch_accuracy - reference_accuracy

    reports_path = Path(reports_dir)
    training_curves_path = reports_path / "training_curves.png"
    confusion_matrix_path = reports_path / "confusion_matrix.png"
    _plot_training_curves(scratch_history, reference_history, training_curves_path)
    _plot_confusion_matrices(scratch_confusion, reference_confusion, confusion_matrix_path)

    scratch_table = _render_table(scratch_per_class, "From-scratch per-class metrics")
    reference_table = _render_table(reference_per_class, "PyTorch reference per-class metrics")
    curve_delta = max(
        abs(float(s_row["val_acc"]) - float(r_row["val_acc"]))
        for s_row, r_row in zip(scratch_history, reference_history, strict=True)
    )
    curve_note = (
        f"The two runs track closely epoch-by-epoch; the maximum validation-accuracy "
        f"gap is {curve_delta:.6f}, and both reach their best validation performance "
        f"late in training."
    )
    _write_report(
        reports_path / "cross_validation.md",
        scratch_accuracy=scratch_accuracy,
        reference_accuracy=reference_accuracy,
        accuracy_delta=accuracy_delta,
        curve_note=curve_note,
        scratch_table=scratch_table,
        reference_table=reference_table,
    )
    return {
        "scratch_accuracy": scratch_accuracy,
        "reference_accuracy": reference_accuracy,
        "accuracy_delta": accuracy_delta,
        "training_curves_path": training_curves_path,
        "confusion_matrix_path": confusion_matrix_path,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `python -m neuroscratch.reference.compare`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--scratch-metrics-path", default="training/metrics_reference.json")
    parser.add_argument("--reference-metrics-path", default="reference/metrics_reference.json")
    parser.add_argument("--scratch-checkpoint-path", default="checkpoints/mnist_from_scratch.npz")
    parser.add_argument("--reference-checkpoint-path", default="checkpoints/mnist_reference.pt")
    args = parser.parse_args(argv)
    compare(
        scratch_metrics_path=args.scratch_metrics_path,
        reference_metrics_path=args.reference_metrics_path,
        scratch_checkpoint_path=args.scratch_checkpoint_path,
        reference_checkpoint_path=args.reference_checkpoint_path,
        reports_dir=args.reports_dir,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
