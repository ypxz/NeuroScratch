"""Loss functions."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


class CrossEntropyLoss:
    """Mean categorical cross-entropy on probabilities with `-(1/N) log p_y`."""

    def __init__(self) -> None:
        """Initialize the cached probabilities and integer targets."""
        self._probs: Array | None = None
        self._targets: Labels | None = None

    def forward(self, probs: Array, targets: Labels) -> float:
        """Compute `-mean(log p_correct)` for class-index targets."""
        self._probs = probs
        self._targets = targets
        batch = targets.shape[0]
        return float(
            np.mean(-np.log(probs[np.arange(batch), targets] + 1e-12))
        )

    def backward(self) -> Array:
        """Return `dL/dp = -(1/N) * (y / p)` for the cached probability targets."""
        if self._probs is None or self._targets is None:
            raise RuntimeError("forward must be called before backward")
        batch = self._targets.shape[0]
        grad = np.zeros_like(self._probs)
        grad[np.arange(batch), self._targets] = -1.0 / (
            self._probs[np.arange(batch), self._targets] + 1e-12
        )
        return cast(Array, grad / batch)


class SoftmaxCrossEntropy:
    """Stable softmax plus cross-entropy, whose gradient simplifies to `(p - y)/N`."""

    def __init__(self) -> None:
        """Initialize the cached logits-derived probabilities and targets."""
        self._probs: Array | None = None
        self._targets: Labels | None = None

    def forward(self, logits: Array, targets: Labels) -> float:
        """Compute `-mean(log softmax(logits)_correct)` using a max-shifted softmax."""
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self._probs = probs
        self._targets = targets
        batch = targets.shape[0]
        return float(np.mean(-np.log(probs[np.arange(batch), targets] + 1e-12)))

    def backward(self) -> Array:
        """Return `(softmax(logits) - onehot(targets)) / N`, the combined softmax-CE gradient."""
        if self._probs is None or self._targets is None:
            raise RuntimeError("forward must be called before backward")
        batch = self._targets.shape[0]
        grad = self._probs.copy()
        grad[np.arange(batch), self._targets] -= 1.0
        return cast(Array, grad / batch)
