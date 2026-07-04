"""Loss functions."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


class CrossEntropyLoss:
    def forward(self, probs: Array, targets: Array) -> float:
        return float(np.mean(-np.log(probs[np.arange(len(targets)), targets] + 1e-12)))

    def backward(self, probs: Array, targets: Array) -> Array:
        grad = probs.copy()
        grad[np.arange(len(targets)), targets] -= 1.0
        return grad / targets.shape[0]
