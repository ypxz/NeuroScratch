"""Activation functions."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]


class ReLU:
    """Rectifier with derivative `1[x > 0]` applied elementwise."""

    def __init__(self) -> None:
        """Initialize the cached activation mask."""
        self._mask: Mask | None = None

    def forward(self, x: Array) -> Array:
        """Compute `max(0, x)` and cache the positive-input mask for backprop."""
        self._mask = x > 0.0
        return np.maximum(0.0, x)

    def backward(self, dout: Array) -> Array:
        """Multiply `dL/dout` by the ReLU derivative, which is `1` where `x > 0`."""
        if self._mask is None:
            raise RuntimeError("forward must be called before backward")
        return dout * self._mask


class Softmax:
    """Row-wise softmax with the full Jacobian-vector product for backprop."""

    def __init__(self) -> None:
        """Initialize the cached softmax probabilities."""
        self._probs: Array | None = None

    def forward(self, x: Array) -> Array:
        """Compute a numerically stable softmax by subtracting each row maximum."""
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        self._probs = probs
        return cast(Array, probs)

    def backward(self, dout: Array) -> Array:
        """Apply `J_ij = s_i(δ_ij - s_j)` row by row, i.e. `ds = s * (dout - <dout,s>)`."""
        if self._probs is None:
            raise RuntimeError("forward must be called before backward")
        dot = np.sum(dout * self._probs, axis=1, keepdims=True)
        return cast(Array, self._probs * (dout - dot))
