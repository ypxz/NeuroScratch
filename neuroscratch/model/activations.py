"""Activation functions."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


class ReLU:
    def forward(self, x: Array) -> Array:
        return np.maximum(0.0, x)

    def backward(self, dout: Array) -> Array:
        return dout


class Softmax:
    def forward(self, x: Array) -> Array:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, dout: Array) -> Array:
        return dout
