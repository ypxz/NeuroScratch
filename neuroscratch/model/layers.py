"""Layer primitives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


class Dense:
    """Affine map `z = xW + b` and its matrix-calculus derivatives."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        std: float,
        rng: np.random.Generator,
    ):
        """Initialize `W ~ N(0, std)` and `b = 0` for the affine transform."""
        self.in_features = in_features
        self.out_features = out_features
        self.W: Array = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
        self.b: Array = np.zeros(out_features, dtype=np.float64)
        self._x: Array | None = None
        self.dW: Array = np.zeros_like(self.W)
        self.db: Array = np.zeros_like(self.b)

    def forward(self, x: Array) -> Array:
        """Compute `z = x @ W + b` for a mini-batch of inputs."""
        self._x = x
        return x @ self.W + self.b

    def backward(self, dout: Array) -> Array:
        """Propagate `dL/dz` through `z = xW + b`.

        The gradients are `dL/dW = x^T @ dL/dz`, `dL/db = sum(dL/dz)`, and
        `dL/dx = dL/dz @ W^T`.
        """
        if self._x is None:
            raise RuntimeError("forward must be called before backward")
        self.dW[...] = self._x.T @ dout
        self.db[...] = dout.sum(axis=0)
        return dout @ self.W.T

    def params(self) -> list[Array]:
        """Return the learnable parameters `[W, b]` in optimizer-update order."""
        return [self.W, self.b]

    def grads(self) -> list[Array]:
        """Return the matching gradients `[dW, db]` in the same order as `params()`."""
        return [self.dW, self.db]
