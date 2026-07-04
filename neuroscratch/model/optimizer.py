"""Optimizers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass
class SGD:
    """Stochastic gradient descent with optional momentum velocity updates."""

    lr: float = 1e-3
    momentum: float = 0.0

    def __post_init__(self) -> None:
        """Initialize the lazily allocated momentum buffers."""
        self._velocity: list[Array] | None = None

    def step(self, params: list[Array], grads: list[Array]) -> None:
        """Apply `v = m v - lr g`, `param += v` when `m > 0`, else `param -= lr g`."""
        if self.momentum > 0.0:
            if self._velocity is None:
                self._velocity = [np.zeros_like(param) for param in params]
            for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
                self._velocity[index] = self.momentum * self._velocity[index] - self.lr * grad
                param += self._velocity[index]
            return
        for param, grad in zip(params, grads, strict=True):
            param -= self.lr * grad


@dataclass
class Adam:
    """Adam with first/second moments and bias correction on each parameter tensor."""

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        """Initialize the lazily allocated moment buffers and timestep."""
        self._m: list[Array] | None = None
        self._v: list[Array] | None = None
        self._t = 0

    def step(self, params: list[Array], grads: list[Array]) -> None:
        """Update `m` and `v`, bias-correct them, then apply the Adam in-place parameter step."""
        self._t += 1
        if self._m is None or self._v is None:
            self._m = [np.zeros_like(param) for param in params]
            self._v = [np.zeros_like(param) for param in params]
        beta1_correction = 1.0 - self.beta1**self._t
        beta2_correction = 1.0 - self.beta2**self._t
        for index, (param, grad) in enumerate(zip(params, grads, strict=True)):
            self._m[index] = self.beta1 * self._m[index] + (1.0 - self.beta1) * grad
            self._v[index] = self.beta2 * self._v[index] + (1.0 - self.beta2) * (grad * grad)
            m_hat = self._m[index] / beta1_correction
            v_hat = self._v[index] / beta2_correction
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
