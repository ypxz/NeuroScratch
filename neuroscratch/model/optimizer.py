"""Optimizers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass
class SGD:
    lr: float = 1e-3

    def step(self, params: Iterable[Array], grads: Iterable[Array]) -> None:
        for param, grad in zip(params, grads, strict=True):
            param -= self.lr * grad


@dataclass
class Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def step(self, params: Iterable[Array], grads: Iterable[Array]) -> None:
        for param, grad in zip(params, grads, strict=True):
            param -= self.lr * grad
