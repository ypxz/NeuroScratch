"""Network containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

Array = np.ndarray


class Layer(Protocol):
    def forward(self, x: Array) -> Array: ...

    def backward(self, dout: Array) -> Array: ...


@dataclass
class Sequential:
    layers: list[Layer] = field(default_factory=list)

    def forward(self, x: Array) -> Array:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: Array) -> Array:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> list[Array]:
        return []

    def grads(self) -> list[Array]:
        return []


Network = Sequential
