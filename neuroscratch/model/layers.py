"""Layer primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Array = np.ndarray


@dataclass
class Dense:
    """Fully connected layer stub."""

    in_features: int
    out_features: int
    W: Array = field(init=False, repr=False)
    b: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.W = np.zeros((self.in_features, self.out_features), dtype=np.float64)
        self.b = np.zeros((self.out_features,), dtype=np.float64)

    def forward(self, x: Array) -> Array:
        return x @ self.W + self.b

    def backward(self, dout: Array) -> Array:
        return dout
