"""Network containers."""

from __future__ import annotations

import math
from typing import Protocol, cast, runtime_checkable

import numpy as np
import numpy.typing as npt

from .activations import ReLU
from .layers import Dense

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


class Layer(Protocol):
    """A differentiable mapping with a forward pass and a vector-Jacobian backward pass."""

    def forward(self, x: Array) -> Array:
        """Compute the layer output for a mini-batch."""
        ...

    def backward(self, dout: Array) -> Array:
        """Propagate `dL/dout` to `dL/din` using the layer Jacobian transpose."""
        ...


@runtime_checkable
class Parametric(Protocol):
    """A layer that exposes learnable parameters and their matching gradients."""

    def params(self) -> list[Array]:
        """Return mutable parameter arrays in optimization order."""
        ...

    def grads(self) -> list[Array]:
        """Return gradients aligned one-to-one with `params()`."""
        ...


class Sequential:
    """A left-to-right composition of layers with reverse-mode backpropagation."""

    def __init__(self, layers: list[Layer]) -> None:
        """Store the layer sequence in forward order."""
        self.layers = layers

    def forward(self, x: Array) -> Array:
        """Compute the function composition `f_n(...f_2(f_1(x)))`."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: Array) -> Array:
        """Propagate gradients from the final output back through the layer list in reverse."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def params(self) -> list[Array]:
        """Concatenate every parameter tensor from layers that expose learnable state."""
        params: list[Array] = []
        for layer in self.layers:
            if isinstance(layer, Parametric):
                params.extend(layer.params())
        return params

    def grads(self) -> list[Array]:
        """Concatenate every gradient tensor in the same order as `params()`."""
        grads: list[Array] = []
        for layer in self.layers:
            if isinstance(layer, Parametric):
                grads.extend(layer.grads())
        return grads

    def predict(self, x: Array) -> Labels:
        """Return `argmax(logits)` per row, which matches `argmax(softmax(logits))`."""
        return cast(Labels, np.argmax(self.forward(x), axis=1).astype(np.int64, copy=False))


def build_mnist_network(seed: int = 42) -> Sequential:
    """Build the canonical 784-256-128-10 MLP with He initialization before ReLUs."""
    rng = np.random.default_rng(seed)
    layers: list[Layer] = [
        Dense(784, 256, std=math.sqrt(2.0 / 784.0), rng=rng),
        ReLU(),
        Dense(256, 128, std=math.sqrt(2.0 / 256.0), rng=rng),
        ReLU(),
        Dense(128, 10, std=math.sqrt(1.0 / 128.0), rng=rng),
    ]
    return Sequential(layers)


Network = Sequential
