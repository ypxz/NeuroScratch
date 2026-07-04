"""Finite-difference gradient checks for the from-scratch NumPy model."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from neuroscratch.model import Dense, ReLU, Sequential, SoftmaxCrossEntropy

Array = npt.NDArray[np.float64]
Labels = npt.NDArray[np.int64]


def _max_relative_error(numerical: Array, analytical: Array) -> float:
    """Compute `max |g_num - g_ana| / max(1e-12, |g_num| + |g_ana|)` over entries."""
    denom = np.maximum(1e-12, np.abs(numerical) + np.abs(analytical))
    return float(np.max(np.abs(numerical - analytical) / denom))


def _finite_difference(
    evaluate: Callable[[], float],
    parameter: Array,
    eps: float,
) -> Array:
    """Approximate `dL/dθ` with central differences `[(L+ - L-) / (2ε)]`."""
    grad = np.zeros_like(parameter)
    for index in np.ndindex(parameter.shape):
        original = float(parameter[index])
        parameter[index] = original + eps
        loss_plus = evaluate()
        parameter[index] = original - eps
        loss_minus = evaluate()
        parameter[index] = original
        grad[index] = (loss_plus - loss_minus) / (2.0 * eps)
    return grad


def dense_gradient_error(seed: int = 0) -> float:
    """Check `dL/dW = x^T dL/dz`, `dL/db = Σ dL/dz`, and `dL/dx = dL/dz W^T` for Dense."""
    rng = np.random.default_rng(seed)
    dense = Dense(3, 2, std=0.2, rng=rng)
    x = rng.normal(size=(4, 3)).astype(np.float64)
    dout = rng.normal(size=(4, 2)).astype(np.float64)

    def loss() -> float:
        output = dense.forward(x)
        return float(np.sum(output * dout))

    dense.forward(x)
    dense.backward(dout)
    numerical_w = _finite_difference(loss, dense.W, 1e-6)
    numerical_b = _finite_difference(loss, dense.b, 1e-6)
    return max(
        _max_relative_error(numerical_w, dense.dW),
        _max_relative_error(numerical_b, dense.db),
    )


def relu_gradient_error(seed: int = 0) -> float:
    """Check the ReLU derivative `1[x > 0]` against central differences."""
    rng = np.random.default_rng(seed)
    relu = ReLU()
    x = rng.normal(size=(4, 5)).astype(np.float64)
    dout = rng.normal(size=(4, 5)).astype(np.float64)

    def loss() -> float:
        output = relu.forward(x)
        return float(np.sum(output * dout))

    relu.forward(x)
    analytical = relu.backward(dout)
    numerical = _finite_difference(loss, x, 1e-6)
    return _max_relative_error(numerical, analytical)


def softmax_cross_entropy_gradient_error(seed: int = 0) -> float:
    """Check the combined softmax-plus-CE gradient `(softmax(logits) - y)/N`."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(3, 4)).astype(np.float64)
    targets = np.array([0, 2, 1], dtype=np.int64)
    loss = SoftmaxCrossEntropy()

    def evaluate() -> float:
        return loss.forward(logits, targets)

    loss.forward(logits, targets)
    analytical = loss.backward()
    numerical = _finite_difference(evaluate, logits, 1e-6)
    return _max_relative_error(numerical, analytical)


def max_relative_gradient_error(seed: int = 0) -> float:
    """Check a tiny Dense-ReLU-Dense network against finite differences on all params."""
    rng = np.random.default_rng(seed)
    network = Sequential(
        [
            Dense(4, 5, std=0.2, rng=rng),
            ReLU(),
            Dense(5, 3, std=0.2, rng=rng),
        ]
    )
    x = rng.normal(size=(3, 4)).astype(np.float64)
    targets = np.array([0, 2, 1], dtype=np.int64)
    loss = SoftmaxCrossEntropy()

    def evaluate() -> float:
        logits = network.forward(x)
        return loss.forward(logits, targets)

    logits = network.forward(x)
    loss.forward(logits, targets)
    network.backward(loss.backward())

    worst = 0.0
    for parameter, gradient in zip(network.params(), network.grads(), strict=True):
        numerical = _finite_difference(evaluate, parameter, 1e-6)
        worst = max(worst, _max_relative_error(numerical, gradient))
    return worst
