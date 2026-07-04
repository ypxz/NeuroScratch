from __future__ import annotations

from neuroscratch.training.gradient_check import (
    dense_gradient_error,
    max_relative_gradient_error,
    relu_gradient_error,
    softmax_cross_entropy_gradient_error,
)


def test_dense_gradient_error_is_small() -> None:
    """Finite differences should match `dW = x^T dz` and `db = Σ dz`."""
    assert dense_gradient_error() < 1e-6


def test_relu_gradient_error_is_small() -> None:
    """Finite differences should match the ReLU mask `1[x > 0]`."""
    assert relu_gradient_error() < 1e-6


def test_softmax_cross_entropy_gradient_error_is_small() -> None:
    """Finite differences should match the combined `(softmax - y)/N` gradient."""
    assert softmax_cross_entropy_gradient_error() < 1e-6


def test_tiny_network_gradient_error_is_small() -> None:
    """The full Dense-ReLU-Dense network should be numerically consistent."""
    assert max_relative_gradient_error() < 1e-6
