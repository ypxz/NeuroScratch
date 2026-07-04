"""Training utilities for NeuroScratch."""

from .data import MnistSplits, load_mnist
from .gradient_check import (
    dense_gradient_error,
    max_relative_gradient_error,
    relu_gradient_error,
    softmax_cross_entropy_gradient_error,
)
from .train import TrainResult, load_checkpoint, train

__all__ = [
    "MnistSplits",
    "TrainResult",
    "dense_gradient_error",
    "load_checkpoint",
    "load_mnist",
    "max_relative_gradient_error",
    "relu_gradient_error",
    "softmax_cross_entropy_gradient_error",
    "train",
]
