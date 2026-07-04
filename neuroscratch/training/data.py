"""MNIST loading helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class MnistSplits:
    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array
    x_test: Array
    y_test: Array


def load_mnist() -> MnistSplits:
    raise NotImplementedError("MNIST loading will be implemented by a later task.")
