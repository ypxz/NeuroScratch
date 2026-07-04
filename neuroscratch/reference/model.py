"""PyTorch reference model matching the canonical MNIST MLP."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn


class _ModuleBase:
    """Minimal module interface used for static analysis when torch is absent."""

    def __call__(self, x: Any) -> Any:
        """Execute the module forward pass."""
        return x

    def state_dict(self) -> dict[str, object]:
        """Return a module state mapping."""
        return {}

    def load_state_dict(self, state_dict: object) -> object:
        """Load a module state mapping."""
        return state_dict

    def train(self, mode: bool = True) -> _ModuleBase:
        """Switch the module into training mode."""
        del mode
        return self

    def eval(self) -> _ModuleBase:
        """Switch the module into evaluation mode."""
        return self

    def parameters(self) -> list[Any]:
        """Return learnable parameters."""
        return []


if not TYPE_CHECKING:
    _ModuleBase = nn.Module  # type: ignore[assignment]


class ReferenceMLP(_ModuleBase):
    """Canonical 784-256-128-10 multilayer perceptron for MNIST."""

    def __init__(self) -> None:
        """Construct the network and apply the shared initialization scheme."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply the contractually specified weight initialization."""
        layers = [module for module in self.network if isinstance(module, nn.Linear)]
        first, second, final = layers
        nn.init.kaiming_normal_(first.weight, nonlinearity="relu")
        nn.init.zeros_(first.bias)
        nn.init.kaiming_normal_(second.weight, nonlinearity="relu")
        nn.init.zeros_(second.bias)
        nn.init.normal_(final.weight, mean=0.0, std=math.sqrt(1.0 / 128.0))
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for a batch of flattened MNIST images."""
        return cast(torch.Tensor, self.network(x))


def build_reference_model(seed: int = 42) -> ReferenceMLP:
    """Build the canonical reference model with a deterministic seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return ReferenceMLP()
