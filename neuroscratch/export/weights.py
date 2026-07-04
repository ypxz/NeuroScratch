"""Weight export helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ExportableNetwork(Protocol):
    def params(self) -> list[object]: ...


def export_weights(network: ExportableNetwork, path: str | Path) -> None:
    raise NotImplementedError("Weight export will be implemented by a later task.")
