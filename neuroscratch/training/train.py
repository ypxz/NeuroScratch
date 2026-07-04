"""Training entrypoint."""

from __future__ import annotations

from typing import Any


def train(**kwargs: Any) -> None:
    raise NotImplementedError("Training loop will be implemented by a later task.")


def main() -> None:
    raise NotImplementedError("CLI training entrypoint will be implemented by a later task.")


if __name__ == "__main__":
    main()
