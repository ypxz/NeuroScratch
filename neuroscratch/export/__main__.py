"""CLI entrypoint: export the checkpointed network to ``web/weights.json``."""

from __future__ import annotations

from .weights import main

if __name__ == "__main__":
    main()
