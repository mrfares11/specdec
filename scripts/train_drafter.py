#!/usr/bin/env python
"""CLI: train the drafter (see src/train/train_drafter.py). Run under tmux."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import parse_cli
from src.train.train_drafter import run


def main() -> None:
    run(parse_cli("Train the drafter on extracted feature shards"))


if __name__ == "__main__":
    main()
