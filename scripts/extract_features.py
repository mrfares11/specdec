#!/usr/bin/env python
"""CLI: extract target features into resumable shards (see src/data/extract_features.py).

Run under tmux on the pod; use --set data.dry_run=true for the size estimate only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import parse_cli
from src.data.extract_features import run


def main() -> None:
    run(parse_cli("Extract target hidden states into feature shards"))


if __name__ == "__main__":
    main()
