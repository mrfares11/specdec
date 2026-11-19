#!/usr/bin/env python
"""CLI: benchmark speculative decoding across the K sweep (see src/eval/benchmark.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import parse_cli
from src.eval.benchmark import run


def main() -> None:
    run(parse_cli("Benchmark acceptance length and throughput across K"))


if __name__ == "__main__":
    main()
