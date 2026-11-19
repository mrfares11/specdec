#!/usr/bin/env python
"""CLI: Phase 0 verifier latency profile (see src/profiling/verifier_latency.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.config import parse_cli
from src.profiling.verifier_latency import run


def main() -> None:
    run(parse_cli("Profile verifier forward latency vs query-token count"))


if __name__ == "__main__":
    main()
