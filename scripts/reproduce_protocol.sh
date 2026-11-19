#!/usr/bin/env bash
# Regenerate every protocol finding from committed raw data. No GPU, no pod.
#
# This is the reproducibility check a reviewer should be able to run: clone the
# repo, run this, and get the same tables the paper cites. Each analysis reads
# per-prompt rows and writes a markdown table; nothing is hand-entered.
#
#   bash scripts/reproduce_protocol.sh [--boot N]
#
# --boot lowers the bootstrap resample count for a quick pass (default 10000,
# which is what the reported intervals use).
set -uo pipefail
cd "$(dirname "$0")/.."

BOOT=10000
[ "${1:-}" = "--boot" ] && BOOT="${2:-2000}"

PY=${PYTHON:-python}
OUT=docs/tables
mkdir -p "$OUT"
fail=0

step() {          # name, then command
  local name="$1"; shift
  printf '\n=== %s ===\n' "$name"
  if "$@"; then :; else
    echo "  FAILED: $name"; fail=$((fail+1))
  fi
}

step "E1 correctness (float32 exactness, bf16 fidelity, margins)" \
  "$PY" scripts/e1_analyze.py --root results/e1 --out "$OUT/e1-summary.md"

step "E2 headline timing (Gate B, predeclared comparisons)" \
  "$PY" scripts/e2_analyze.py --root results/e2 --out "$OUT/e2-headline.md" --boot "$BOOT"

step "E3 interaction and transfer (Gate C)" \
  "$PY" scripts/e3_interaction.py --root results/e2 --out "$OUT/e3-interaction.md" --boot "$BOOT"

step "E4 host thread sensitivity (Gate D, incl. acceptance control)" \
  "$PY" scripts/e4_analyze.py --root results/e4 --out "$OUT/e4-host-sensitivity.md"

if [ -d results/e8 ]; then
  step "E8 sampling profile (Gate F evidence)" \
    "$PY" scripts/e8_analyze.py --root results/e8 --out "$OUT/e8-sampling.md"
fi

step "Legacy campaign tables (M1/M3/M4/M5)" \
  "$PY" scripts/make_paper_tables.py

step "LaTeX tables for the paper" \
  "$PY" scripts/make_latex_tables.py --out paper/tables

step "Paper figures and generated tables (paired bootstraps; ~10 min at n=10000)" \
  "$PY" scripts/make_figures.py --boot "$BOOT"

step "Artifact completeness" \
  "$PY" scripts/verify_campaign.py

printf '\n'
if [ "$fail" -eq 0 ]; then
  echo "ALL PROTOCOL ANALYSES REPRODUCED (bootstrap n=$BOOT)"
else
  echo "$fail step(s) FAILED -- see output above"
fi
exit "$fail"
