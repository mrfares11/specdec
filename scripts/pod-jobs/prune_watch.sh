#!/usr/bin/env bash
# Drop E5 intermediate checkpoints as soon as an arm finishes.
#
# The 1 TB volume filled on 2026-09-07 and killed training mid-checkpoint-write
# with no error (df reports the shared MooseFS cluster, not our quota, so the
# usual space check looks fine right up until a write fails). Each arm piles up
# 11 x 3.02 GB checkpoints, of which only step_10000.pt has a consumer: chainC
# reads $A/step_10000.pt and nothing else.
#
# Runs beside chainB rather than inside it: bash reads a running script by byte
# offset, so editing chainB.sh mid-flight can resume the live shell at a
# corrupted position.
#
# e6-* is deliberately NOT pruned -- chainD's convergence phase evaluates
# per-checkpoint dev metrics and needs every step_$S.pt it writes.
set -uo pipefail

C=/workspace/specdec-runs/checkpoints
L=/workspace/specdec-runs/logs/prune.log

while true; do
  for d in "$C"/e5-*/; do
    [ -d "$d" ] || continue
    a=$(basename "$d")
    f="$d/step_10000.pt"

    # Only prune a genuinely finished arm. Requiring the final checkpoint to be
    # a complete ~3 GB write means a partial save (exactly what the quota crash
    # produced) can never trigger deletion of the intermediates we would need to
    # resume from.
    [ -f "$f" ] || continue
    sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$sz" -gt 2000000000 ] || continue

    n=0
    for s in 1000 2000 3000 4000 5000 6000 7000 8000 9000; do
      if [ -f "$d/step_$s.pt" ]; then
        rm -f "$d/step_$s.pt" && n=$((n + 1))
      fi
    done
    if [ "$n" -gt 0 ]; then
      echo "PRUNED $a $n files, free=$(df -h /workspace | awk 'NR==2{print $4}') $(date -u +%m-%dT%H:%MZ)" >> "$L"
    fi
  done
  sleep 600
done
