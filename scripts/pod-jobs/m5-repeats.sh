# M5 full compliance (roadmap Milestone 5):
#   "Repeat headline timings and report variance."
#   "Report tokens committed per target forward, drafting latency, verification
#    latency, total milliseconds per token, speedup, and output-equivalence."
#
# Two passes over all 12 factorial configurations:
#   A. repeats r2,r3 with timers OFF  -> variance on clean wall-clock (r1 = the
#      existing results/m5 cells, so each config reaches n=3)
#   B. one run per config at K=8 with profile_timing=true -> the draft / tree /
#      verify / commit decomposition. Kept separate because timers add overhead
#      and must never contaminate a reported wall-clock number.
#
# Runs from /workspace/specdec-v2, an isolated checkout carrying the
# output-equivalence recording, so the primary chain is untouched.
# Marker-guarded: waits for the primary chain's terminal marker first.
set -o pipefail
R=/workspace/specdec-runs
TAILLOG=$R/logs/m3-rq2-matched.log
cd /workspace/specdec-v2

for i in $(seq 1 480); do
  grep -qE 'M3_RQ2_MATCHED_(DONE|ALREADY_DONE)' "$TAILLOG" 2>/dev/null && break
  sleep 300
done
grep -qE 'M3_RQ2_MATCHED_(DONE|ALREADY_DONE)' "$TAILLOG" 2>/dev/null \
  || { echo M5_REPEATS_TIMEOUT_WAITING_FOR_TAIL; exit 1; }

export HF_TOKEN=$(cat /workspace/.secrets/hf_token_read) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
SCALED=$R/checkpoints/m3-sw80k/step_20000.pt
BASE=$R/checkpoints/drafter-20k.pt

# cfg_for <stem> -> config files + drafter override
cfg_for() {
  case $1 in
    base)   echo "configs/base.yaml|eval.drafter_ckpt=$BASE" ;;
    scaled) echo "configs/base.yaml|eval.drafter_ckpt=$SCALED" ;;
    eagle)  echo "configs/base.yaml configs/eagle.yaml|" ;;
  esac
}

run() { # OUT CFG CKARG DYN BENCH KSWEEP PROFILE
  OUT=$1; CFG=$2; CK=$3; DYN=$4; B=$5; KS=$6; PROF=$7
  if [ -s "$OUT/benchmark.csv" ]; then echo "SKIP $(basename $OUT)"; return 0; fi
  python3 scripts/run_eval.py --config $CFG --set \
    $CK eval.prompt_source=$B eval.num_prompts=80 eval.max_new_tokens=128 \
    "eval.K_sweep=[$KS]" tree.allocation=geometric tree.num_nodes=256 \
    decode.dynamic_tree=$DYN decode.profile_timing=$PROF eval.out_dir=$OUT \
    2>&1 | tail -2
}

# --- Pass A: repeats r2, r3 (timers OFF) ------------------------------------
for REP in 2 3; do
  for B in mtbench humaneval; do
    for STEM in base scaled eagle; do
      IFS='|' read -r CFG CK <<< "$(cfg_for $STEM)"
      for T in static dyn; do
        [ "$T" = dyn ] && DYN=true || DYN=false
        run "$R/results/m5-rep/$STEM-$T-$B-r$REP" "$CFG" "$CK" $DYN $B 4,8 false
        echo "M5REP_${STEM}-${T}-${B}_r${REP}_DONE"
      done
    done
  done
done
echo M5_REPEATS_DONE

# --- Pass B: latency decomposition, K=8, timers ON --------------------------
for B in mtbench humaneval; do
  for STEM in base scaled eagle; do
    IFS='|' read -r CFG CK <<< "$(cfg_for $STEM)"
    for T in static dyn; do
      [ "$T" = dyn ] && DYN=true || DYN=false
      run "$R/results/m5-timing/$STEM-$T-$B" "$CFG" "$CK" $DYN $B 8 true
      echo "M5TIME_${STEM}-${T}-${B}_DONE"
    done
  done
done
echo M5_TIMING_DONE
echo M5_COMPLIANCE_ALL_DONE
