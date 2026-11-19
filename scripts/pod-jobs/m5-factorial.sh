# M5: final factorial + reviewer-mandated extras. Runs after M3_HELDOUT_ALL_DONE.
#
# Factorial (2x2 drafter x tree) + external drafter + AR baseline (measured in
# every run), on 2 benchmarks x K={4,8}:
#   drafter in {v1.0 baseline (det-20k), scaled (best M3 arm)}
#   tree    in {static geometric-256, dynamic (same budget profile)}
#   plus official EAGLE x {static, dynamic}   <- review fix: symmetric comparison
# Extras: 512-token generation setting; temperature=1 acceptance; vLLM
# reference row (isolated venv, failure-tolerant).
#
# v2 (billing fix): K=4 and K=8 are swept INSIDE one process per cell instead of
# one process per (cell,K). Halves the model loads, and both K rows are then
# measured against AR baselines taken in the SAME warm process (the harness
# still times AR once per K, so they are not literally one measurement -- but
# they no longer differ by process/warm-up state: 18.001 vs 17.982 ms/tok in
# the first cell).
# Every cell is resume-guarded on its benchmark.csv, so a restart never repeats
# completed work.
set -o pipefail
R=/workspace/specdec-runs
cd /workspace/specdec
git pull -q origin main
export HF_TOKEN=$(cat /workspace/.secrets/hf_token_read) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# Pick the scaled drafter = best MT-Bench accept among M3 arms (data-driven).
SCALED=$(python3 - <<'PY'
import csv, glob, os
best, bestv = None, -1
for p in glob.glob('/workspace/specdec-runs/results/m3/m3-*-mtbench/benchmark.csv'):
    tag = os.path.basename(os.path.dirname(p)).replace('-mtbench','')
    if tag.endswith('-s2000') or tag.endswith('-s5000') or tag.endswith('-s10'):
        continue
    with open(p) as f:
        for row in csv.DictReader(f):
            v = float(row['mean_accept_len'])
            if v > bestv: best, bestv = tag, v
ck = {'m3-sw80k':'step_20000'}.get(best, 'step_10000')
print(f"/workspace/specdec-runs/checkpoints/{best}/{ck}.pt" if best else "")
PY
)
[ -n "$SCALED" ] || SCALED=$R/checkpoints/m3-sw80k/step_20000.pt
BASE=$R/checkpoints/drafter-20k.pt
echo "M5_SCALED_DRAFTER=$SCALED"
echo "M5_BASE_DRAFTER=$BASE"

# ev NAME CFG CKARG DYN BENCH KSWEEP MAXTOK [extra --set key=val ...]
ev() {
  NAME=$1; CFG=$2; CKARG=$3; DYN=$4; B=$5; KS=$6; MT=$7; shift 7
  if [ -s "$R/results/m5/$NAME/benchmark.csv" ]; then echo "M5_${NAME}_SKIP"; return 0; fi
  python3 scripts/run_eval.py --config $CFG --set \
    $CKARG eval.prompt_source=$B eval.num_prompts=80 eval.max_new_tokens=$MT \
    "eval.K_sweep=[$KS]" tree.allocation=geometric tree.num_nodes=256 \
    decode.dynamic_tree=$DYN eval.out_dir=$R/results/m5/$NAME "$@" \
    2>&1 | tee $R/logs/m5-$NAME.log
  echo "M5_${NAME}_DONE"
}

CB=configs/base.yaml
CE="configs/base.yaml configs/eagle.yaml"

# --- 2x2 factorial: {base, scaled} x {static, dynamic}, both benchmarks -------
for B in mtbench humaneval; do
  ev base-static-$B   "$CB" eval.drafter_ckpt=$BASE   false $B 4,8 128
  ev base-dyn-$B      "$CB" eval.drafter_ckpt=$BASE   true  $B 4,8 128
  ev scaled-static-$B "$CB" eval.drafter_ckpt=$SCALED false $B 4,8 128
  ev scaled-dyn-$B    "$CB" eval.drafter_ckpt=$SCALED true  $B 4,8 128
done
echo M5_FACTORIAL_OURS_DONE

# --- symmetric external baseline: official EAGLE x {static, dynamic} ---------
for B in mtbench humaneval; do
  ev eagle-static-$B "$CE" "" false $B 4,8 128
  ev eagle-dyn-$B    "$CE" "" true  $B 4,8 128
done
echo M5_FACTORIAL_EAGLE_DONE

# --- Extras: long generation + sampling-mode acceptance (protocol breadth) ---
ev long-base-static    "$CB" eval.drafter_ckpt=$BASE   false mtbench 4 512
ev long-scaled-static  "$CB" eval.drafter_ckpt=$SCALED false mtbench 4 512
ev temp1-base-static   "$CB" eval.drafter_ckpt=$BASE   false mtbench 4 128 eval.greedy=false sampling.temperature=1.0
ev temp1-scaled-static "$CB" eval.drafter_ckpt=$SCALED false mtbench 4 128 eval.greedy=false sampling.temperature=1.0
echo M5_EXTRAS_DONE

# vLLM reference row (isolated venv; never blocks the chain)
if [ -s "$R/results/m5/vllm-reference.json" ]; then echo M5_VLLM_SKIP; else
( python3 -m venv /workspace/vllm-venv 2>/dev/null \
  && /workspace/vllm-venv/bin/pip install -q vllm 2>&1 | tail -1 \
  && /workspace/vllm-venv/bin/python scripts/vllm_reference.py \
       --out $R/results/m5/vllm-reference.json 2>&1 | tail -5 \
  && echo M5_VLLM_DONE ) || echo M5_VLLM_SKIPPED
fi
echo M5_ALL_DONE
