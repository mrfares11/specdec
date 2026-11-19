# Step-matched M3 completion: the two cells that leave M3's headline claims
# confounded by training steps.  Both use m3-sw80k/model_10000.pt (the
# step-weighted 80k-row arm at the SAME 10k steps as det/bp/tf and as the
# 20k/40k scaling points).
#
#   1. Held-out arm-independent CE.  The M3 held-out phase scored m3-sw80k only
#      at step_20000, so the RQ2 objective table compared a 20k-step arm against
#      10k-step arms.  (Acceptance was already step-matched via m3-sw80k-s10.)
#   2. HumanEval acceptance.  The data-scaling curve is step-matched on MT-Bench
#      (1.52 / 1.57 / 1.69) but on HumanEval the 80k point existed only at 20k
#      steps, so "more data helps" was confounded with "more steps".
#
# GPU-serialised: waits for M5_ALL_DONE before touching the device, so it can
# never perturb M5's wall-clock timing rows.  Marker-guarded and idempotent.
set -o pipefail
R=/workspace/specdec-runs
M5LOG=$R/logs/m5-master.log
CE_OUT=$R/results/m3/heldout-ce/m3-sw80k-s10.json
HE_OUT=$R/results/m3/m3-sw80k-s10-humaneval
CK=$R/checkpoints/m3-sw80k/model_10000.pt

if [ -s "$CE_OUT" ] && [ -s "$HE_OUT/benchmark.csv" ]; then
  echo M3_RQ2_MATCHED_ALREADY_DONE; exit 0
fi

# Wait up to 40h for M5 to finish; poll every 5 min.
for i in $(seq 1 480); do
  grep -q M5_ALL_DONE "$M5LOG" 2>/dev/null && break
  sleep 300
done
grep -q M5_ALL_DONE "$M5LOG" 2>/dev/null || { echo M3_RQ2_MATCHED_TIMEOUT; exit 1; }

cd /workspace/specdec
GIT_TERMINAL_PROMPT=0 timeout 20 git pull -q origin main || echo GIT_PULL_SKIPPED
export HF_TOKEN=$(cat /workspace/.secrets/hf_token_read) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

# (1) Identical settings to the M3 held-out phase (same dev shards, K, budget).
if [ -s "$CE_OUT" ]; then echo M3_RQ2_MATCHED_CE_SKIP; else
  python3 scripts/heldout_ce.py --config configs/base.yaml --set \
    data.out_dir=$R/features/dev-heldout eval.drafter_ckpt=$CK \
    train.rollout_K=8 --samples 400 --out "$CE_OUT" 2>&1 | tail -3
  echo M3_RQ2_MATCHED_CE_DONE
fi

# (2) Identical settings to the M3 eval phase (ev() in m3-train.sh).
if [ -s "$HE_OUT/benchmark.csv" ]; then echo M3_RQ2_MATCHED_HE_SKIP; else
  python3 scripts/run_eval.py --config configs/base.yaml --set \
    eval.drafter_ckpt=$CK eval.prompt_source=humaneval eval.num_prompts=80 \
    eval.max_new_tokens=128 'eval.K_sweep=[4,8]' tree.allocation=geometric \
    tree.num_nodes=256 eval.out_dir=$HE_OUT 2>&1 | tail -3
  echo M3_RQ2_MATCHED_HE_DONE
fi
# (3) MBPP static+dynamic for the baseline drafter, under the CAPPED host.
# The M4 MBPP dynamic timing is contaminated by pre-cap thread thrashing and is
# otherwise never re-measured; these two cells make the C3 dynamic-tree
# reversal a three-benchmark result instead of two. Last in the queue, so they
# cannot delay anything that matters.
for DYN in false true; do
  TAG=$([ "$DYN" = "true" ] && echo dyn || echo static)
  OUT=$R/results/m5/base-$TAG-mbpp
  if [ -s "$OUT/benchmark.csv" ]; then echo "M3_MBPP_${TAG}_SKIP"; continue; fi
  python3 scripts/run_eval.py --config configs/base.yaml --set     eval.drafter_ckpt=$R/checkpoints/drafter-20k.pt eval.prompt_source=mbpp     eval.num_prompts=80 eval.max_new_tokens=128 'eval.K_sweep=[4,8]'     tree.allocation=geometric tree.num_nodes=256 decode.dynamic_tree=$DYN     eval.out_dir=$OUT 2>&1 | tail -3
  echo "M3_MBPP_${TAG}_DONE"
done

echo M3_RQ2_MATCHED_DONE
