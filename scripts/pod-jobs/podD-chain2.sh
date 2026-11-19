#!/usr/bin/env bash
# Pod D (2026-09-22): the revision campaign, second launch (adds D3b). Every step is resume-safe.
#
#   D0  greedy continuations + target features for the dev/test prompt sets
#   D1  per-depth rollout statistics on DEV prompts (scaled, base, EAGLE)
#   D2  offline-optimised static trees from D1 (same-profile / free-profile, K=4,8)
#   D3  W2 diagnostics: rollout statistics on TEST prompts for every objective;
#       ShareGPT held-out CE with --full-target for the same checkpoints
#   D4  Q3 timing: scaled drafter, chain-static / dynamic / optimised trees (3 repeats)
#   D5  W3 timing: RQ1 checkpoints x {static, dynamic} (2 repeats)
#   D6  Q3 transfer: baseline and EAGLE drafters with the optimised trees (2 repeats)
#
#   tmux new -d -s podD 'bash /workspace/specdec/scripts/pod-jobs/podD-chain2.sh'
set -o pipefail
R=/workspace/specdec-runs
cd /workspace/specdec
export HF_HOME=/workspace/hf-cache HF_TOKEN=$(cat /workspace/.secrets/hf_token)
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p $R/logs $R/e10/data $R/e10/trees $R/e10/rollout $R/results/e10 $R/results/e10-heldout
MODEL=NousResearch/Meta-Llama-3-8B-Instruct
MODEL_DIR=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL', allow_patterns=['*.json']))")
LOG=$R/logs/podD-chain.log
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }
say "podD chain start; model dir $MODEL_DIR"

# ---- D0: data -------------------------------------------------------------
for spec in mtbench:test:50 humaneval:test:50 mtbench:dev:26 humaneval:dev:54; do
  B=${spec%%:*}; rest=${spec#*:}; S=${rest%%:*}; N=${rest#*:}
  OUT=$R/e10/data/data_${B}_${S}.pt
  [ -f $OUT ] && { say "D0 have $OUT"; continue; }
  python3 scripts/e9_gpu_data.py --config configs/base.yaml --set target.model_id=$MODEL \
    eval.prompt_source=$B eval.prompt_split=$S eval.num_prompts=$N --out $OUT 2>&1 | tail -2 | tee -a $LOG
done
say "D0 done"

# ---- D1: dev rollout statistics for the drafters whose trees we fit --------
roll() { # name ckpt drafter-cfg feature split
  local NAME=$1 CK=$2 DC=$3 FE=$4 SP=$5
  for B in mtbench humaneval; do
    OUT=$R/e10/rollout/${NAME}-${B}-${SP}
    [ -f $OUT/summary.csv ] && continue
    python3 scripts/e9_rollout_stats.py --model $MODEL_DIR --ckpt $CK --drafter-cfg $DC --feature $FE \
      --data $R/e10/data/data_${B}_${SP}.pt --benchmark $B --split $SP --K 8 --device cuda --out $OUT \
      2>&1 | tail -1 | tee -a $LOG
  done
}
roll scaled $R/checkpoints/m3-sw80k-step_20000.pt base feat dev
roll base   $R/checkpoints/drafter-20k.pt         base feat dev
roll eagle  $R/checkpoints/eagle-import.pt        eagle final dev
say "D1 done"

# ---- D2: optimised static trees (pooled over both benchmarks' dev prompts) --
for D in scaled base eagle; do
  for K in 4 8; do
    for V in same-profile:optsame free-profile:optfree; do
      VAR=${V%%:*}; TAG=${V##*:}
      OUT=$R/e10/trees/${D}-${TAG}-k${K}.json
      [ -f $OUT ] && continue
      python3 scripts/e9_build_tree.py --stats $R/e10/rollout/${D}-mtbench-dev/rollout.pt \
        --stats $R/e10/rollout/${D}-humaneval-dev/rollout.pt --K $K --nodes 256 --variant $VAR --out $OUT \
        2>&1 | tail -1 | tee -a $LOG
    done
  done
done
say "D2 done"

# ---- D3: W2 diagnostics ---------------------------------------------------
for A in sw-s1 sw-s2 det-s1 det-s2 tf-s1 tf-s2 tf-s0; do
  roll e5-$A $R/checkpoints/e5-$A-step_10000.pt base feat test
done
roll scaled $R/checkpoints/m3-sw80k-step_20000.pt base feat test
roll base   $R/checkpoints/drafter-20k.pt         base feat test
roll eagle  $R/checkpoints/eagle-import.pt        eagle final test
roll m3-sw10k  $R/checkpoints/m3-sw80k-model_10000.pt base feat test
roll m3-det10k $R/checkpoints/m3-det80k-step_10000.pt base feat test
roll m3-tf10k  $R/checkpoints/m3-tf80k-step_10000.pt  base feat test
roll m3-bp10k  $R/checkpoints/m3-bp80k-step_10000.pt  base feat test
say "D3 rollouts done"
DEV=$R/features/dev-heldout
[ -f $DEV/manifest.json ] || python3 scripts/extract_features.py --config configs/base.yaml --set \
  target.model_id=$MODEL data.skip_samples=100000 data.max_samples=400 data.out_dir=$DEV 2>&1 | tail -2 | tee -a $LOG
for A in e5-sw-s1 e5-sw-s2 e5-det-s1 e5-det-s2 e5-tf-s1 e5-tf-s2 e5-tf-s0 m3-det80k m3-tf80k m3-bp80k; do
  OUT=$R/results/e10-heldout/$A.json
  [ -f $OUT ] && continue
  python3 scripts/heldout_ce.py --config configs/base.yaml --set target.model_id=$MODEL \
    data.out_dir=$DEV eval.drafter_ckpt=$R/checkpoints/$A-step_10000.pt train.rollout_K=8 \
    --samples 400 --full-target --out $OUT 2>&1 | tail -1 | tee -a $LOG
done
for A in m3-sw80k-model_10000:m3-sw10k m3-sw80k-step_20000:m3-sw20k; do
  CK=${A%%:*}; NAME=${A##*:}; OUT=$R/results/e10-heldout/$NAME.json
  [ -f $OUT ] && continue
  python3 scripts/heldout_ce.py --config configs/base.yaml --set target.model_id=$MODEL \
    data.out_dir=$DEV eval.drafter_ckpt=$R/checkpoints/$CK.pt train.rollout_K=8 \
    --samples 400 --full-target --out $OUT 2>&1 | tail -1 | tee -a $LOG
done
say "D3 done"

# ---- D3b: the same test continuations in the TRAINING rollout convention -----
rollt() { # name ckpt drafter-cfg feature
  local NAME=$1 CK=$2 DC=$3 FE=$4
  for B in mtbench humaneval; do
    OUT=$R/e10/rollout/${NAME}-${B}-test-train
    [ -f $OUT/summary.csv ] && continue
    python3 scripts/e9_rollout_stats.py --model $MODEL_DIR --ckpt $CK --drafter-cfg $DC --feature $FE       --data $R/e10/data/data_${B}_test.pt --benchmark $B --split test --K 8 --device cuda       --convention training --out $OUT 2>&1 | tail -1 | tee -a $LOG
  done
}
for A in sw-s1 sw-s2 det-s1 det-s2 tf-s1 tf-s2; do
  rollt e5-$A $R/checkpoints/e5-$A-step_10000.pt base feat
done
rollt scaled $R/checkpoints/m3-sw80k-step_20000.pt base feat
say "D3b done"

# ---- D4-D6: timing campaigns (E2 protocol) --------------------------------
python3 scripts/e10_campaign.py --plan q3-scaled   --session d1 --repeats 3 --out-root $R --model-id $MODEL 2>&1 | tee -a $LOG
say "D4 done"
python3 scripts/e10_campaign.py --plan w3          --session d1 --repeats 2 --out-root $R --model-id $MODEL 2>&1 | tee -a $LOG
say "D5 done"
python3 scripts/e10_campaign.py --plan q3-transfer --session d1 --repeats 2 --out-root $R --model-id $MODEL 2>&1 | tee -a $LOG
say "D6 done"
say "PODD_CHAIN_COMPLETE"
