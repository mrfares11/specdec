# Held-out arm-independent CE phase (runs after M3_ALL_DONE).
# Extracts a dev slice (rows >= 100k, past train+eval-prompt ranges) and
# evaluates every final arm checkpoint with identical batching.
set -o pipefail
R=/workspace/specdec-runs
cd /workspace/specdec
export HF_TOKEN=$(cat /workspace/.secrets/hf_token_read) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
DEV=$R/features/dev-heldout
[ -f $DEV/manifest.json ] || python3 scripts/extract_features.py --config configs/base.yaml --set \
  data.skip_samples=100000 data.max_samples=400 data.out_dir=$DEV \
  2>&1 | tee $R/logs/m3-dev-extract.log
for A in m3-sw80k:step_20000 m3-tf80k:step_10000 m3-det80k:step_10000 m3-bp80k:step_10000 m3-sw40k:step_10000 m3-sw20k:step_10000; do
  NAME=${A%%:*}; CK=${A##*:}
  python3 scripts/heldout_ce.py --config configs/base.yaml --set \
    data.out_dir=$DEV eval.drafter_ckpt=$R/checkpoints/$NAME/$CK.pt \
    train.rollout_K=8 --samples 400 \
    --out $R/results/m3/heldout-ce/$NAME.json 2>&1 | tail -3
  echo M3_heldout_${NAME}_DONE
done
echo M3_HELDOUT_ALL_DONE
