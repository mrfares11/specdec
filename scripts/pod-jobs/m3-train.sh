# M3 training campaign: scaling curve (sw @ 20k/40k/80k rows) + re-certified
# RQ2 arms (det/bp @ 80k) — 5 arms x 10k steps, matched everything else.
# Checkpoint hygiene: after each arm keep step_10000 (full) + model-only
# 2000/5000; delete the rest (disk headroom on the 1TB volume).
set -o pipefail
R=/workspace/specdec-runs
cd /workspace/specdec
git pull -q origin main
export HF_TOKEN=$(cat /workspace/.secrets/hf_token_read) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 224-core box: uncapped OMP threads starve the GPU (0%<->98% oscillation,
# 38s/step vs 11.5 on pod A). Cap CPU-side parallelism.
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TORCH_NUM_THREADS=8
F=$R/features/llama3-8b
python3 scripts/make_feature_subset.py --src $F --out $R/features/llama3-8b-40k --samples 40000
python3 scripts/make_feature_subset.py --src $F --out $R/features/llama3-8b-20k --samples 20000

train_arm() { NAME=$1; DIR=$2; DETACH=$3; DECAY=$4;
  for attempt in 1 2 3 4 5; do
    python3 scripts/train_drafter.py --config configs/base.yaml --set \
      train.batch_size=4 train.grad_accum=4 train.max_steps=10000 \
      train.save_every=1000 train.log_every=50 \
      train.rollout_detach=$DETACH train.ce_chunk=256 train.rollout_step_decay=$DECAY \
      train.target_dtype=bfloat16 train.allow_tf32=true \
      data.out_dir=$DIR train.ckpt_dir=$R/checkpoints/$NAME \
      2>&1 | tee -a $R/logs/$NAME.log && { echo M3_${NAME}_TRAINED; break; }
    sleep 30
  done
  python3 - "$R/checkpoints/$NAME" <<'PY'
import sys, torch, pathlib
d = pathlib.Path(sys.argv[1])
for s in (2000, 5000):
    p = d / f"step_{s}.pt"
    if p.exists():
        st = torch.load(p, map_location="cpu")
        torch.save({"step": st["step"], "model": st["model"]}, d / f"model_{s}.pt")
for p in d.glob("step_*.pt"):
    if p.stem != "step_10000":
        p.unlink()
(d / "latest.pt").unlink(missing_ok=True)
print("pruned", d)
PY
  echo M3_${NAME}_DONE
}

# v2 (mock-review driven): RQ3 ladder re-run first (pod-B, headline drafter);
# sw80k to MATCHED 20k steps (headline det-20k parity); teacher-forced control.
lad() { TAG=$1; shift; python3 scripts/run_eval.py --config configs/base.yaml --set   eval.drafter_ckpt=$R/checkpoints/drafter-20k.pt eval.prompt_source=mtbench   eval.num_prompts=80 eval.max_new_tokens=128 'eval.K_sweep=[8]'   tree.allocation=geometric tree.num_nodes=256 decode.profile_timing=true "$@"   eval.out_dir=$R/results/m3/ladder-$TAG 2>&1 | tee $R/logs/m3-ladder-$TAG.log; echo M3_ladder_${TAG}_DONE; }
lad replay decode.commit_fold=false decode.kv_gather=false
lad gather decode.commit_fold=false decode.kv_gather=true
lad fold   decode.commit_fold=true
echo M3_LADDER_DONE

train_arm_n() { NAME=$1; DIR=$2; DETACH=$3; DECAY=$4; STEPS=$5; RK=$6;
  for attempt in 1 2 3 4 5; do
    python3 scripts/train_drafter.py --config configs/base.yaml --set       train.batch_size=4 train.grad_accum=4 train.max_steps=$STEPS       train.save_every=1000 train.log_every=50 train.rollout_K=$RK       train.rollout_detach=$DETACH train.ce_chunk=256 train.rollout_step_decay=$DECAY       train.target_dtype=bfloat16 train.allow_tf32=true       data.out_dir=$DIR train.ckpt_dir=$R/checkpoints/$NAME       2>&1 | tee -a $R/logs/$NAME.log && { echo M3_${NAME}_TRAINED; break; }
    sleep 30
  done
  python3 - "$R/checkpoints/$NAME" "$STEPS" <<'PY2'
import sys, torch, pathlib
d = pathlib.Path(sys.argv[1]); final = f"step_{sys.argv[2]}"
for s in (2000, 5000, 10000):
    p = d / f"step_{s}.pt"
    if p.exists() and p.stem != final:
        st = torch.load(p, map_location="cpu")
        torch.save({"step": st["step"], "model": st["model"]}, d / f"model_{s}.pt")
for p in d.glob("step_*.pt"):
    if p.stem != final:
        p.unlink()
(d / "latest.pt").unlink(missing_ok=True)
print("pruned", d)
PY2
  echo M3_${NAME}_DONE
}

train_arm_n m3-sw80k  $F                          false 0.5 20000 8
train_arm_n m3-tf80k  $F                          false 1.0 10000 1
train_arm_n m3-det80k $F                          true  1.0 10000 8
train_arm_n m3-bp80k  $F                          false 1.0 10000 8
train_arm_n m3-sw40k  $R/features/llama3-8b-40k   false 0.5 10000 8
train_arm_n m3-sw20k  $R/features/llama3-8b-20k   false 0.5 10000 8
echo M3_TRAIN_ALL_DONE

ev() { CK=$1; TAG=$2; B=$3; python3 scripts/run_eval.py --config configs/base.yaml --set \
  eval.drafter_ckpt=$CK eval.prompt_source=$B eval.num_prompts=80 eval.max_new_tokens=128 \
  'eval.K_sweep=[4,8]' tree.allocation=geometric tree.num_nodes=256 \
  eval.out_dir=$R/results/m3/$TAG-$B 2>&1 | tee $R/logs/m3-eval-$TAG-$B.log; echo M3_eval_${TAG}_${B}_DONE; }
ev $R/checkpoints/m3-sw80k/step_20000.pt m3-sw80k mtbench
ev $R/checkpoints/m3-sw80k/model_10000.pt m3-sw80k-s10 mtbench
for A in m3-tf80k m3-det80k m3-bp80k m3-sw40k m3-sw20k; do ev $R/checkpoints/$A/step_10000.pt $A mtbench; done
for S in 2000 5000; do ev $R/checkpoints/m3-sw80k/model_$S.pt m3-sw80k-s$S mtbench; done
ev $R/checkpoints/m3-sw80k/step_20000.pt m3-sw80k humaneval
for A in m3-sw40k m3-sw20k; do ev $R/checkpoints/$A/step_10000.pt $A humaneval; done
echo M3_ALL_DONE
