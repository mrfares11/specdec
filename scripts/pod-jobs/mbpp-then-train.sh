# As run on the H100 pod (tmux session `master`), 2026-08-10/11.
# Produced: results/bench/mbpp.csv, checkpoints/drafter-sw (10k steps),
# results/bench/rq2-sw10. Preserved verbatim for traceability (RESULTS.md).
set -o pipefail
cd /workspace/specdec
export HF_TOKEN=$(cat /workspace/.secrets/hf_token) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/specdec-runs
git pull -q origin main
echo '=== MBPP (solo: clean accept + timing) ==='
python scripts/run_eval.py --config configs/base.yaml --set eval.drafter_ckpt=$R/checkpoints/drafter/step_20000.pt   eval.prompt_source=mbpp eval.num_prompts=80 eval.max_new_tokens=128   'eval.K_sweep=[4,8]' tree.allocation=geometric tree.num_nodes=256   eval.out_dir=$R/results/bench/mbpp 2>&1 | tee $R/logs/bench-mbpp.log
echo MBPP_DONE
echo '=== resuming step-weighted training ==='
for attempt in 1 2 3 4 5; do
  python scripts/train_drafter.py --config configs/base.yaml --set     train.batch_size=4 train.grad_accum=4 train.max_steps=10000     train.save_every=1000 train.log_every=50     train.rollout_detach=false train.ce_chunk=256 train.rollout_step_decay=0.5     train.target_dtype=bfloat16 train.allow_tf32=true     train.ckpt_dir=$R/checkpoints/drafter-sw 2>&1     && { echo SW_TRAIN_COMPLETE; break; }
  sleep 30
done
python scripts/run_eval.py --config configs/base.yaml --set eval.drafter_ckpt=$R/checkpoints/drafter-sw/step_10000.pt   eval.prompt_source=mtbench eval.num_prompts=80 eval.max_new_tokens=128   'eval.K_sweep=[4,8]' tree.allocation=geometric tree.num_nodes=256   eval.out_dir=$R/results/bench/rq2-sw10 2>&1 | tee $R/logs/bench-rq2-sw10.log
echo EVERYTHING_DONE
