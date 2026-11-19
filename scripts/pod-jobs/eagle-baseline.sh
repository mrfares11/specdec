# As run on the H100 pod (tmux session `eagle`), 2026-08-11.
# Produced: checkpoints/eagle-import/eagle.pt and results/bench/eagle-{humaneval,mbpp,mtbench}.
# Ran at repo commit with configs/eagle.yaml = fc_bias:false, layer0_no_input_norm:true,
# layer_index:-1, apply_final_norm:false (post-norm EAGLE feature space).
# Preserved verbatim for traceability (RESULTS.md).
set -o pipefail
cd /workspace/specdec
export HF_TOKEN=$(cat /workspace/.secrets/hf_token) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/workspace/specdec-runs
git pull -q origin main
echo '=== import official EAGLE weights ==='
python scripts/import_eagle_weights.py --src yuhuili/EAGLE-LLaMA3-Instruct-8B \
  --out $R/checkpoints/eagle-import/eagle.pt 2>&1 | tee $R/logs/eagle-import.log \
  || { echo EAGLE_IMPORT_FAILED; exit 1; }
for B in humaneval mbpp mtbench; do
  echo "=== EAGLE $B (solo) ==="
  python scripts/run_eval.py --config configs/base.yaml configs/eagle.yaml --set \
    eval.prompt_source=$B eval.num_prompts=80 eval.max_new_tokens=128 \
    'eval.K_sweep=[4,8]' tree.allocation=geometric tree.num_nodes=256 \
    eval.out_dir=$R/results/bench/eagle-$B 2>&1 | tee $R/logs/bench-eagle-$B.log
  echo EAGLE_${B}_DONE
done
echo EAGLE_ALL_DONE
