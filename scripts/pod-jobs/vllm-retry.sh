# vLLM reference row, retried after the earlier attempt failed on a MISSING
# DEPENDENCY rather than on vLLM itself: vllm 0.28.0 installed fine, but
# scripts/vllm_reference.py imports `datasets` to load the benchmark prompts and
# the isolated venv did not have it.
#
# Beyond the roadmap's M5 system list (which does not include vLLM); kept
# because the draft references an optimized-engine row for external context.
# It is an ANCHOR, not a like-for-like lossless-speculation comparison.
#
# Marker-guarded behind the compliance phase so it never shares the GPU.
set -o pipefail
R=/workspace/specdec-runs
LOG=$R/logs/m5-compliance.log
OUT=$R/results/m5/vllm-reference.json

[ -s "$OUT" ] && { echo VLLM_ALREADY_DONE; exit 0; }

for i in $(seq 1 480); do
  grep -q M5_COMPLIANCE_ALL_DONE "$LOG" 2>/dev/null && break
  sleep 300
done
grep -q M5_COMPLIANCE_ALL_DONE "$LOG" 2>/dev/null || { echo VLLM_TIMEOUT; exit 1; }

export HF_TOKEN=$(cat /workspace/.secrets/hf_token_read) HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /workspace/specdec
/workspace/vllm-venv/bin/python scripts/vllm_reference.py --out "$OUT" 2>&1 | tail -8 \
  && echo VLLM_RETRY_DONE || echo VLLM_RETRY_FAILED
