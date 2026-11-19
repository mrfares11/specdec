#!/usr/bin/env bash
# Bootstrap a FRESH pod attached to the specdec network volume.
# Everything durable lives on /workspace; this script rebuilds the ephemeral
# container half (tmux, pip deps, env vars) and verifies the durable half.
#
#   Usage (on the new pod):   bash /workspace/specdec/scripts/pod_setup.sh
#   Then on the laptop:       git remote set-url pod ssh://root@<IP>:<PORT>/workspace/specdec.git
#
# Secrets: the HF token is read from /workspace/.secrets/hf_token (chmod 600,
# on the volume, never in git). Recreate it with:
#   mkdir -p /workspace/.secrets && printf '%s' 'hf_...' > /workspace/.secrets/hf_token && chmod 600 /workspace/.secrets/hf_token
set -euo pipefail

echo "== specdec pod bootstrap =="
[ -d /workspace/specdec ] || { echo "FATAL: /workspace/specdec missing — wrong/unattached volume?"; exit 1; }

# 1. environment (persist for future interactive shells too)
export HF_HOME=/workspace/hf-cache
if [ -f /workspace/.secrets/hf_token ]; then
    HF_TOKEN=$(cat /workspace/.secrets/hf_token)
    export HF_TOKEN
else
    echo "WARN: /workspace/.secrets/hf_token missing — gated model access will fail until recreated"
fi
if ! grep -q 'specdec bootstrap' /root/.bashrc 2>/dev/null; then
    cat >> /root/.bashrc <<'RC'
# specdec bootstrap
[ -f /workspace/.secrets/hf_token ] && export HF_TOKEN=$(cat /workspace/.secrets/hf_token)
export HF_HOME=/workspace/hf-cache
export PYTHONUNBUFFERED=1
RC
fi

# 2. system deps
command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; }

# 3. python deps (torch comes with the pod image; pip fills in the rest)
cd /workspace/specdec
pip install -q -e '.[dev,data]'

# 4. verify: CPU test suite must be green before any GPU spend
python -m pytest -q || { echo "FATAL: test suite failed on this pod image"; exit 1; }

# 5. durable-state inventory
python - <<'PY'
import json, os, pathlib
import torch

print("cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(none)")

def check(path, what):
    ok = pathlib.Path(path).exists()
    print(f"{what:<18} {'OK' if ok else 'MISSING'}  {path}")
    return ok

check("/workspace/specdec.git", "bare git remote")
if check("/workspace/specdec-runs/features/llama3-8b/manifest.json", "feature shards"):
    m = json.load(open("/workspace/specdec-runs/features/llama3-8b/manifest.json"))
    print(f"{'':<18} {sum(len(s['samples']) for s in m['shards'])} samples, "
          f"{sum(s['num_tokens'] for s in m['shards']):,} tokens")
check("/workspace/specdec-runs/checkpoints/drafter/latest.pt", "drafter ckpt")
check("/workspace/specdec-runs/results", "results dir")
if os.environ.get("HF_TOKEN"):
    from huggingface_hub import whoami
    print("hf account:", whoami()["name"])
PY

echo "== bootstrap complete =="
echo "Reminders: long jobs under tmux; laptop must update its 'pod' remote URL."
