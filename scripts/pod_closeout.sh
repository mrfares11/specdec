#!/bin/bash
# POD CLOSE-OUT CHECKLIST — run BEFORE ending (or risking) any pod/volume rental.
# Pushes everything pushable, then prints EXACTLY what will be destroyed with
# the volume so ending the rental is an informed decision, never an accident.
#
#   bash scripts/pod_closeout.sh            # sync + report
#
# Lesson learned 2026-08: the 1TB campaign volume was destroyed when its rental
# ended; ~800GB of extracted features (≈20 H100-hours to regenerate) were lost
# because "everything was pushed" covered products, not bulk intermediates.
# Bulk that must survive belongs in cold object storage (~$1-6/TB/month), not
# on a rented volume.
set -uo pipefail
R=${R:-/workspace/specdec-runs}
REPO=${REPO:-/workspace/specdec}

echo "=== 1/4  Sync small artifacts (results, logs, manifests) into git ==="
mkdir -p "$REPO/artifacts/pod"
rsync -a --exclude '*.pt' --exclude '*.safetensors' --prune-empty-dirs \
      --include '*/' --include '*.csv' --include '*.log' --include '*.txt' \
      --include '*.json' --include '*.png' --exclude '*' \
      "$R/" "$REPO/artifacts/pod/" 2>/dev/null
cd "$REPO"
git add -A artifacts/pod && git commit -q -m "pod close-out sync $(date -u +%F)" && git push -q origin main \
  && echo "   pushed." || echo "   nothing new to push (or push failed - CHECK)."

echo "=== 2/4  Checkpoints not yet on HF (upload the ones that matter) ==="
find "$R/checkpoints" -name '*.pt' -newer "$R/.last-hf-upload" 2>/dev/null \
  || find "$R/checkpoints" -name '*.pt' 2>/dev/null
echo "   upload via: python -c 'from huggingface_hub import HfApi; ...' then: touch $R/.last-hf-upload"

echo "=== 3/4  BULK DATA THAT DIES WITH THIS VOLUME ==="
du -sh "$R/features" "$R/checkpoints" /workspace/hf-cache 2>/dev/null
echo "   features  -> regenerable (~20 H100-h) OR archive to cold storage (~\$5/mo for 1TB):"
echo "       rclone copy $R/features b2:specdec-features   # configure rclone once"
echo "   hf-cache  -> disposable (re-downloads)."

echo "=== 4/4  Decision ==="
echo "   Resuming within ~a month?  STOP the pod (volume persists, billed)."
echo "   Done for longer?           Archive features (step 3), THEN terminate."
