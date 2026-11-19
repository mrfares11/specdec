"""Create a sample-count subset VIEW of a feature-shard directory.

Symlinks the needed shard files and writes a trimmed manifest (the last
included shard's sample list is cut so the subset holds exactly --samples
ids). No tensor data is copied; ShardReader loads only listed samples.

    python scripts/make_feature_subset.py --src <dir> --out <dir> --samples 40000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, required=True)
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))

    kept, total = [], 0
    for shard in manifest["shards"]:
        if total >= args.samples:
            break
        take = shard["samples"][: args.samples - total]
        kept.append({**shard, "samples": take})
        total += len(take)
    if total < args.samples:
        raise SystemExit(f"source has only {total} samples, wanted {args.samples}")

    out.mkdir(parents=True, exist_ok=True)
    for shard in kept:
        link = out / shard["file"]
        if not link.exists():
            os.symlink((src / shard["file"]).resolve(), link)
    trimmed = {**manifest, "shards": kept}
    (out / "manifest.json").write_text(json.dumps(trimmed), encoding="utf-8")
    print(f"subset: {total} samples, {len(kept)} shards -> {out}")


if __name__ == "__main__":
    main()
