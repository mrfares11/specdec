"""Held-out CE evaluator: determinism and arm-independence of batching."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from heldout_ce import evaluate  # noqa: E402

from src.data.shards import ShardWriter
from src.models.drafter import Drafter


def _make_dev_shards(tmp_path, cfg, target, n=6):
    out = tmp_path / "dev"
    g = torch.Generator().manual_seed(11)
    h = int(cfg.drafter.hidden_size)
    with ShardWriter(out, shard_tokens=512, storage_dtype="float32",
                     meta={"layer_index": -2}) as w:
        for i in range(n):
            t = int(torch.randint(6, 20, (1,), generator=g).item())
            ids = torch.randint(0, int(cfg.target.random_config.vocab_size), (t,), generator=g)
            with torch.no_grad():
                feats = target(input_ids=ids.unsqueeze(0), output_hidden_states=True
                               ).hidden_states[-2][0]
            w.add(f"dev-{i}", feats, ids)
    return out


def test_heldout_ce_deterministic_and_ckpt_sensitive(tmp_path, tiny_cfg, tiny_target):
    cfg = tiny_cfg.copy()
    dev = _make_dev_shards(tmp_path, cfg, tiny_target)
    cfg.data.out_dir = str(dev)
    cfg.train.rollout_K = 3

    def ckpt_of(seed):
        torch.manual_seed(seed)
        d = Drafter.from_config(cfg.drafter)
        p = tmp_path / f"ck{seed}.pt"
        torch.save({"step": 0, "model": d.state_dict()}, p)
        return str(p)

    a1 = evaluate(cfg, ckpt_of(1), n_samples=6)
    a2 = evaluate(cfg, ckpt_of(1), n_samples=6)
    b = evaluate(cfg, ckpt_of(2), n_samples=6)
    assert a1["per_step_ce"] == a2["per_step_ce"], "must be deterministic"
    assert a1["samples"] == b["samples"] == 6, "identical batching across ckpts"
    assert a1["uniform_ce"] != b["uniform_ce"], "must distinguish checkpoints"
    assert all(c is None or c > 0 for c in a1["per_step_ce"])
    assert a1["K"] == 3 and len(a1["per_step_ce"]) == 3
