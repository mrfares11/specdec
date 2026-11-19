"""Training loop smoke test on tiny synthetic shards: runs, checkpoints, resumes."""

from __future__ import annotations

import torch

from src.data.shards import ShardWriter
from src.train.train_drafter import run


def _make_shards(out_dir, cfg):
    """Synthetic (tokens, features) shards matching the tiny target dims."""
    g = torch.Generator().manual_seed(0)
    vocab = int(cfg.target.random_config.vocab_size)
    hidden = int(cfg.target.random_config.hidden_size)
    with ShardWriter(out_dir, shard_tokens=256, storage_dtype="float32") as w:
        for i in range(6):
            t = int(torch.randint(8, 17, (1,), generator=g))
            w.add(f"t-{i}", torch.randn(t, hidden, generator=g), torch.randint(0, vocab, (t,), generator=g))


def test_train_runs_checkpoints_and_resumes(tiny_cfg, tmp_path):
    cfg = tiny_cfg.copy()
    shard_dir = tmp_path / "feats"
    ckpt_dir = tmp_path / "ckpt"
    _make_shards(shard_dir, cfg)
    cfg.data.out_dir = shard_dir.as_posix()
    cfg.train.ckpt_dir = ckpt_dir.as_posix()
    cfg.train.max_steps = 3
    cfg.train.rollout_K = 2

    final = run(cfg)
    assert final == 3
    assert (ckpt_dir / "latest.pt").exists()
    assert (ckpt_dir / "step_2.pt").exists(), "save_every=2 snapshot missing"

    state = torch.load(ckpt_dir / "latest.pt", map_location="cpu")
    assert state["step"] == 3
    # Only drafter weights in the checkpoint — never target weights.
    assert all(k.startswith(("fc.", "layers.")) for k in state["model"])
    assert all(torch.isfinite(v).all() for v in state["model"].values())

    # Resume: raising max_steps continues from step 3, not from scratch.
    cfg.train.max_steps = 5
    final = run(cfg)
    assert final == 5
    assert torch.load(ckpt_dir / "latest.pt", map_location="cpu")["step"] == 5

    # Loss log exists and is finite.
    log = (ckpt_dir / "train_log.csv").read_text().strip().splitlines()
    assert len(log) >= 2
    loss = float(log[1].split(",")[1])
    assert loss == loss and loss < float("inf")
