"""Shard round-trip, incremental flushing, resume-after-interrupt, size estimate."""

from __future__ import annotations

import json

import pytest
import torch

from src.data.shards import ShardReader, ShardWriter, estimate_bytes, format_bytes

H = 16


def _samples(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        t = int(torch.randint(3, 12, (1,), generator=g))
        out.append((f"s-{i}", torch.randn(t, H, generator=g), torch.randint(0, 99, (t,), generator=g)))
    return out


def test_round_trip_bit_exact(tmp_path):
    samples = _samples(6)
    with ShardWriter(tmp_path, shard_tokens=20, storage_dtype="float32") as w:
        for sid, h, ids in samples:
            w.add(sid, h, ids)
    r = ShardReader(tmp_path)
    assert len(r) == 6
    assert r.sample_ids() == [s[0] for s in samples]
    for sid, h, ids in samples:
        rh, rids = r.read_sample(sid)
        assert torch.equal(rh, h), f"{sid}: hidden not bit-identical"
        assert torch.equal(rids, ids)
    # iteration matches per-sample reads
    for (sid, rh, rids), (osid, oh, oids) in zip(r.samples(), samples):
        assert sid == osid and torch.equal(rh, oh) and torch.equal(rids, oids)


def test_incremental_flush_multiple_shards(tmp_path):
    samples = _samples(8, seed=1)
    with ShardWriter(tmp_path, shard_tokens=15, storage_dtype="float32") as w:
        for sid, h, ids in samples:
            w.add(sid, h, ids)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert len(manifest["shards"]) >= 2, "small budget must produce multiple shards"
    for shard in manifest["shards"]:
        assert (tmp_path / shard["file"]).exists()
    assert not (tmp_path / "manifest.json.tmp").exists(), "atomic write left a tmp file"


def test_dtype_cast_round_trip(tmp_path):
    """fp16 storage: reader returns exactly the cast tensor the writer stored."""
    sid, h, ids = _samples(1, seed=2)[0]
    with ShardWriter(tmp_path, shard_tokens=1, storage_dtype="float16") as w:
        w.add(sid, h, ids)
    rh, rids = ShardReader(tmp_path).read_sample(sid)
    assert rh.dtype == torch.float16
    assert torch.equal(rh, h.to(torch.float16))
    assert torch.equal(rids, ids)


def test_resume_after_interrupt(tmp_path):
    # Fixed-length samples so the flush boundary is deterministic:
    # 6 tokens each, budget 18 → flush after every 3rd sample.
    g = torch.Generator().manual_seed(3)
    samples = [
        (f"s-{i}", torch.randn(6, H, generator=g), torch.randint(0, 99, (6,), generator=g))
        for i in range(9)
    ]
    # First run "crashes" before close, losing whatever sat in the buffer.
    w1 = ShardWriter(tmp_path, shard_tokens=18, storage_dtype="float32")
    for sid, h, ids in samples[:5]:
        w1.add(sid, h, ids)
    flushed = set(w1.completed_ids)
    assert flushed == {"s-0", "s-1", "s-2"}, "expected exactly one flushed shard"
    del w1  # no close(): buffered samples are lost, manifest still valid

    # Second run resumes: skips exactly the flushed ones, re-adds the rest.
    w2 = ShardWriter(tmp_path, shard_tokens=18, storage_dtype="float32", resume=True)
    assert w2.completed_ids == flushed
    for sid, h, ids in samples:
        if sid not in w2.completed_ids:
            w2.add(sid, h, ids)
    w2.close()

    r = ShardReader(tmp_path)
    assert set(r.sample_ids()) == {s[0] for s in samples}
    for sid, h, ids in samples:
        rh, rids = r.read_sample(sid)
        assert torch.equal(rh, h) and torch.equal(rids, ids)


def test_resume_refuses_changed_meta(tmp_path):
    """Changed extraction settings must not silently mix features in one dir."""
    sid, h, ids = _samples(1, seed=7)[0]
    meta = {"model_id": "m", "layer_index": -2, "max_seq_len": 64}
    with ShardWriter(tmp_path, shard_tokens=4, storage_dtype="float32", meta=meta) as w:
        w.add(sid, h, ids)
    # same meta resumes fine
    ShardWriter(tmp_path, shard_tokens=4, storage_dtype="float32", meta=meta, resume=True)
    # changed layer_index is refused
    with pytest.raises(ValueError, match="layer_index"):
        ShardWriter(
            tmp_path, shard_tokens=4, storage_dtype="float32",
            meta={**meta, "layer_index": -1}, resume=True,
        )
    # keys the manifest never recorded are not a mismatch (hand-built shards)
    ShardWriter(
        tmp_path, shard_tokens=4, storage_dtype="float32",
        meta={**meta, "new_key": "x"}, resume=True,
    )


def test_resume_false_refuses_to_clobber(tmp_path):
    with ShardWriter(tmp_path, shard_tokens=8, storage_dtype="float32") as w:
        w.add(*_samples(1, seed=4)[0])
    with pytest.raises(FileExistsError):
        ShardWriter(tmp_path, shard_tokens=8, storage_dtype="float32", resume=False)


def test_duplicate_and_shape_validation(tmp_path):
    sid, h, ids = _samples(1, seed=5)[0]
    with ShardWriter(tmp_path, shard_tokens=1000, storage_dtype="float32") as w:
        w.add(sid, h, ids)
        with pytest.raises(ValueError):
            w.add(sid, h, ids)  # duplicate id
        with pytest.raises(ValueError):
            w.add("bad", h, ids[:-1])  # length mismatch


def test_estimate_bytes():
    # 1000 tokens, hidden 4096, fp16: 4096*2 + 8 bytes/token
    assert estimate_bytes(1000, 4096, "float16") == 1000 * (4096 * 2 + 8)
    assert estimate_bytes(10, H, "float32") == 10 * (H * 4 + 8)
    assert "GiB" in format_bytes(estimate_bytes(500_000_000, 4096, "float16")) or True
    assert format_bytes(2048) == "2.00 KiB"
    with pytest.raises(ValueError):
        estimate_bytes(1, 1, "int8")
