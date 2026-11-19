"""Incremental, resumable feature shards: safetensors files + a JSON manifest.

Design goals (hard requirements from the extraction spec):
  - shards are written INCREMENTALLY — nothing accumulates in RAM beyond one
    shard's token budget;
  - runs are RESUMABLE — the manifest records which samples are safely on disk,
    and a restarted writer skips them (an unflushed buffer is the only loss);
  - round-trips are BIT-EXACT — safetensors, guarded by ``tests/test_shards.py``.

Layout of ``out_dir``:
    manifest.json                  (atomic: written to *.tmp then os.replace)
    shard_00000.safetensors        keys: "<sample_id>/hidden", "<sample_id>/input_ids"
    shard_00001.safetensors        ...
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import torch
from safetensors.torch import load_file, save_file

TORCH_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

_DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "float32": 4}

_MANIFEST = "manifest.json"


def estimate_bytes(total_tokens: int, hidden_size: int, storage_dtype: str) -> int:
    """Disk estimate: hidden states + int64 token ids, manifest overhead ignored."""
    if storage_dtype not in _DTYPE_BYTES:
        raise ValueError(f"unknown storage_dtype '{storage_dtype}'")
    per_token = hidden_size * _DTYPE_BYTES[storage_dtype] + 8
    return total_tokens * per_token


def format_bytes(n: int) -> str:
    """Human-readable byte count (binary units)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError  # pragma: no cover


class ShardWriter:
    """Incremental shard writer with resume support.

    Usage::

        with ShardWriter(out_dir, shard_tokens=500_000, storage_dtype="float16",
                         meta={...}, resume=True) as w:
            for sid, hidden, ids in samples:
                if sid in w.completed_ids:
                    continue
                w.add(sid, hidden, ids)

    A shard is flushed to disk whenever the buffered token count reaches
    ``shard_tokens``; the manifest is atomically rewritten after every flush,
    so an interrupted run loses at most the current buffer.
    """

    def __init__(
        self,
        out_dir: str | Path,
        shard_tokens: int,
        storage_dtype: str = "float16",
        meta: Optional[Dict] = None,
        resume: bool = True,
    ):
        if storage_dtype not in TORCH_DTYPES:
            raise ValueError(f"unknown storage_dtype '{storage_dtype}'")
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_tokens = int(shard_tokens)
        self.storage_dtype = storage_dtype
        self._buffer: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        self._buf_tokens = 0
        self._closed = False

        manifest_path = self.out_dir / _MANIFEST
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"{manifest_path} exists; pass resume=True to continue or use a fresh out_dir"
                )
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if self._manifest.get("storage_dtype") != storage_dtype:
                raise ValueError(
                    f"resuming with storage_dtype={storage_dtype} but manifest says "
                    f"{self._manifest.get('storage_dtype')}"
                )
            # A resume with different extraction settings would silently mix
            # incompatible features into one dir — refuse on any recorded
            # meta key that differs from what the caller passes now.
            recorded = self._manifest.get("meta") or {}
            diffs = {
                k: (recorded[k], v)
                for k, v in (meta or {}).items()
                if k in recorded and recorded[k] != v
            }
            if diffs:
                detail = ", ".join(f"{k}: manifest={a!r} vs now={b!r}" for k, (a, b) in diffs.items())
                raise ValueError(
                    f"refusing to resume into {self.out_dir}: extraction meta changed ({detail}); "
                    "use a fresh out_dir or restore the original settings"
                )
        else:
            self._manifest = {
                "version": 1,
                "storage_dtype": storage_dtype,
                "meta": meta or {},
                "shards": [],
            }
            self._write_manifest()
        self._completed: Set[str] = {
            sid for shard in self._manifest["shards"] for sid in shard["samples"]
        }
        self._buffered: Set[str] = set()

    @property
    def completed_ids(self) -> Set[str]:
        """Sample ids already safely on disk (skip these when resuming). Returns a copy."""
        return set(self._completed)

    def add(self, sample_id: str, hidden: torch.Tensor, input_ids: torch.Tensor) -> None:
        """Buffer one sample (hidden [T, H], input_ids [T]); flush when full."""
        if self._closed:
            raise RuntimeError("writer is closed")
        sid = str(sample_id)
        if sid in self._completed or sid in self._buffered:
            raise ValueError(f"duplicate sample_id '{sid}'")
        if hidden.ndim != 2 or input_ids.ndim != 1 or hidden.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"expected hidden [T, H] and input_ids [T]; got {tuple(hidden.shape)} / "
                f"{tuple(input_ids.shape)}"
            )
        hidden = hidden.detach().to("cpu", TORCH_DTYPES[self.storage_dtype]).contiguous()
        input_ids = input_ids.detach().to("cpu", torch.long).contiguous()
        self._buffer.append((sid, hidden, input_ids))
        self._buffered.add(sid)
        self._buf_tokens += int(input_ids.shape[0])
        if self._buf_tokens >= self.shard_tokens:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        idx = len(self._manifest["shards"])
        fname = f"shard_{idx:05d}.safetensors"
        tensors: Dict[str, torch.Tensor] = {}
        sids: List[str] = []
        for sid, hidden, ids in self._buffer:
            tensors[f"{sid}/hidden"] = hidden
            tensors[f"{sid}/input_ids"] = ids
            sids.append(sid)
        save_file(tensors, str(self.out_dir / fname))
        self._manifest["shards"].append(
            {"file": fname, "samples": sids, "num_tokens": self._buf_tokens}
        )
        self._write_manifest()
        self._completed.update(sids)
        self._buffer = []
        self._buffered = set()
        self._buf_tokens = 0

    def _write_manifest(self) -> None:
        """Durable atomic write: fsync the tmp file, then os.replace.

        The fsync matters: without it a hard power cut can let the rename
        survive while the file contents do not, leaving a truncated manifest
        that orphans every shard in the directory.
        """
        tmp = self.out_dir / (_MANIFEST + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps(self._manifest, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.out_dir / _MANIFEST)

    def close(self) -> None:
        """Flush the remaining buffer and finalize the manifest."""
        if not self._closed:
            self._flush()
            self._closed = True

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc) -> None:
        # Flush on clean exit only; on an exception the buffer is dropped and
        # the manifest still describes exactly what is on disk.
        if exc[0] is None:
            self.close()


class ShardReader:
    """Reader over a shard directory written by :class:`ShardWriter`."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        manifest_path = self.out_dir / _MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no manifest at {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._shard_of: Dict[str, str] = {}
        for shard in self.manifest["shards"]:
            for sid in shard["samples"]:
                self._shard_of[sid] = shard["file"]
        self._cache: Tuple[Optional[str], Dict[str, torch.Tensor]] = (None, {})

    @property
    def meta(self) -> Dict:
        """Extraction metadata recorded by the writer."""
        return self.manifest.get("meta", {})

    def sample_ids(self) -> List[str]:
        """All sample ids in shard order."""
        out: List[str] = []
        for shard in self.manifest["shards"]:
            out.extend(shard["samples"])
        return out

    def __len__(self) -> int:
        return len(self._shard_of)

    def _load_shard(self, fname: str) -> Dict[str, torch.Tensor]:
        if self._cache[0] != fname:
            self._cache = (fname, load_file(str(self.out_dir / fname)))
        return self._cache[1]

    def read_sample(self, sample_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """(hidden [T, H], input_ids [T]) for one sample."""
        sid = str(sample_id)
        if sid not in self._shard_of:
            raise KeyError(f"sample '{sid}' not in manifest")
        tensors = self._load_shard(self._shard_of[sid])
        return tensors[f"{sid}/hidden"], tensors[f"{sid}/input_ids"]

    def samples(self) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
        """Iterate (sample_id, hidden, input_ids) in shard order (one shard in RAM at a time)."""
        for shard in self.manifest["shards"]:
            tensors = self._load_shard(shard["file"])
            for sid in shard["samples"]:
                yield sid, tensors[f"{sid}/hidden"], tensors[f"{sid}/input_ids"]
