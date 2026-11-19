"""Target (verifier) loading and KV-cache helpers.

The target is ALWAYS loaded frozen: ``eval()`` + ``requires_grad_(False)``.
``source: random`` builds a tiny random-weight Llama from config dims for CPU
tests — it never touches the network.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

LegacyCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]


def load_target(cfg: DictConfig, device: torch.device, dtype: torch.dtype):
    """Load the frozen verifier per ``cfg.target``.

    ``source: hub`` downloads ``cfg.target.model_id`` (gated models need
    ``HF_TOKEN``). ``source: random`` builds a random-weight
    :class:`LlamaForCausalLM` from ``cfg.target.random_config`` — used by all
    tests, no network access.
    """
    t = cfg.target
    if t.source == "random":
        params = OmegaConf.to_container(t.random_config, resolve=True)
        assert isinstance(params, dict)
        mconf = LlamaConfig(**params)
        mconf._attn_implementation = t.attn_implementation
        model = LlamaForCausalLM(mconf).to(device=device, dtype=dtype)
    elif t.source == "hub":
        model = AutoModelForCausalLM.from_pretrained(
            t.model_id,
            revision=t.revision,
            torch_dtype=dtype,
            attn_implementation=t.attn_implementation,
        ).to(device)
    else:
        raise ValueError(f"unknown target.source '{t.source}' (expected hub|random)")
    model.eval()
    model.requires_grad_(False)
    return model


def validate_feature_config(cfg: DictConfig) -> None:
    """Fail loudly on the invalid (layer_index, apply_final_norm) pairing.

    In HF transformers, ``hidden_states[-1]`` is the POST-final-norm hidden
    state (the norm is applied before the last entry is appended). With
    ``data.layer_index=-1`` the drafter head must therefore be applied WITHOUT
    re-applying the norm — ``drafter.apply_final_norm=false`` — which makes
    ``lm_head(feature)`` reproduce the target logits exactly. Keeping
    ``apply_final_norm=true`` there double-applies the norm and silently
    corrupts both the training CE target and decode-time proposals.

    ``data.feature_tap=prenorm_last`` selects the last decoder layer's OUTPUT
    before the final norm (not reachable by hidden_states indexing — see
    :class:`FeatureTap`). That feature space REQUIRES the norm on the head,
    so ``apply_final_norm=false`` is rejected there.
    """
    tap_mode = str(cfg.data.get("feature_tap", "index"))
    if tap_mode not in ("index", "prenorm_last"):
        raise ValueError(f"data.feature_tap='{tap_mode}' (expected index|prenorm_last)")
    if tap_mode == "prenorm_last":
        if not bool(cfg.drafter.apply_final_norm):
            raise ValueError(
                "data.feature_tap=prenorm_last selects a PRE-final-norm feature; "
                "the head must apply the norm — set drafter.apply_final_norm=true"
            )
        return  # layer_index is ignored by the tap
    if int(cfg.data.layer_index) == -1 and bool(cfg.drafter.apply_final_norm):
        raise ValueError(
            "data.layer_index=-1 selects the post-final-norm hidden state; set "
            "drafter.apply_final_norm=false (see validate_feature_config docstring)"
        )


class FeatureTap:
    """Config-selected extractor for the target features the drafter consumes.

    ``mode="index"`` returns ``out.hidden_states[layer_index]`` (project
    default: ``-2``, the last decoder layer's INPUT).

    ``mode="prenorm_last"`` returns the last decoder layer's OUTPUT before the
    final norm. HF never exposes this tensor in ``hidden_states`` (``[-2]`` is
    the last layer's input, ``[-1]`` is post-norm), so it is captured with a
    forward-pre-hook on ``model.norm``. The hook fires on every target forward;
    ``take()`` must be called right after the forward whose features are wanted.

    NOTE: official EAGLE drafters do NOT use this space — they consume the
    POST-norm ``hidden_states[-1]`` and apply lm_head raw, i.e. plain
    ``layer_index=-1`` + ``drafter.apply_final_norm=false`` (configs/eagle.yaml).
    The prenorm tap is kept as an experimentation knob.
    """

    def __init__(self, mode: str, layer_index: int, target=None):
        if mode not in ("index", "prenorm_last"):
            raise ValueError(f"unknown FeatureTap mode '{mode}'")
        self.mode = mode
        self.layer_index = int(layer_index)
        self._buf: Optional[torch.Tensor] = None
        self._handle = None
        if mode == "prenorm_last":
            if target is None:
                raise ValueError("prenorm_last tap needs the target model")
            self._handle = target.model.norm.register_forward_pre_hook(self._capture)

    def _capture(self, _module, args):
        self._buf = args[0]

    def take(self, out) -> torch.Tensor:
        """Features for the forward that produced ``out``."""
        if self.mode == "index":
            return out.hidden_states[self.layer_index]
        if self._buf is None:
            raise RuntimeError("prenorm_last tap: no forward captured yet")
        return self._buf

    def remove(self) -> None:
        """Detach the hook (idempotent)."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @classmethod
    def from_config(cls, cfg: DictConfig, target=None) -> "FeatureTap":
        return cls(
            str(cfg.data.get("feature_tap", "index")),
            int(cfg.data.layer_index),
            target=target,
        )


def load_tokenizer(cfg: DictConfig):
    """Tokenizer for the hub target. Random targets have no tokenizer."""
    if cfg.target.source != "hub":
        raise ValueError("tokenizers exist only for target.source=hub")
    return AutoTokenizer.from_pretrained(cfg.target.model_id, revision=cfg.target.revision)


def clone_prefix_cache(legacy: LegacyCache) -> DynamicCache:
    """Fresh :class:`DynamicCache` over a fixed prefix, safe to append to.

    In transformers 4.5x ``from_legacy_cache`` COPIES the prefix tensors
    (each layer's (k, v) goes through ``cat`` against an empty buffer), so
    this call costs a full prefix copy — which is exactly why the profiler
    invokes it OUTSIDE the timed region. What matters for the methodology is
    the isolation invariant: the ``legacy`` tuple is never mutated, so every
    timed iteration sees a cache of exactly ``prefix_len`` positions
    (guarded by ``tests/test_pipeline_smoke.py::test_prefix_cache_isolation``).
    """
    return DynamicCache.from_legacy_cache(legacy)


def crop_cache(cache: DynamicCache, length: int) -> DynamicCache:
    """Truncate a cache back to ``length`` positions (drop draft-tree rows)."""
    if hasattr(cache, "crop"):
        cache.crop(length)
        return cache
    legacy = tuple(
        (k[:, :, :length, :], v[:, :, :length, :]) for k, v in cache.to_legacy_cache()
    )  # pragma: no cover - fallback for older transformers
    return DynamicCache.from_legacy_cache(legacy)
