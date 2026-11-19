# Speculative Decoding Research — working agreements

Target/verifier: `meta-llama/Meta-Llama-3-8B-Instruct` — **frozen, never trained**.
Drafter: EAGLE-2/HASS-style, 1 transformer layer, hidden 4096 — **the only trainable component**.
The unifying variable is the draft horizon **K** (top-level config key; `tree.max_depth`
and `train.rollout_K` interpolate from it).

## Infrastructure (GPU pod)

- **`/workspace` is the persistent network volume.** Checkpoints, extracted features,
  results — anything that must survive a pod restart — go under `/workspace`
  (that is what `paths.*` in `configs/base.yaml` defaults to).
- **`/root` is EPHEMERAL. Never write checkpoints, features, or results there.**
  Pip/HF caches are the only acceptable losses.
- **All long-running jobs run under tmux** (`tmux new -s <job>`, reattach with
  `tmux attach -t <job>`). Never launch training or extraction on a bare SSH session.
- `meta-llama` models are gated: `HF_TOKEN` must be set in the pod environment.
- **Pod stop/resume**: stopping loses only the container (pip deps, tmux,
  /root — including .bashrc). The volume keeps repo + bare git remote,
  features, checkpoints, results, HF cache, and the token file
  `/workspace/.secrets/hf_token` (chmod 600, never in git). To resume on a
  fresh pod: `bash /workspace/specdec/scripts/pod_setup.sh`, then on the
  laptop `git remote set-url pod ssh://root@<IP>:<PORT>/workspace/specdec.git`.

## Local (Windows) scaffolding

- Everything imports and runs on CPU. Override paths when running locally:
  `--set paths.root=./runs` (use forward slashes in overrides, even on Windows).
- Keep bulk data (features, checkpoints) OUT of OneDrive — point `paths.root`
  somewhere unsynced.
- Local venv: `%USERPROFILE%\venvs\specdec` (created with `--system-site-packages`
  over the Anaconda torch 2.3.1). Run tests from repo root:
  `~/venvs/specdec/Scripts/python -m pytest`

## Modeling ground rules

- The target is loaded frozen (`eval()`, `requires_grad_(False)`) in
  `src/common/modeling.py`. It never receives gradients.
- The drafter (`src/models/drafter.py`) borrows the target's embedding, final norm,
  and lm-head as **bound frozen modules** — they are deliberately kept out of its
  `state_dict()` and `parameters()`, so optimizers only ever see drafter weights.
- Feature layer: `data.layer_index` (default `-2`, i.e. `hidden_states[-2]` =
  last decoder layer INPUT, pre-norm). In HF transformers, `hidden_states[-1]`
  is the POST-final-norm hidden state — so with `layer_index=-1` you must set
  `drafter.apply_final_norm: false` (then `lm_head(feature)` reproduces the
  target's logits EXACTLY; keeping the norm on would double-apply it).
  `validate_feature_config` in src/common/modeling.py rejects the invalid
  pairing at train/eval startup. For any other layer index the recomputed
  logits in the training CE loss are an approximation (EAGLE-style
  simplification). Keep the index configurable; revisit before real training.

## Correctness rules

- Tests use `configs/tiny.yaml` (random-weight 2-layer hidden-128 target) and must
  **never download weights**.
- Device handling only via `src/common/device.py` (`resolve_device` /
  `resolve_dtype` / `sync`). **No bare `.cuda()` anywhere** — enforced by
  `tests/test_device.py::test_no_bare_cuda_calls_anywhere`.
- All sampling/verification randomness is drawn ON CPU with a CPU
  `torch.Generator` (RNG contract in `src/decode/verify.py`) — torch
  generators are device-bound, and CPU-side RNG keeps seeded runs
  reproducible on any device. Do not "optimize" this to device-side RNG
  without reworking the contract.
- Losslessness of verification (`src/decode/verify.py`) is a hard invariant,
  guarded by `tests/test_lossless_stats.py` (statistical, sampling mode) and the
  greedy-equivalence test in `tests/test_generate.py` (exact, greedy mode).
  Never weaken these tests to make a change pass.
- Sampling-mode tree drafting draws sibling tokens **i.i.d. WITH replacement**
  from the drafter distribution — that is the hypothesis of the multi-candidate
  losslessness theorem. Deduplicating siblings breaks the guarantee.

## Commands

```
# tests (CPU, ~1-2 min)
python -m pytest

# Phase 0 — first thing to run on the pod
python scripts/profile_verifier.py --config configs/base.yaml --set profiling.prefix_len=1024

# feature extraction (tmux!)
python scripts/extract_features.py --config configs/base.yaml --set data.dry_run=true   # size estimate only

# training / eval — ON THE POD always add the pod overrides (base.yaml keeps
# CPU-exact defaults for tests; the comments in base.yaml mark pod values):
python scripts/train_drafter.py --config configs/base.yaml \
    --set K=8 train.target_dtype=bfloat16 train.allow_tf32=true
python scripts/run_eval.py --config configs/base.yaml
```

Config convention: every script takes `--config <yaml> [<yaml>...]` (merged
left-to-right) and `--set key=value ...` (OmegaConf dotlist, highest priority).
