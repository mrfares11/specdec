"""End-to-end speculative generation: draft → tree-verify → commit.

Two cache strategies, selectable via ``kv_gather`` (config ``decode.kv_gather``):

  - GATHER (default): after verification, the accepted path's KV rows are
    index-selected out of the tree forward's cache (their RoPE positions
    ``C + depth`` are exactly their final absolute positions), and only the
    single final token (correction/bonus) is forwarded to obtain the next
    root distribution and its feature. Accepted-token features come from the
    tree forward's hidden states. One tree forward + one 1-token forward per
    step.
  - REPLAY (fallback, ``kv_gather=False``): crop the cache back to the
    committed prefix and re-forward all newly committed tokens — the original
    correctness-first scaffold path, kept for A/B validation.

The drafter cache is incremental across steps: it holds (token, true-feature)
pairs for the committed context (extended by ``a + 1`` pairs per step, using
target features gathered from the tree forward), plus drafted-node rows that
are cropped off after each verification. No O(C) drafter re-prefill per step.

Per-phase wall-time decomposition (RQ3 groundwork) via ``profile_timing=True``:
sync-bracketed timers around draft / tree-forward / verify / commit. Adds
synchronization overhead — leave off except when measuring.

RNG contract: see ``verify.py`` — all sampling on CPU with a CPU generator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from ..common.device import sync
from ..common.modeling import FeatureTap, crop_cache
from ..models.drafter import Drafter, DrafterCache
from .tree import DraftTree, build_from_parents
from .verify import verify_tree, verify_tree_greedy

_TEMP_EPS = 1e-5


@dataclass
class GenerationResult:
    """Outcome of one generation run.

    Attributes:
        tokens: committed continuation (prompt excluded).
        accept_lengths: accepted draft tokens per verification step.
        num_steps: number of draft/verify steps taken.
        timings: per-phase total seconds (only when ``profile_timing``):
            keys ``draft``, ``tree_forward``, ``verify``, ``commit``.
    """

    tokens: List[int]
    accept_lengths: List[int] = field(default_factory=list)
    num_steps: int = 0
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def mean_accept_length(self) -> float:
        """Mean accepted draft tokens per step (0 if no steps ran)."""
        if not self.accept_lengths:
            return 0.0
        return sum(self.accept_lengths) / len(self.accept_lengths)


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Temperature softmax in float32, ON CPU (RNG contract; see verify.py)."""
    t = max(float(temperature), _TEMP_EPS)
    return torch.softmax(logits.detach().float() / t, dim=-1).cpu()


def _pick(probs: torch.Tensor, greedy: bool, generator: Optional[torch.Generator]) -> int:
    if greedy:
        return int(probs.argmax().item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


class _Timer:
    """Optionally sync-bracketed phase timer accumulating into a dict."""

    def __init__(self, enabled: bool, device: torch.device):
        self.enabled = enabled
        self.device = device
        self.totals: Dict[str, float] = {}
        self._t0 = 0.0
        self._phase: Optional[str] = None

    def start(self, phase: str) -> None:
        if not self.enabled:
            return
        sync(self.device)
        self._phase = phase
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if not self.enabled or self._phase is None:
            return
        sync(self.device)
        self.totals[self._phase] = self.totals.get(self._phase, 0.0) + (
            time.perf_counter() - self._t0
        )
        self._phase = None

    def split(self, phase: str) -> None:
        """Close the running phase and open another.

        The timer holds one active phase, so a nested start() would silently
        drop the outer one. split() keeps phases DISJOINT and additive, which is
        what the per-phase breakdown needs: topology and mask construction are
        carved out of drafting rather than double-counted inside it.
        """
        self.stop()
        self.start(phase)


def _build_level_plans(
    tree: DraftTree, dtype: torch.dtype, device: torch.device
) -> List[Dict]:
    """Precompute per-level static state (once per generation call).

    Per level: the additive mask block over NODE columns (drafted-so-far
    ancestors + within-level self-eye — the committed-context columns are
    all-visible and get prepended as zeros each step), plus per-parent child
    index tensors for sync-free token assignment. Removes the O(levels × N)
    Python mask loop from every draft step, which otherwise serializes the
    CPU against the GPU kernel queue (the single-device form of "overlap").
    """
    depth_list = tree.depth.tolist()
    levels: List[List[int]] = [[] for _ in range(tree.max_depth)]
    for i, d in enumerate(depth_list):
        levels[d].append(i)
    neg = torch.finfo(dtype).min
    plans: List[Dict] = []
    for d, level in enumerate(levels):
        n_prev = sum(len(levels[e]) for e in range(d))
        size = len(level)
        block = torch.full((1, 1, size, n_prev + size), neg, dtype=dtype, device=device)
        for row, i in enumerate(level):
            for j in range(tree.num_nodes):
                if depth_list[j] < d and bool(tree.attn_mask[i, j]):
                    block[0, 0, row, j] = 0.0  # BFS order: node j occupies column j
            block[0, 0, row, n_prev + row] = 0.0
        parents = []
        for p in sorted({int(tree.parent[i].item()) for i in level}):
            kids = tree.children_of(p)
            parents.append((p, torch.tensor(kids, dtype=torch.long, device=device)))
        plans.append({
            "level": level,
            "level_t": torch.tensor(level, dtype=torch.long, device=device),
            "block": block,
            "parents": parents,
        })
    return plans


def _draft_levels(
    drafter: Drafter,
    tree: DraftTree,
    plans: List[Dict],
    dcache: DrafterCache,
    ctx_rows: int,
    context_len: int,
    q0: torch.Tensor,
    feat_root_prev: torch.Tensor,
    *,
    temperature: float,
    greedy: bool,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, DrafterCache]:
    """Expand the drafter over the tree, level by level, on top of ``dcache``.

    Args:
        dcache: drafter cache holding ``ctx_rows`` committed-context rows.
        context_len: committed token count C (absolute positions of depth-d
            nodes are ``C + d``).
        q0: slot-0 drafter LOGITS (device tensor, over depth-0 candidates).
        feat_root_prev: [1, 1, H] drafter-input feature for depth-0 nodes —
            the true target feature of the LAST committed token.

    Returns:
        (node_tokens [N] on device,
         draft_probs [N+1, V] on CPU in sampling mode / None in greedy mode
         (greedy verification never reads drafter distributions),
         extended cache).
    Sampling mode draws sibling tokens i.i.d. WITH replacement (losslessness
    hypothesis); greedy mode takes distinct top-k per parent.
    """
    device = feat_root_prev.device
    n = tree.num_nodes
    node_tokens = torch.full((n,), -1, dtype=torch.long, device=device)
    # Greedy: keep per-slot LOGITS on device (topk needs no softmax, verify
    # needs no drafter dists at all). Sampling: CPU probs per RNG contract.
    slot_dev: List[Optional[torch.Tensor]] = [None] * (n + 1)
    draft_probs: Optional[torch.Tensor] = None
    if greedy:
        slot_dev[0] = q0
    else:
        vocab = q0.shape[-1]
        draft_probs = torch.empty(n + 1, vocab, dtype=torch.float32)
        draft_probs[0] = _probs(q0, temperature)
    node_feat: List[Optional[torch.Tensor]] = [None] * n

    for d, plan in enumerate(plans):
        level = plan["level"]
        for p, kid_idx in plan["parents"]:
            if greedy:
                picks = torch.topk(slot_dev[p + 1], k=kid_idx.shape[0]).indices
            else:
                picks = torch.multinomial(
                    draft_probs[p + 1], kid_idx.shape[0], replacement=True, generator=generator
                ).to(device)
            node_tokens[kid_idx] = picks  # device-side, no per-child sync

        toks = node_tokens[plan["level_t"]].unsqueeze(0)
        prev = torch.cat(
            [
                (feat_root_prev if int(tree.parent[i].item()) == -1 else node_feat[int(tree.parent[i].item())])
                for i in level
            ],
            dim=1,
        )
        pos_lvl = torch.full((1, len(level)), context_len + d, dtype=torch.long, device=device)

        # context columns all-visible + precomputed static node-column block
        block = plan["block"]
        bias = torch.cat(
            [
                torch.zeros(1, 1, len(level), ctx_rows, dtype=block.dtype, device=device),
                block,
            ],
            dim=-1,
        )

        h_lvl, dcache = drafter(toks, prev, pos_lvl, attn_bias=bias, cache=dcache)
        logits_lvl = drafter.logits(h_lvl)
        for row, i in enumerate(level):
            if greedy:
                slot_dev[i + 1] = logits_lvl[0, row]
            else:
                draft_probs[i + 1] = _probs(logits_lvl[0, row], temperature)
            node_feat[i] = h_lvl[:, row : row + 1]

    return node_tokens, draft_probs, dcache


def _draft_dynamic(
    drafter: Drafter,
    level_widths: List[int],
    dcache: DrafterCache,
    ctx_rows: int,
    context_len: int,
    q0: torch.Tensor,
    feat_root_prev: torch.Tensor,
) -> Tuple[torch.Tensor, List[int], DrafterCache]:
    """Greedy DYNAMIC drafting: per-level global parent competition (M4).

    Holds the per-level node budget (``level_widths``, from the static
    allocation) fixed, but instead of static per-parent quotas, each level's
    ``w_d`` slots go to the globally best (parent, token) pairs ranked by
    cumulative drafter log-probability — the EAGLE-2 reallocation idea under
    a fixed budget profile, so the static-vs-dynamic comparison isolates
    topology choice alone.

    GREEDY ONLY: deterministic top-k structure selection violates the
    i.i.d.-with-replacement hypothesis of the sampling-mode losslessness
    theorem; callers must reject ``dynamic + sampling`` (enforced in
    :func:`speculative_generate`). Greedy losslessness is unaffected — the
    verifier never trusts the drafter regardless of topology.

    Returns (node_tokens [N] device, BFS parent list, extended cache).
    """
    device = feat_root_prev.device
    # All per-level work stays ON DEVICE (vectorized bias via an incremental
    # ancestor matrix; no per-element writes, no per-level GPU->CPU sync) —
    # asymmetric scaffolding overhead would bias the static-vs-dynamic timing
    # comparison this mode exists to make. ONE sync at the end (parent list).
    frontier_cum = torch.zeros(1, device=device)
    frontier_logits = q0.unsqueeze(0)  # [F, V]
    frontier_feat = feat_root_prev  # [1, F, H]
    frontier_off = 0  # per-step node id of the first frontier node (d>0)

    parent_chunks: List[torch.Tensor] = []
    token_chunks: List[torch.Tensor] = []
    anc = torch.zeros(0, 0, dtype=torch.bool, device=device)  # [n, n] ancestors
    n_prev = 0

    for d, w in enumerate(level_widths):
        logp = torch.log_softmax(frontier_logits.float(), dim=-1)
        k = min(w, logp.shape[-1])
        topv, topi = torch.topk(logp, k=k, dim=-1)  # [F, k]
        scores = frontier_cum.unsqueeze(1) + topv  # [F, k]
        flat = scores.flatten()
        take = min(w, flat.shape[0])
        win = torch.topk(flat, k=take).indices  # global competition
        win = win.sort().values  # ascending flat idx == (parent, rank): BFS-stable
        f_row = win // k
        c_col = win % k

        lvl_tokens = topi[f_row, c_col]  # [take] device
        if d == 0:
            par_ids = torch.full((take,), -1, dtype=torch.long, device=device)
        else:
            par_ids = frontier_off + f_row  # absolute per-step node ids
        parent_chunks.append(par_ids)
        token_chunks.append(lvl_tokens)
        prev_feat = frontier_feat[:, f_row, :]  # parent features, drafter input

        # ancestor rows of the new nodes: parent's row + the parent column
        anc_new = torch.zeros(take, n_prev + take, dtype=torch.bool, device=device)
        if d > 0:
            anc_new[:, :n_prev] = anc[par_ids]
            anc_new[torch.arange(take, device=device), par_ids] = True
        vis = anc_new.clone()
        vis[torch.arange(take, device=device), n_prev + torch.arange(take, device=device)] = True

        neg = torch.finfo(frontier_feat.dtype).min
        block = torch.full((1, 1, take, n_prev + take), neg,
                           dtype=frontier_feat.dtype, device=device)
        block.masked_fill_(vis[None, None], 0.0)
        bias = torch.cat(
            [torch.zeros(1, 1, take, ctx_rows, dtype=block.dtype, device=device), block],
            dim=-1,
        )
        pos = torch.full((1, take), context_len + d, dtype=torch.long, device=device)
        h_lvl, dcache = drafter(lvl_tokens.unsqueeze(0), prev_feat, pos,
                                attn_bias=bias, cache=dcache)

        grown = torch.zeros(n_prev + take, n_prev + take, dtype=torch.bool, device=device)
        if n_prev:
            grown[:n_prev, :n_prev] = anc
        grown[n_prev:] = anc_new
        anc = grown

        frontier_off = n_prev
        frontier_cum = flat[win]
        frontier_logits = drafter.logits(h_lvl)[0]
        frontier_feat = h_lvl
        n_prev += take

    parents = torch.cat(parent_chunks).tolist()  # the ONE host sync
    return torch.cat(token_chunks), parents, dcache


def _build_fold_block(tree: DraftTree, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Static additive block [1, 1, N+1, N+1] over [pending | nodes] columns.

    Row 0 is the pending (last-committed-but-uncached) token: within the block
    it sees only itself. Row i+1 is node i: it sees the pending root, its
    ancestors, and itself. The all-visible committed-prefix columns get
    prepended per step (cheap cat).
    """
    n = tree.num_nodes
    neg = torch.finfo(dtype).min
    block = torch.full((n + 1, n + 1), neg, dtype=dtype, device=device)
    block[0, 0] = 0.0
    block[1:, 0] = 0.0
    inner = torch.full((n, n), neg, dtype=dtype, device=device)
    inner.masked_fill_(tree.attn_mask.to(device), 0.0)
    block[1:, 1:] = inner
    return block[None, None]


@torch.no_grad()
def _generate_folded(
    target,
    drafter: Drafter,
    tree: DraftTree,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float,
    greedy: bool,
    tap: FeatureTap,
    generator: Optional[torch.Generator],
    eos_token_id: Optional[int],
    profile_timing: bool,
    dynamic_tree: bool = False,
) -> GenerationResult:
    """Commit-fold decode loop: ONE target forward per step.

    The pending token (previous step's correction/bonus) rides as row 0 of the
    tree forward — its logits give the root distribution and its hidden state
    the feature the next drafter extension needs, eliminating the separate
    commit forward entirely. Depth-0 drafting uses the drafter's own predicted
    feature for the pending position (the EAGLE self-rollout convention —
    the target feature does not exist until the fold forward runs). Greedy
    output remains exactly equal to autoregressive greedy decoding (tested).
    """
    device = input_ids.device
    timer = _Timer(profile_timing, device)

    timer.start("prefill")
    out = target(input_ids=input_ids, use_cache=True, output_hidden_states=True)
    cache = out.past_key_values
    feats = tap.take(out)
    timer.stop()
    c = int(input_ids.shape[1])
    param_dtype = next(target.parameters()).dtype

    committed: List[int] = []
    accept_lengths: List[int] = []
    pending = _pick(_probs(out.logits[0, -1], temperature), greedy, generator)
    feat_last = feats[:, -1:, :]  # true feature of the last IN-CACHE token

    if c >= 2:
        d_tokens, d_prev = input_ids[:, 1:], feats[:, : c - 1]
        d_pos = torch.arange(1, c, device=device).unsqueeze(0)
    else:
        d_tokens, d_prev = input_ids, feats.new_zeros(1, 1, feats.shape[-1])
        d_pos = torch.arange(0, 1, device=device).unsqueeze(0)
    _h, dcache = drafter(d_tokens, d_prev, d_pos, attn_bias=None, cache=None)
    ctx_rows = dcache.seq_length()

    if dynamic_tree:
        level_widths = torch.bincount(tree.depth).tolist()  # same budget profile
        plans = fold_block = fold_pos_tail = None
    else:
        plans = _build_level_plans(tree, feats.dtype, device)
        fold_block = _build_fold_block(tree, param_dtype, device)
        fold_pos_tail = (tree.depth + 1).unsqueeze(0).to(device)  # + c at use time
    hit_eos = False
    steps = 0

    while len(committed) < max_new_tokens and not hit_eos:
        # --- drafter pair for the pending token; its output feature seeds depth 0
        timer.start("draft")
        p_tok = torch.tensor([[pending]], dtype=torch.long, device=device)
        p_pos = torch.tensor([[c]], dtype=torch.long, device=device)
        h_p, dcache = drafter(p_tok, feat_last, p_pos, attn_bias=None, cache=dcache)
        ctx_rows = dcache.seq_length()
        q0 = drafter.logits(h_p[:, -1])[0]
        if dynamic_tree:
            node_tokens, parents, dcache = _draft_dynamic(
                drafter, level_widths, dcache, ctx_rows, c + 1, q0, h_p[:, -1:, :],
            )
            timer.split("topology")
            tree_s = build_from_parents(parents)
            draft_probs = None
            timer.split("mask")
            fold_block_s = _build_fold_block(tree_s, param_dtype, device)
            pos_tail = (tree_s.depth + 1).unsqueeze(0).to(device) + c
            timer.split("draft")
        else:
            node_tokens, draft_probs, dcache = _draft_levels(
                drafter, tree, plans, dcache, ctx_rows, c + 1, q0, h_p[:, -1:, :],
                temperature=temperature, greedy=greedy, generator=generator,
            )
            tree_s = tree
            fold_block_s = fold_block
            pos_tail = fold_pos_tail + c
        timer.stop()

        # --- ONE target forward over [pending | tree]
        timer.start("tree_forward")
        inp = torch.cat([p_tok, node_tokens.unsqueeze(0)], dim=1)
        pos = torch.cat([p_pos, pos_tail], dim=1)
        bias = torch.cat(
            [
                torch.zeros(1, 1, tree_s.num_nodes + 1, c, dtype=param_dtype, device=device),
                fold_block_s,
            ],
            dim=-1,
        )
        out_t = target(
            input_ids=inp, past_key_values=cache, attention_mask=bias,
            position_ids=pos, use_cache=True, output_hidden_states=True,
        )
        cache = out_t.past_key_values
        logits = out_t.logits[0]                      # [N+1, V]; row 0 = root slot
        fold_feats = tap.take(out_t)[0]               # row 0 = f_pending (true)
        timer.stop()

        # --- verify (slot layout matches: row i+1 is node i's slot)
        timer.start("verify")
        if greedy:
            outcome = verify_tree_greedy(tree_s, node_tokens, logits.argmax(dim=-1))
        else:
            target_probs = _probs(logits, temperature)
            outcome = verify_tree(
                tree_s, node_tokens, draft_probs, target_probs,
                generator=generator, greedy=False,
            )
        accept_lengths.append(outcome.num_accepted)
        steps += 1
        timer.stop()

        # --- fold commit: keep pending row + accepted path; extend drafter with true feats
        timer.start("commit")
        a = outcome.num_accepted
        cache = _gather_cache(cache, c + 1, outcome.path)  # pending row at c is kept
        dcache.crop(ctx_rows)
        new_tokens = [pending] + outcome.tokens[:a]
        f_pending = fold_feats[0:1].unsqueeze(0)  # [1, 1, H] true feature of pending
        if a > 0:
            # node i's feature sits at fold row i + 1 (row 0 is the pending token)
            acc_feats = fold_feats.index_select(
                0, torch.tensor([p + 1 for p in outcome.path], dtype=torch.long, device=device)
            ).unsqueeze(0)
            acc_t = torch.tensor([outcome.tokens[:a]], dtype=torch.long, device=device)
            acc_pos = torch.arange(c + 1, c + 1 + a, device=device).unsqueeze(0)
            _h2, dcache = drafter(
                acc_t,
                torch.cat([f_pending, acc_feats[:, :-1]], dim=1),
                acc_pos,
                attn_bias=None,
                cache=dcache,
            )
            ctx_rows = dcache.seq_length()
            feat_last = acc_feats[:, -1:, :]
        else:
            feat_last = f_pending
        committed.extend(new_tokens)
        if eos_token_id is not None and eos_token_id in new_tokens:
            hit_eos = True
        c += 1 + a
        pending = outcome.tokens[a]
        timer.stop()

    if eos_token_id is not None and eos_token_id in committed:
        committed = committed[: committed.index(eos_token_id) + 1]
    return GenerationResult(
        committed[:max_new_tokens], accept_lengths, steps, dict(timer.totals)
    )


def _gather_cache(cache, keep_prefix: int, path: List[int]):
    """Keep [0, keep_prefix) plus the accepted tree rows (at prefix + node_idx)."""
    legacy = cache.to_legacy_cache()
    device = legacy[0][0].device
    idx = torch.cat(
        [
            torch.arange(keep_prefix, device=device),
            torch.tensor([keep_prefix + p for p in path], dtype=torch.long, device=device),
        ]
    )
    from transformers.cache_utils import DynamicCache  # local: version-tolerant

    return DynamicCache.from_legacy_cache(
        tuple(
            (k.index_select(2, idx), v.index_select(2, idx)) for k, v in legacy
        )
    )


@torch.no_grad()
def speculative_generate(
    target,
    drafter: Drafter,
    tree: DraftTree,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    greedy: bool = False,
    layer_index: int = -2,
    tap: Optional[FeatureTap] = None,
    generator: Optional[torch.Generator] = None,
    eos_token_id: Optional[int] = None,
    kv_gather: bool = True,
    commit_fold: bool = True,
    profile_timing: bool = False,
    dynamic_tree: bool = False,
) -> GenerationResult:
    """Generate with draft-tree speculation, losslessly w.r.t. the target.

    ``commit_fold=True`` (default) uses the single-forward-per-step folded
    loop (:func:`_generate_folded`); False falls back to the two-forward loop
    with the ``kv_gather``/replay cache strategies. Greedy output must match
    plain greedy decoding token-for-token in EVERY mode (tested).

    ``tap`` overrides plain ``hidden_states[layer_index]`` feature extraction
    (needed for imported EAGLE drafters — see :class:`FeatureTap`).
    """
    if tap is None:
        tap = FeatureTap("index", layer_index)
    if dynamic_tree:
        if not greedy:
            raise ValueError(
                "dynamic_tree requires greedy mode: deterministic structure "
                "selection breaks the i.i.d. sampling-losslessness hypothesis"
            )
        if not commit_fold:
            raise ValueError("dynamic_tree is implemented for the commit_fold loop only")
    was_training = drafter.training
    drafter.eval()
    try:
        if commit_fold:
            return _generate_folded(
                target, drafter, tree, input_ids, max_new_tokens,
                temperature=temperature, greedy=greedy, tap=tap,
                generator=generator, eos_token_id=eos_token_id,
                profile_timing=profile_timing, dynamic_tree=dynamic_tree,
            )
        device = input_ids.device
        timer = _Timer(profile_timing, device)

        # --- prompt prefill (target + drafter)
        timer.start("prefill")
        out = target(input_ids=input_ids, use_cache=True, output_hidden_states=True)
        cache = out.past_key_values
        feats = tap.take(out)  # [1, C, H]
        c = int(input_ids.shape[1])
        timer.stop()

        committed: List[int] = []
        accept_lengths: List[int] = []
        first = _pick(_probs(out.logits[0, -1], temperature), greedy, generator)

        # Drafter prefill over (t_i, f_{i-1}) for i >= 1 — training pairing.
        if c >= 2:
            d_tokens = input_ids[:, 1:]
            d_prev = feats[:, : c - 1]
            d_pos = torch.arange(1, c, device=device).unsqueeze(0)
        else:  # degenerate 1-token prompt: single zero-feature cold-start pair
            d_tokens = input_ids
            d_prev = feats.new_zeros(1, 1, feats.shape[-1])
            d_pos = torch.arange(0, 1, device=device).unsqueeze(0)
        _h_ctx, dcache = drafter(d_tokens, d_prev, d_pos, attn_bias=None, cache=None)
        ctx_rows = dcache.seq_length()
        hit_eos = False
        plans = _build_level_plans(tree, feats.dtype, device)

        # Tokens committed but not yet folded into caches, with the features
        # of their PREDECESSORS (drafter input pairing).
        pending_tokens: List[int] = [first]
        pending_prev_feats = feats[:, -1:, :]  # f_{C-1}
        feat_last_committed: Optional[torch.Tensor] = None  # set by first commit

        steps = 0
        while True:
            # --- commit pending tokens into both caches
            timer.start("commit")
            n_new = len(pending_tokens)
            new_t = torch.tensor([pending_tokens], dtype=torch.long, device=device)
            pos = torch.arange(c, c + n_new, device=device).unsqueeze(0)

            # Gather path: this is a 1-token forward (the pending correction/
            # bonus). Replay path: all newly committed tokens re-forward here.
            out = target(
                input_ids=new_t, past_key_values=cache, position_ids=pos,
                use_cache=True, output_hidden_states=True,
            )
            cache = out.past_key_values
            new_feats = tap.take(out)
            root_logits = out.logits[0, -1]

            # drafter extension pairs: (t_i, f_{i-1}) for the pending tokens
            h_ext, dcache = drafter(
                new_t,
                torch.cat([pending_prev_feats, new_feats[:, :-1]], dim=1),
                pos,
                attn_bias=None,
                cache=dcache,
            )
            ctx_rows = dcache.seq_length()
            q0 = drafter.logits(h_ext[:, -1])[0]  # raw device logits
            feat_last_committed = new_feats[:, -1:, :]

            committed.extend(pending_tokens)
            c += n_new
            timer.stop()

            if eos_token_id is not None and eos_token_id in pending_tokens:
                hit_eos = True
            if len(committed) >= max_new_tokens or hit_eos:
                break

            # --- draft
            timer.start("draft")
            node_tokens, draft_probs, dcache = _draft_levels(
                drafter, tree, plans, dcache, ctx_rows, c, q0, feat_last_committed,
                temperature=temperature, greedy=greedy, generator=generator,
            )
            timer.stop()

            # --- one target forward over the whole tree
            timer.start("tree_forward")
            param_dtype = next(target.parameters()).dtype
            bias = tree.attn_bias(c, dtype=param_dtype, device=device)
            tree_pos = tree.position_ids(c).to(device)
            out_tree = target(
                input_ids=node_tokens.unsqueeze(0),
                past_key_values=cache,
                attention_mask=bias,
                position_ids=tree_pos,
                use_cache=True,
                output_hidden_states=kv_gather,
            )
            tree_logits = out_tree.logits[0]
            cache = out_tree.past_key_values
            timer.stop()

            # --- accept / reject
            timer.start("verify")
            if greedy:
                # Device-side argmaxes; transfer N+1 ints, not [N+1, V] probs.
                slot_argmax = torch.cat(
                    [root_logits.argmax().view(1), tree_logits.argmax(dim=-1)]
                )
                outcome = verify_tree_greedy(tree, node_tokens, slot_argmax)
            else:
                target_probs = torch.empty_like(draft_probs)
                target_probs[0] = _probs(root_logits, temperature)
                target_probs[1:] = _probs(tree_logits, temperature)
                outcome = verify_tree(
                    tree, node_tokens, draft_probs, target_probs,
                    generator=generator, greedy=False,
                )
            accept_lengths.append(outcome.num_accepted)
            steps += 1
            timer.stop()

            # --- fold accepted path into caches; final token stays pending
            timer.start("commit")
            a = outcome.num_accepted
            if kv_gather:
                cache = _gather_cache(cache, c, outcome.path)
                dcache.crop(ctx_rows)
                if a > 0:
                    tree_feats = tap.take(out_tree)[0]  # [N, H]
                    acc_idx = torch.tensor(outcome.path, dtype=torch.long, device=device)
                    acc_feats = tree_feats.index_select(0, acc_idx).unsqueeze(0)
                    acc_t = torch.tensor([outcome.tokens[:a]], dtype=torch.long, device=device)
                    acc_pos = torch.arange(c, c + a, device=device).unsqueeze(0)
                    h_acc, dcache = drafter(
                        acc_t,
                        torch.cat([feat_last_committed, acc_feats[:, :-1]], dim=1),
                        acc_pos,
                        attn_bias=None,
                        cache=dcache,
                    )
                    ctx_rows = dcache.seq_length()
                    committed.extend(outcome.tokens[:a])
                    if eos_token_id is not None and eos_token_id in outcome.tokens[:a]:
                        hit_eos = True
                    c += a
                    feat_last_committed = acc_feats[:, -1:, :]
                pending_tokens = [outcome.tokens[a]]
                pending_prev_feats = feat_last_committed
            else:
                cache = crop_cache(cache, c)
                dcache.crop(ctx_rows)
                pending_tokens = outcome.tokens
                pending_prev_feats = feat_last_committed
            timer.stop()

        if eos_token_id is not None and eos_token_id in committed:
            committed = committed[: committed.index(eos_token_id) + 1]
        return GenerationResult(
            committed[:max_new_tokens], accept_lengths, steps, dict(timer.totals)
        )
    finally:
        if was_training:
            drafter.train()


@torch.no_grad()
def greedy_generate(
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    eos_token_id: Optional[int] = None,
) -> List[int]:
    """Plain autoregressive argmax decoding — the reference for equivalence tests."""
    device = input_ids.device
    out = target(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    cur = int(out.logits[0, -1].argmax().item())
    tokens = [cur]
    pos = int(input_ids.shape[1])
    while len(tokens) < max_new_tokens and cur != eos_token_id:
        out = target(
            input_ids=torch.tensor([[cur]], dtype=torch.long, device=device),
            past_key_values=cache,
            position_ids=torch.tensor([[pos]], dtype=torch.long, device=device),
            use_cache=True,
        )
        cache = out.past_key_values
        cur = int(out.logits[0, -1].argmax().item())
        tokens.append(cur)
        pos += 1
    return tokens
