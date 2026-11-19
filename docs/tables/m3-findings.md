# M3 re-certification — findings

Pod B, fully version-controlled invocations (`scripts/pod-jobs/m3-train.sh`,
`m3-heldout.sh`). Supersedes the pod-A RQ2 numbers, which RESULTS.md already
flags as partially-reconstructed preliminary evidence.

**Fixed across every arm:** 80k-row ShareGPT feature set (stream seed 0),
batch 4 x accum 4, `rollout_K=8` unless noted, geometric-256 tree,
MT-Bench/HumanEval 80 prompts, 128 new tokens, greedy, `runtime.seed=0`.
All arms compared at **10k steps** unless the row says otherwise.

## 1. Arm-independent held-out rollout CE

`scripts/heldout_ce.py`, 400 dev samples from feature rows >= 100k (past both
the training range and the eval-prompt range), K=8, identical batching, masks
and shard order for every checkpoint. This metric exists because each arm's
*training* CE was computed under that arm's own step weights — the confound
that produced the original C2 claim.

| Arm (10k steps unless noted) | s1 | s2 | s4 | s8 | uniform | MT-Bench accept K=4 |
|---|---|---|---|---|---|---|
| sw-80k @ **20k steps** | 2.15 | 2.46 | 2.79 | 3.12 | **2.76** | 1.85 |
| sw-80k @ 10k steps | 2.43 | 2.73 | 3.02 | 3.33 | **3.33** | 1.69 |
| sw-40k | 2.59 | 3.18 | 3.69 | 4.14 | 3.62 | 1.57 |
| sw-20k | 2.78 | 3.41 | 3.92 | 4.37 | 3.85 | 1.52 |
| **tf-80k** (teacher-forced, rollout_K=1) | **2.29** | 3.97 | 6.71 | 12.33 | 7.62 | 1.49 |
| det-80k (detached rollout) | 3.04 | 3.46 | 3.60 | 3.76 | 3.56 | 1.37 |
| bp-80k (BP-through, uniform) | 3.30 | 3.66 | 3.80 | 3.96 | 3.77 | 1.34 |

## 2. The original C2 dissociation does NOT reproduce

The paper claims BP-uniform *improves* CE (3.3 vs 4.2) while *degrading*
acceptance (1.35 vs 1.50). On the arm-independent metric, BP-uniform is worse
than detached on **both** axes — CE 3.77 vs 3.56 and acceptance 1.34 vs 1.37 —
and det's per-step curve dominates bp's at **every one of the 8 steps**, so the
reversal is not an artifact of how the new metric weights steps. The pod-A
ordering was produced by comparing each arm's own step-weighted training CE.

The det-vs-bp acceptance gap also collapses: 0.15 on pod A (1.50 vs 1.35),
0.03 here (1.37 vs 1.34). Detached and BP-uniform are close to equivalent.

## 3. A stronger dissociation replaces it

Across the five step-matched arms, **multi-step CE is uncorrelated with
acceptance while step-1 CE tracks it**:

| Predictor | Spearman rho vs MT-Bench accept K=4 (n=6) |
|---|---|
| uniform (multi-step) held-out CE | **+0.37** |
| step-1 held-out CE | **+0.66** |

n=6 — descriptive, not significant. Reported as a direction, not a test.

**Corrected after the step-matched cell landed.** On the five arms available
before `m3-sw80k-s10`'s held-out CE existed, these read -0.10 and +0.70, and
the claim was written as "multi-step CE does not predict acceptance at all."
The missing arm has both the best uniform CE (3.33) and the best acceptance
(1.69), so adding it pulls the multi-step correlation up to +0.37. The
supportable claim is that step-1 CE is the BETTER predictor, not that
multi-step CE is uninformative. This is precisely what the step-matched cell
was queued to test, and it went against the original framing.

The teacher-forced control makes the point in a single arm: it has the **best
step-1 CE of any 10k-step arm** (2.29) and by far the **worst multi-step CE**
(7.62; 5.4x degradation s1->s8 vs 1.2-1.6x for every rollout arm), yet it still
**beats both rollout arms on acceptance** (1.49 vs 1.37 / 1.34). Training
through the rollout is not what buys acceptance; step-decayed weighting is.

This preserves the paper's mechanism ("step-1 proposals gate the tree") and its
prescription (decay), but the evidence is now the tf control and the rho gap
rather than a CE/acceptance inversion between det and bp.

## 4. The step-weighted win is much larger than reported

| Comparison | Pod A (paper) | Pod B (step-matched, 10k) |
|---|---|---|
| step-weighted vs detached, accept K=4 | 1.58 vs 1.50 (**+5%**) | 1.69 vs 1.37 (**+23%**) |

## 5. Data scaling (step-weighted, all at 10k steps)

| Feature rows | MT-Bench K=4 | HumanEval K=4 | held-out uniform CE |
|---|---|---|---|
| 20k | 1.52 | 1.81 | 3.85 |
| 40k | 1.57 | 1.86 | 3.62 |
| 80k | 1.69 | 2.07 | 3.33 |

Per-doubling increments do not shrink on either benchmark (+0.05 then +0.12 on
MT-Bench; +0.05 then +0.21 on HumanEval) and held-out CE falls monotonically —
so **no saturation is observed in the tested range**.

Wording corrected: this was written as "gains accelerate," which three data
points cannot support. The protocol's E6 makes the same point, and the
supportable claim is the absence of saturation over the range measured, not a
trend beyond it. The HumanEval 80k point is
now step-matched at 10k steps (2.07); the earlier 2.30 figure was the 20k-step
checkpoint and confounded data with steps.

## 6. Training-progress curve (sw-80k, MT-Bench)

| Steps | 2k | 5k | 10k | 20k |
|---|---|---|---|---|
| accept K=4 | 1.26 | 1.48 | 1.69 | 1.85 |
| accept K=8 | 1.26 | 1.49 | 1.74 | 1.95 |

Still climbing at 20k steps with no sign of a plateau — the headline drafter is
**under-trained, so every number reported for it is a lower bound**. Worth
stating explicitly rather than letting a reviewer infer convergence.

## 7. RQ3 loop ladder (K=8, geometric-256, baseline drafter, timers ON)

| Stage | accept | speedup |
|---|---|---|
| crop + replay | 1.97 | 0.85x |
| KV gather | 1.97 | 0.84x |
| commit-fold | 1.67 | 1.08x |

`decode.profile_timing=true` for all three, so these speedups are **not citable
wall-clock** — they exist to decompose per-phase cost. Note commit-fold trades
acceptance (1.97 -> 1.67) for the single-forward schedule and still wins
end-to-end; that trade should be stated, not hidden.

## 8. Cross-session determinism (validates the cross-milestone comparisons)

M5's `base-static-mtbench` cell re-ran the v1.0 baseline drafter in a fresh
session, on a different day, from a rebuilt chain:

| | M1 (n=3) | M5 cell |
|---|---|---|
| accept K=4 | 1.6390 | 1.6390 |
| accept K=8 | 1.6716 | 1.6716 |

Exact to four decimals — greedy acceptance is fully deterministic across
sessions. Comparisons that pair an M3 acceptance number against an M1 EAGLE
number are therefore sound, provided tree, prompts, tokens and seed match.

## 9. The EAGLE acceptance gap reverses under data scaling

Official EAGLE-LLaMA3-8B weights in our harness (same verifier, tree budget,
prompts, lossless verification):

| Benchmark | K | v1.0 drafter | scaled drafter | EAGLE | v1.0 vs EAGLE | scaled vs EAGLE |
|---|---|---|---|---|---|---|
| MT-Bench | 4 | 1.639 | 1.848 | 1.746 | -6.5% | **+5.6%** |
| MT-Bench | 8 | 1.672 | 1.950 | 1.810 | -8.3% | **+7.2%** |
| HumanEval | 4 | 1.948 | 2.296 | 2.141 | -9.9% | **+6.8%** |
| HumanEval | 8 | 2.037 | 2.535 | 2.339 | -14.8% | **+7.7%** |

The scaled drafter is one transformer layer trained on 80k rows for 20k steps
and it out-accepts the official release. Section 8 makes the cross-milestone
pairing defensible; M5 re-measures both systems in one session regardless.

## 10. The M4 dynamic-tree conclusion was a host-configuration artifact

M5's `base-dyn-mtbench` cell re-ran the M4 dynamic configuration in the same
session as its static control:

| MT-Bench, baseline drafter | accept K=4 | speedup K=4 | accept K=8 | speedup K=8 |
|---|---|---|---|---|
| static | 1.6390 | 1.20 | 1.6716 | 1.08 |
| dynamic (M5, capped host) | 2.2551 | **1.59** | 2.3317 | **1.37** |
| dynamic (M4, uncapped host) | 2.2551 | 0.80 | 2.3317 | 0.72 |

Acceptance is bit-identical; only wall-clock moves. Attribution:

| | M4 (pre-cap) | M5 (capped) | change |
|---|---|---|---|
| dynamic spec ms/token | 23.01 | 11.61 | **-49.5%** |
| dynamic AR ms/token | 18.35 | 18.42 | +0.4% |
| static spec ms/token | 15.37 | 15.04 | -2.1% |

The slowdown fell almost entirely on the CPU-heavy path (per-step topology
construction) and left GPU-bound paths alone — the signature of host thread
thrashing on a 224-core machine, fixed by `OMP/MKL_NUM_THREADS=8`
(commit 8ca7f57, dated after the M4 runs). `git log` shows **no changes to
`src/decode/` or `src/eval/`** between the two measurements, ruling out an
implementation change.

This also resolves the "+28.7 ms/step in the target tree forward despite
identical input shapes" anomaly that Sec 3.6 had flagged as under
investigation: it was kernel-launch and mask-construction work starved of CPU.

Confirmed on all three benchmarks and both horizons once the MBPP cells were
re-measured under the cap: static -> dynamic goes 1.20->1.59 / 1.08->1.37
(MT-Bench), 1.34->1.86 / 1.23->1.67 (HumanEval), 1.33->1.85 / 1.21->1.64
(MBPP). All six contaminated measurements read as losses (0.73-0.96x); all six
capped measurements read as substantial wins. The reversal is complete, not
partial.

**Consequences.** (i) The paper's C3 dynamic-tree conclusion reverses —
dynamic reallocation pays. (ii) M4's dynamic *timings* are unusable; its
*acceptance* numbers remain valid (deterministic). (iii) The M1 headline table
is also pre-cap, but static decode moved only ~2%, so those speedups are
conservative rather than wrong. (iv) MBPP dynamic timings WERE
re-measured under the cap (queued to the tail of the campaign for exactly this
reason), so no benchmark is left resting on a contaminated number.

**Process lesson:** the same-session symmetric cell is what caught this. A
cross-session comparison of two numbers that were each individually plausible
would have shipped an inverted conclusion.

## 11. Which measurements the thread artifact does and does not touch

Audited after finding it, because "what else is contaminated?" is the first
question this raises:

| Measurement | Ran | vs cap (2026-08-21 08:06 UTC) | Status |
|---|---|---|---|
| M4 dynamic timings | 2026-08-20 21:01 | BEFORE | **contaminated, superseded** |
| M1 headline timings | 2026-08-20 18:44 | BEFORE | static path moved ~2% -> conservative, usable |
| M3 RQ3 ladder | 2026-08-21 09:15-09:28 | after | clean |
| M3 arm evals | 2026-08-25 onward | after | clean |
| M5 factorial | 2026-08-28 | after | clean |

The ladder verifies independently rather than by timestamp alone:
`ladder-fold` (K=8, timers ON) reads accept 1.67 / 1.08x, and M5's capped
`base-static-mtbench` at K=8 reads 1.672 / 1.084x — same drafter, tree,
benchmark and budget. A contaminated ladder could not land on the capped
number.

Acceptance is unaffected everywhere: it is deterministic under greedy decoding
and has reproduced bit-identically across every session, host and thread
setting we have compared.

## 12. Symmetric EAGLE comparison — the dynamic tree TRANSFERS

Every system measured on both trees, same session, same verifier, prompts,
budget and seed (MT-Bench; HumanEval cells pending at time of writing):

| System | static acc | static spd | dynamic acc | dynamic spd | dynamic gain |
|---|---|---|---|---|---|
| base (v1.0) | 1.639 | 1.197x | 2.255 | 1.586x | +32.5% |
| **scaled (ours)** | **1.848** | **1.299x** | **2.571** | **1.762x** | **+35.6%** |
| official EAGLE | 1.746 | 1.266x | 2.408 | 1.684x | +33.0% |

Ours vs EAGLE: +5.9% / +7.7% acceptance on static (K=4/K=8), widening to
+6.8% / +10.0% on dynamic. Wall-clock lead is consistently smaller than the
acceptance lead (+2.6% static, +4.6% dynamic at K=4) — our own C1 thesis
applied to our own result.

**The important row is EAGLE's.** Dynamic reallocation buys the external
drafter +33.0%, nearly what it buys ours (+35.6%). The tree contribution is
therefore a general property of tree-based speculative decoding under a fixed
node budget, not an artifact co-tuned to our drafter. This pre-empts the
obvious reviewer objection ("your tree is tuned to your drafter") with our own
measurement rather than an argument, and it is a stronger claim than a
self-serving result would have been.

That our lead *widens* on the dynamic tree is also consistent with the
interaction effect in section 11's factorial: a sharper drafter gives global
parent competition better signal to reallocate on.

## Caveats to carry into the M5 writeup

- **Temperature-1 rows:** `src/eval/benchmark.py:195` always measures the AR
  baseline with `greedy_generate`, so the sampling-mode cells compare
  sampling-mode speculative decoding against a GREEDY autoregressive baseline.
  The latency difference between greedy and sampled AR is a multinomial draw
  per token (negligible), so the speedup stays meaningful — but the row should
  be footnoted, and acceptance is the primary number there regardless.
- **512-token rows** change only `eval.max_new_tokens`; prompts, tree and seed
  are otherwise identical to the 128-token cells, so they are a within-system
  horizon sweep, not an independent benchmark.
- **vLLM reference** is an optimized-engine anchor for external context, NOT a
  like-for-like lossless-speculation comparison; it must never be placed in the
  same column as our speedups.

## 13. Protocol-breadth extras (roadmap M5)

**512-token generation helps.** Same config, only `eval.max_new_tokens` changed
(MT-Bench, K=4, static):

| Drafter | 128 tokens | 512 tokens |
|---|---|---|
| v1.0 baseline | 1.639 acc / 1.197x | 1.700 / **1.281x** |
| scaled | 1.848 acc / 1.299x | 1.930 / **1.378x** |

Both acceptance and speedup rise with the longer horizon, so the headline
128-token numbers are CONSERVATIVE for realistic generation lengths. Worth
stating, since the natural reviewer assumption runs the other way.

**Sampling mode is lossless but not competitive in this implementation.**
Temperature 1.0, MT-Bench, K=4, static, v1.0 baseline:

| Mode | accept | spec ms/token | AR ms/token | speedup |
|---|---|---|---|---|
| greedy | 1.639 | 15.04 | 18.00 | 1.20x |
| temperature 1.0 | 1.293 | **109.42** | 17.77 | **0.16x** |

A 6x SLOWDOWN, and it has a specific, identified cause rather than being a
mystery. `src/decode/generate.py` allocates `draft_probs` as an
`(n+1) x vocab` **CPU** tensor in sampling mode — with a 256-node tree and
Llama-3's 128256-token vocabulary that is 257 x 128256 x 4 B = **132 MB moved
GPU->CPU per decode step**. Greedy mode never materialises drafter
distributions at all (it keeps per-slot logits on device and verification never
reads them), which is why only sampling pays this.

This follows from the CPU-RNG contract (`src/decode/verify.py`), which is a
deliberate reproducibility choice, not an oversight — docs/working-agreements.md explicitly
warns against moving to device-side RNG without reworking the contract. So it
is an implementation property, not a property of sampling-mode speculative
decoding, and it is fixable by anyone willing to redo that contract.

Acceptance also falls (1.639 -> 1.293): sampled continuations are simply harder
to predict. That part is inherent, unlike the latency.

**Consequence for the paper:** every speed claim must be scoped to greedy
decoding. Sampling-mode losslessness holds (distribution-matching is tested in
CI) but sampling-mode SPEED does not, and reporting the 0.16x openly is the
only defensible option — a reviewer running one temperature experiment would
find it immediately.

## Open cells

- sw-80k @ 10k held-out CE, and sw-80k @ 10k HumanEval — both queued behind
  M5 in `m3-rq2-matched.sh` (GPU-serialised so M5 timing rows stay clean).
- M5 factorial (2x2 drafter x tree, symmetric EAGLE cells, long/temp
  settings, vLLM reference) running.
