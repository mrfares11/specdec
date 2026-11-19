# Protocol campaign — findings (E1, E3, E4)

Pod C, 2026-09-03/04. Llama-3-8B-Instruct verifier sourced from a
**hash-verified mirror** (`NousResearch/Meta-Llama-3-8B-Instruct`): all four
weight shards match `meta-llama` on LFS SHA-256 and byte size, checked before
use because our own account's gated access is still pending review.

All numbers below are on the **frozen test split** (`eval.prompt_split=test`),
disjoint from the dev pool used for any configuration choice. Every earlier
result used `legacy` (first-n) prompts with no dev/test separation, so figures
here are not directly comparable to the previous campaign — by design.

---

## E1 — numerical correctness (Gate A)

### float32 free-running equivalence: EXACT

| Precision | Positions compared | Agreement |
|---|---|---|
| **float32** | **52,224** | **1.000000** |
| bfloat16 | 51,200 | 0.7995 |

Zero divergent cells across 4 systems × 4 benchmarks, at **5× the protocol's
>10,000 requirement**. This is the load-bearing result: the decoding rule is
exactly equivalent to the target's greedy decoding in exact arithmetic, so any
bf16 difference is arithmetic rather than a verification defect. Had this
failed, no speed result would have mattered.

### The bf16 figure is compounding, not per-position error

Free-running agreement of 0.80 is easy to misread as "20% of tokens are wrong".
It is not. Procedure B scores the **identical prefix** through both paths, so
nothing compounds:

| Precision | Positions | Argmax agreement | Median margin (agree) | Median margin (disagree) |
|---|---|---|---|---|
| float32 | 12,800 | **1.0000** | 5.752 | — none |
| bfloat16 | 12,800 | **0.9900** | 5.875 | **0** |

The true per-position disagreement rate is **1%**. Over 128 tokens that
predicts roughly 1 − 0.99¹²⁸ ≈ 72% of sequences diverging somewhere, which is
what the free-running numbers show. Median first divergence is **28–88
positions**, so sequences agree for a long prefix and then follow different
trajectories.

### Agreement tracks task determinism

| Benchmark | bf16 agreement | Sequences exact | Median first divergence |
|---|---|---|---|
| HumanEval | 0.96–0.99 | 21–24 / 25 | 73–88 |
| GSM8K | 0.76–0.84 | 16–17 / 25 | 28–78 |
| MT-Bench | 0.72–0.77 | 13–15 / 25 | 32–46 |
| Alpaca | 0.63–0.79 | 9–15 / 25 | 34–56 |

Code generation has sharp, confident distributions and few near-ties;
open-ended instruction following has flat distributions and many. This is
indirect confirmation of the near-tie mechanism *before* the margins are
inspected directly.

### The margins must be read in ULPs, not absolute units

Ten disagreements initially appeared above the widest absolute margin bucket
(>1e-1) and looked unexplained. They are not:

- their margins are exactly **0.125 and 0.25** — powers of two;
- in **every** case the sequential winner is still **rank 0** under the batched
  logits, i.e. tied for first with the tie broken by index;
- the inter-path logit difference (0.14–0.20) *exceeds* the margin, which is
  what permits the swap.

bfloat16 has a 7-bit explicit mantissa, so at logit magnitudes of 16–32 the
representable spacing **is** 0.125. Those margins are one and two **units in
the last place** — the tightest non-zero gaps the format can express. Bucketing
by absolute margin mixes logit magnitudes and manufactures an unexplained tail
out of what are, at the precision in use, exact ties.

Procedure B now records `top1_logit`, `ulp_at_top1` and `margin_in_ulps`. The
expected restatement: *every* bf16 disagreement occurs where the two candidates
are indistinguishable at the precision being used.

### Downstream quality: no measurable harm — GATE A CLOSED

Same prompts through both paths, one deterministic harness, paired difference
(n=60 held-out prompts each):

| Benchmark | Spec | AR | Paired diff | 95% CI | Identical outputs |
|---|---|---|---|---|---|
| HumanEval pass@1 | 0.250 | 0.217 | **+0.033** | [−0.033, +0.100] | 29 / 60 |
| GSM8K exact match | 0.417 | 0.417 | **0.000** | [−0.067, +0.067] | 15 / 60 |

Per the protocol, a non-significant difference is **not** reported as "no
quality loss". The largest degradation still compatible with these intervals is
**3.3 points (HumanEval)** and **6.7 points (GSM8K)**. That bound is loose;
tightening it needs more prompts, not a different analysis.

**The absolute scores are not capability numbers.** 0.25 pass@1 and 0.42 exact
match are far below Llama-3-8B-Instruct's published figures. The harness caps at
384 new tokens, completes raw prompts with no stop-sequence trimming, and takes
GSM8K's answer as the trailing number. Those choices apply identically to both
arms and cancel in the difference — which is the only quantity being claimed.
Citing the absolute values as capability would be a misreading. MT-Bench is
excluded: it needs an LLM judge, which would make the result depend on a third
model's version.

Raw data: `results/e1q/`.

---

## E3 — interaction and transfer (Gate C) — CONFIRMED across two sessions

Paired by prompt; percentile bootstrap over prompts (4,000 resamples).
`G` = static − dynamic ms/token (positive: dynamic faster).

Pooled over sessions s1 and s2 (n=6 per cell):

| Benchmark | K | G(v1.0) | G(scaled) | I = interaction [95% CI] | T = G(EAGLE) [95% CI] |
|---|---|---|---|---|---|
| HumanEval | 4 | +4.082 | +3.474 | **−0.608 [−0.855, −0.364]** | +3.360 [+3.177, +3.552] |
| HumanEval | 8 | +4.178 | +3.688 | **−0.490 [−0.832, −0.116]** | +3.146 [+2.910, +3.377] |
| MT-Bench | 4 | +3.258 | +3.735 | **+0.478 [+0.259, +0.693]** | +3.635 [+3.451, +3.821] |
| MT-Bench | 8 | +3.460 | +3.366 | −0.094 [−0.389, +0.209] | +2.899 [+2.664, +3.132] |

The second session did not rescue the claim — it sharpened the refutation.
HumanEval K=8 moved from inconclusive on session 1 alone to **significantly
negative** with both sessions pooled.

### The "levers reinforce" claim does not survive

The draft states the interaction is "positive in every speedup cell". Tested
properly across two sessions it is **significantly negative on HumanEval at
both horizons**, significantly positive only on MT-Bench K=4, and
indistinguishable from zero on MT-Bench K=8. One cell of four supports the
claim; two significantly oppose it. The original claim came from cell means on
rounded speedups with no intervals, which is exactly what the protocol warned
against.

**A better hypothesis: headroom.** On HumanEval the scaled drafter already
accepts 2.27 statically, leaving little for the tree to add, so dynamic helps
it *less* than the weaker baseline. On MT-Bench, where acceptance is lower, the
better drafter has more room to exploit reallocation. Testable, and more
interesting than synergy.

### Transfer is confirmed and strengthened

`T` is positive with a 95% interval excluding zero in **4 of 4 cells** —
dynamic reallocation saves the official EAGLE drafter 2.90–3.64 ms/token
(pooled). The generality claim now rests on intervals rather than point
estimates, and it strengthens as the interaction claim falls.

**Both conclusions now rest on two independent sessions, as the protocol requires.** Per its own decision rule, a positive interaction may be claimed only where the interval excludes zero: that is one cell of four, and two cells significantly oppose it. The claim as written in the draft cannot stand.

---

## E4 — host thread sensitivity (Gate D)

Development prompts only; the cap is never selected on reported data.

### Latency (ms/token) vs thread cap

| Config | t=1 | t=4 | t=8 | t=32 | t=208 | worst/best |
|---|---|---|---|---|---|---|
| HumanEval dyn K=4 | 9.03 | 8.99 | **8.71** | 8.98 | **17.88** | **2.05×** |
| GSM8K dyn K=4 | 9.31 | 9.67 | 9.39 | 9.35 | **19.22** | **2.06×** |
| HumanEval static K=4 | 12.36 | 12.12 | 11.81 | 12.99 | 12.17 | 1.10× |
| GSM8K static K=4 | 13.08 | 12.75 | 12.44 | 13.06 | 13.41 | 1.08× |

Static decode acts as an internal control and is essentially flat (1.08–1.10×
across the whole sweep) while dynamic collapses by 2× uncapped. That selective
signature is what the C3 host-sensitivity claim needs, now shown as a
dose–response curve rather than inferred from two cross-session measurements.

### The conclusion demonstrably flips

| | t=8 | t=208 |
|---|---|---|
| GSM8K dyn K=4 | 1.89× | **0.91×** |
| GSM8K dyn K=8 | 1.65× | **0.92×** |
| HumanEval dyn K=4 | 1.97× | 1.02× |

Two configurations cross the 1.0× line on host thread configuration alone —
same code, same GPU, same weights.

### The control passes

Acceptance is **identical at every cap** (one distinct value per configuration,
e.g. 3.2334 across all five). Thread count changes only how fast the work runs,
never what is computed. Without this the sweep would be uninterpretable.

### Correction to our own practice

**t=1 is as good as t=8**, often tied or better. The curve is flat from 1 to 32
and falls off a cliff only at 208. The supportable claim is *not* "8 threads is
optimal" — which the protocol explicitly forbids — but **"the CPU-side path is
effectively serial; any cap ≤32 works, and leaving it uncapped on a many-core
host is catastrophic."** We had been describing 8 as if it were tuned. It was
not, and it did not need to be.

**Still owed for Gate D:** the attribution pass (where the uncapped time goes)
and second-host confirmation.

---

## Measurement-instrument bugs found this campaign

Four, all of which produced plausible-looking numbers while being wrong:

1. **`READY` self-match** — the RunPod PTY echoes commands, so a completion
   probe matched the word inside its own command text and every wait returned
   immediately.
2. **Length-sensitive equivalence check** — compared full token lists when
   speculative decoding legitimately overshoots `max_new_tokens`, reporting
   "NO" for every cell.
3. **`STATE_NO_SIGNAL` ambiguity** — a monitor whose silence meant either
   "healthy" or "pod unreachable".
4. **Range-as-drift** — `(max−min)/mean` grows with sample size by
   construction; it flagged 72 valid runs as drifted when the actual temporal
   trend was 0.03%.

Plus one design bug: a fixed `dev_size` consumed HumanEval's 164 rows entirely
and left an empty test pool.

This is not incidental to the paper. Its thesis is that measurement practice is
the substance rather than an afterthought, and the campaign keeps demonstrating
that on itself.

---

## E8 — where the sampling slowdown actually lives (Gate F evidence)

Profiled greedy against temperature 1.0, timers on, static tree, K=4:

| Phase | greedy → temp1, delta ms/step | share of the increase |
|---|---|---|
| **draft** | **+109.5** | 54% |
| **verify** | **+92.0** | 46% |
| kv/commit | +0.4 | ~0 |
| prefill | −0.1 | 0 |

Speedup falls from **1.258× to 0.183×**. The implied per-step device-to-host
transfer is **125.7 MiB** for a 256-node tree over a 128k vocabulary.

### The diagnosis was half right, and the missing half changes the fix

We have been attributing this to one thing: the drafter's per-node
distributions being materialised as an `(N+1) x vocab` CPU tensor to satisfy the
CPU-RNG contract. Drafting is indeed the largest single component (+109.5
ms/step), so that part holds.

But **verification balloons almost as much (+92.0 ms/step)**, which the
materialisation story alone does not predict. The reason is consistent with the
same contract: sampling-mode verification uses recursive rejection sampling,
which *consumes* those distributions on the CPU. The contract therefore charges
twice — once to materialise, once to sample.

**Consequence for E8's engineering decision.** A device-side RNG that only moves
distribution materialisation to the GPU addresses roughly half the excess and
would leave sampling far below 1.0×. Any repair must cover the verification
sampling path as well. Scoping the work as "materialise on device" would have
consumed the protocol's 3–5 day timebox and missed the target.

The paper's sampling paragraph has been widened to state both halves and the
sharper engineering consequence (`paper/main.tex`, Limitations).

