# results/ — index

Every number in the paper traces to a file here. **Nothing in this directory is
hand-edited**; `python scripts/make_paper_tables.py` regenerates all 9 paper
tables from these files alone (no GPU, no pod — verified byte-exact).

## Read these first, in order

1. **`docs/tables/paper-tables.md`** — every paper table, auto-generated.
2. **`docs/tables/m3-findings.md`** — the analysis writeup: what we found, what
   got overturned, and why. This is the most important document here.
3. **`RESULTS.md`** (repo root) — traceability map: number → CSV → checkpoint →
   command.

## Which directories are authoritative

| Directory | Contents | Status |
|---|---|---|
| `m5/` | Final factorial: 3 drafters x 2 trees x 2 benchmarks x K in {4,8}, plus protocol extras and the MBPP re-measure | **AUTHORITATIVE** — headline results |
| `m5-rep/` | Repeats r2, r3 of every factorial cell (with `m5/` these give n=3) | **AUTHORITATIVE** — variance |
| `m5-timing/` | Same cells with `profile_timing=true` — draft / tree-forward / verify / commit breakdown | **AUTHORITATIVE** — cost decomposition only, NOT wall-clock |
| `m3/` | The 6 matched training arms, data-scaling and training-progress curves, RQ3 loop ladder, `heldout-ce/` | **AUTHORITATIVE** |
| `m1/` | Benchmark matrix, 5 benchmarks x {ours, EAGLE} x K, n=3 | **AUTHORITATIVE** (see caveat below) |
| `m4/` | Static-vs-dynamic tree experiment | **ACCEPTANCE ONLY — timings superseded** |
| `bench/` | Pod-A single runs (v1.0 era) | **SUPERSEDED** — see below |
| `rq1/ rq2/ rq3/ fold/` | Pod-A historical grids and ladders | historical; RQ1 grid still cited, RQ2 rows superseded by `m3/` |
| `e1/` | E1 correctness: free-running equivalence at fp32/bf16, same-prefix margins | **AUTHORITATIVE** |
| `e2/` | E2 timing replication, sessions s1/s2/s3, per-prompt rows with paired AR | **AUTHORITATIVE** |
| `e4/` | E4 host thread sweep (`sweep-*`) and attribution (`attrib-*`) | **AUTHORITATIVE** / diagnostic |
| `e8/` | E8 sampling profile, timers on | **DIAGNOSTIC** — cost attribution only |
| `env/` | H100 runtime manifest (driver, torch, full pip freeze) | reference |
| `pod-logs/` | All 95 pod logs — every run's exact tee'd output | provenance |
| `m1-logs/ m3-logs/` | Training logs and master chain logs | provenance |

## Protocol results vs the earlier campaign

`e1/ e2/ e4/ e8/` come from the protocol campaign and use the **frozen
dev/test prompt split**. Everything else used `legacy` (first-n) prompts with
no dev/test separation — and the scaled drafter, the recommended configuration
and the thread cap were all chosen on those same prompts. The two sets are
therefore **not directly comparable, by design**. The protocol numbers are the
defensible ones.

They also use a hash-verified mirror of the verifier
(`NousResearch/Meta-Llama-3-8B-Instruct`, all four weight shards matching
`meta-llama` on LFS SHA-256 and byte size), because our gated access was
pending when the runs happened.

## Three traps — please read before using any number

**1. `m4/` timings are contaminated. Do not cite them.**
Those runs predate the host thread cap. On a 224-core host with an uncapped
thread pool, dynamic-tree decoding measured **0.80x**; capped at 8 threads the
identical configuration measures **1.76x**. Same GPU, same code, bit-identical
acceptance. The `m4/` *acceptance* numbers are fine (deterministic); its
*speedups* are superseded by `m5/`. Details: `m3-findings.md` sections 10-11.

**2. Speed claims hold for GREEDY decoding only.**
Sampling mode at temperature 1.0 measures **0.16x** — a 6x slowdown — because
the drafter's per-node distributions are materialised as an (N+1) x vocab CPU
tensor to satisfy the CPU-RNG reproducibility contract (~132 MB moved per
decode step). Lossless, but not fast. See `m5/temp1-*`.

**3. "Byte-identical" is true in exact arithmetic, not in bf16.**
Greedy speculative decoding matches AR greedy exactly at float32 (640/640
positions). Under bf16 the tree-batched and sequential target forwards round
differently and occasionally flip an argmax at a near-tie (88.4% of positions
matched over 5 prompts x 128 tokens). Acceptance and speed numbers are
unaffected; only a byte-identity claim is. Re-measure with
`scripts/diag_equivalence.py`.

Minor: `m1/` runs also predate the thread cap, but static decode moved only
~2% (15.37 -> 15.04 ms/token), so those speedups are slightly CONSERVATIVE
rather than wrong.

## The headline numbers

Recommended operating point: **scaled drafter + dynamic tree at K=4.**

| Benchmark | v1.0 + static | scaled + dynamic | n |
|---|---|---|---|
| MT-Bench | 1.199x +/- 0.007 | **1.749x +/- 0.014** | 3 |
| HumanEval | 1.331x +/- 0.006 | **2.045x +/- 0.015** | 3 |

The two contributions reinforce rather than substitute (positive drafter x tree
interaction in every speedup cell), and the dynamic tree transfers to the
external EAGLE drafter (+33.0% MT-Bench, +34.8% HumanEval) — so it is a general
property of tree-based speculative decoding, not something tuned to our drafter.

## Regenerating

```bash
python scripts/make_paper_tables.py     # all 9 tables from these files
python scripts/verify_campaign.py       # checks all 55 required artifacts exist
```
