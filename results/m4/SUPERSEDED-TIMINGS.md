# WARNING: the timing columns in this directory are contaminated

**Do not cite any `speedup` or `*_ms_per_token` value from `results/m4/`.**

These runs (2026-08-20 21:01) predate the host thread cap
(`OMP/MKL_NUM_THREADS=8`, commit 8ca7f57, 2026-08-21 08:06 UTC). On the
224-core host, dynamic-tree topology construction is CPU-bound and thrashed
across cores while the GPU starved.

The same configurations, re-measured under the cap:

| MT-Bench, baseline drafter | here (uncapped) | `results/m5/` (capped) |
|---|---|---|
| dynamic, K=4 | 0.80x | **1.59x** |
| dynamic, K=8 | 0.72x | **1.37x** |

Attribution — the loss fell only on the CPU-heavy path:

| | uncapped | capped | change |
|---|---|---|---|
| dynamic spec ms/token | 23.01 | 11.61 | **-49.5%** |
| dynamic AR ms/token | 18.35 | 18.42 | +0.4% |
| static spec ms/token | 15.37 | 15.04 | -2.1% |

`git log` shows no changes to `src/decode/` or `src/eval/` between the two
measurements, so this is environmental, not an implementation change.

## What IS still valid here

The **acceptance** columns (`mean_accept_len`, `tokens_per_step`). Greedy
acceptance is deterministic and reproduced bit-identically under the cap
(2.2551 at K=4 both times).

Use `results/m5/` for anything timing-related. Full analysis:
`docs/tables/m3-findings.md` sections 10-11.
