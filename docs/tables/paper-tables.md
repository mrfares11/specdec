# Paper tables (auto-generated — do not hand-edit)

## Headline wall-clock (solo runs)

Source: results/m1 (pod B, n per cell = repeats found)

| System | Benchmark | K | ms/token (mean±std) | speedup (mean±std) | n |
|---|---|---|---|---|---|
| eagle | alpaca | 4 | 14.97 ± 0.07 | 1.23 ± 0.006 | 3 |
| eagle | alpaca | 8 | 16.14 ± 0.30 | 1.14 ± 0.005 | 3 |
| eagle | gsm8k | 4 | 14.83 ± 0.67 | 1.30 ± 0.009 | 3 |
| eagle | gsm8k | 8 | 15.69 ± 0.31 | 1.20 ± 0.009 | 3 |
| eagle | humaneval | 4 | 12.62 ± 0.36 | 1.45 ± 0.002 | 3 |
| eagle | humaneval | 8 | 13.16 ± 0.41 | 1.39 ± 0.001 | 3 |
| eagle | mbpp | 4 | 13.21 ± 0.17 | 1.40 ± 0.004 | 3 |
| eagle | mbpp | 8 | 14.05 ± 0.05 | 1.31 ± 0.002 | 3 |
| eagle | mtbench | 4 | 14.34 ± 0.21 | 1.27 ± 0.002 | 3 |
| eagle | mtbench | 8 | 15.34 ± 0.29 | 1.18 ± 0.004 | 3 |
| ours | alpaca | 4 | 15.19 ± 0.43 | 1.22 ± 0.005 | 3 |
| ours | alpaca | 8 | 17.07 ± 0.49 | 1.10 ± 0.006 | 3 |
| ours | gsm8k | 4 | 14.66 ± 0.46 | 1.27 ± 0.008 | 3 |
| ours | gsm8k | 8 | 16.22 ± 0.43 | 1.14 ± 0.006 | 3 |
| ours | humaneval | 4 | 13.71 ± 0.05 | 1.34 ± 0.001 | 3 |
| ours | humaneval | 8 | 14.78 ± 0.21 | 1.24 ± 0.001 | 3 |
| ours | mbpp | 4 | 14.36 ± 0.68 | 1.32 ± 0.004 | 3 |
| ours | mbpp | 8 | 15.54 ± 0.50 | 1.21 ± 0.004 | 3 |
| ours | mtbench | 4 | 15.37 ± 0.37 | 1.21 ± 0.003 | 3 |
| ours | mtbench | 8 | 17.13 ± 0.62 | 1.09 ± 0.010 | 3 |

In this table `ours` is the **v1.0 baseline drafter** (detached objective, 20k steps). M3 promoted the step-weighted 80k-row drafter to the headline configuration on acceptance; its wall-clock rows come from the M5 factorial and are NOT yet in this table. Do not mix the acceptance claims for the scaled drafter with the speedups here.

These M1 runs also PREDATE the host thread cap (commit 8ca7f57). Static decode is only mildly affected -- MT-Bench K=4 re-measures at 15.04 vs 15.37 ms/token, ~2% -- so these speedups are slightly CONSERVATIVE rather than wrong. The dynamic-tree rows measured in the same window are NOT usable; see the M4/M5 table. M5 re-measures MT-Bench and HumanEval under the capped host; MBPP, GSM8K and Alpaca timings remain pre-cap.


## Acceptance matrix (tokens committed per target forward)

| System | humaneval | mbpp | gsm8k | alpaca | mtbench |
|---|---|---|---|---|---|
| Plain AR (definition) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Ours (v1.0 drafter), K=4 | 2.91 | 2.90 | 2.78 | 2.66 | 2.62 |
| Ours (v1.0 drafter), K=8 | 3.00 | 2.98 | 2.79 | 2.68 | 2.65 |
| Official EAGLE (our harness), K=4 | 3.11 | 3.00 | 2.76 | 2.65 | 2.72 |
| Official EAGLE (our harness), K=8 | 3.30 | 3.10 | 2.82 | 2.70 | 2.78 |

Source: results/m1 (pod B, n=3 per cell), falling back to the pod-A CSVs only where M1 has no cell. Acceptance is deterministic under greedy decoding, so these are exact across repeats.

The SCALED drafter's acceptance is not in this table -- see the M5 factorial, which measures it against EAGLE in a single session.

## RQ1: acceptance across budget x allocation (mean accept length)

| Allocation | Nodes | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|---|
| uniform | 64 | 1.349 | 1.581 | 1.541 | 1.363 | 1.170 |
| uniform | 128 | 1.365 | 1.579 | 1.579 | 1.541 | 1.363 |
| uniform | 256 | 1.417 | 1.629 | 1.581 | 1.576 | 1.541 |
| geometric | 64 | 1.349 | 1.561 | 1.559 | 1.586 | 1.516 |
| geometric | 128 | 1.367 | 1.619 | 1.619 | 1.632 | 1.599 |
| geometric | 256 | 1.417 | 1.656 | 1.677 | 1.677 | 1.663 |

At K=32: reallocating a FIXED 256-node budget (uniform->geometric) buys +0.122 accept; quadrupling the geometric budget (64->256) buys +0.147. Timing columns in these CSVs predate the RQ3 loop rebuild - acceptance only.

## RQ2 training ablation, re-certified (M3, pod B)

Matched: 80k-row features (stream seed 0), 10k steps, batch 4 x accum 4,
geometric-256 tree, MT-Bench 80 prompts, greedy, seed 0.

| Objective (10k steps, matched) | held-out CE s1 | held-out CE uniform | accept K=4 | accept K=8 |
|---|---|---|---|---|
| BP-through, step weights 0.5^s | 2.43 | 3.33 | 1.69 | 1.74 |
| Teacher-forced control (rollout_K=1) | 2.29 | 7.62 | 1.49 | 1.50 |
| Detached rollout | 3.04 | 3.56 | 1.37 | 1.37 |
| BP-through, uniform weights | 3.30 | 3.77 | 1.34 | 1.35 |

Rank-correlation with acceptance across the 6 step-matched arms: step-1 CE +0.66, multi-step (uniform) CE +0.37. With n=6 these are directions, not significance tests.

Supersedes the pod-A three-arm table (results/bench/rq2-*.csv), whose CE
column was each arm's own step-weighted training CE and whose det/BP
ordering does not reproduce under the arm-independent metric. See
docs/tables/m3-findings.md.

## RQ3 incremental systems ablation (K=8, geometric-256, same drafter+prompts)

| Stage | ms/token | speedup | verify ms/step |
|---|---|---|---|
| Original scaffold loop | 322.13 | 0.046 | — |
| + device-side greedy verification | 20.83 | 0.675 | — |
| + KV gather (vs crop+replay) | 20.85 | 0.684 | — |
| + commit-fold (ONE forward/step) | 15.20 | 0.915 | 0.13 |

Commit-fold row measured with sync-bracketed timers ON (adds overhead); timer-off equivalent: results/fold/real100.csv (13.9 ms/tok, 1.06x at K=4).

## M4/M5: dynamic tree reallocation under a fixed budget

| Benchmark | K | accept static | accept dynamic | d accept | speedup static | speedup dynamic | (superseded) dynamic, uncapped host |
|---|---|---|---|---|---|---|---|
| mtbench | 4 | 1.64 | 2.26 | +0.62 | 1.20 | 1.59 | 0.79 |
| mtbench | 8 | 1.67 | 2.33 | +0.66 | 1.08 | 1.37 | 0.73 |
| humaneval | 4 | 1.95 | 2.88 | +0.93 | 1.34 | 1.86 | 0.95 |
| humaneval | 8 | 2.04 | 3.15 | +1.12 | 1.23 | 1.67 | 0.94 |
| mbpp | 4 | 1.92 | 2.83 | +0.91 | 1.33 | 1.85 | 0.96 |
| mbpp | 8 | 2.00 | 3.06 | +1.06 | 1.21 | 1.64 | 0.94 |

**The last column is a measurement artifact, retained only for the record.** Those dynamic runs predate the host thread cap (OMP/MKL_NUM_THREADS=8, commit 8ca7f57): on a 224-core host the CPU-bound work of per-step topology construction thrashed and starved the GPU. Capping threads halves dynamic decode latency (23.0 -> 11.6 ms/token on MT-Bench K=4) while leaving the AR baseline (18.3 -> 18.4) and static decode (15.4 -> 15.0) essentially unchanged -- the selective signature of host contention on the CPU-heavy path. Acceptance is bit-identical across both (2.2551 at K=4) and no decode or eval code changed between them.

Conclusion: dynamic reallocation DOES pay. The earlier negative result was environmental, not a property of dynamic topologies.

## M3: data scaling and training progress (step-weighted arm)

Data scaling, all held at 10k steps:

| Feature rows | MT-Bench K=4 | MT-Bench K=8 | HumanEval K=4 | held-out CE |
|---|---|---|---|---|
| 20k | 1.52 | 1.53 | 1.81 | 3.85 |
| 40k | 1.57 | 1.59 | 1.86 | 3.62 |
| 80k | 1.69 | 1.74 | 2.07 | 3.33 |

Training progress, 80k rows:

| Steps | accept K=4 | accept K=8 |
|---|---|---|
| 2k | 1.26 | 1.26 |
| 5k | 1.48 | 1.49 |
| 10k | 1.69 | 1.74 |
| 20k | 1.85 | 1.95 |

Acceptance is still climbing at 20k steps with no plateau, so every number reported for the headline drafter is a LOWER BOUND, not a converged value.

## M5: final factorial (drafter x tree), same session, solo GPU

### mtbench

| Drafter | Tree | K | accept | ms/token | AR ms/token | speedup | output-equivalent | n |
|---|---|---|---|---|---|---|---|---|
| v1.0 baseline (det-20k) | static geometric-256 | 4 | 1.64 +/- 0.000 | 15.03 +/- 0.022 | 18.02 +/- 0.075 | 1.20 +/- 0.007 | NO | 3 |
| v1.0 baseline (det-20k) | static geometric-256 | 8 | 1.67 +/- 0.000 | 16.54 +/- 0.059 | 17.96 +/- 0.111 | 1.09 +/- 0.004 | NO | 3 |
| v1.0 baseline (det-20k) | dynamic | 4 | 2.26 +/- 0.000 | 11.44 +/- 0.165 | 18.06 +/- 0.338 | 1.58 +/- 0.007 | NO | 3 |
| v1.0 baseline (det-20k) | dynamic | 8 | 2.33 +/- 0.000 | 13.16 +/- 0.092 | 17.95 +/- 0.226 | 1.36 +/- 0.008 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | static geometric-256 | 4 | 1.85 +/- 0.000 | 14.01 +/- 0.274 | 18.11 +/- 0.297 | 1.29 +/- 0.005 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | static geometric-256 | 8 | 1.95 +/- 0.000 | 15.29 +/- 0.285 | 18.27 +/- 0.278 | 1.19 +/- 0.005 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | dynamic | 4 | 2.57 +/- 0.000 | 10.52 +/- 0.081 | 18.40 +/- 0.168 | 1.75 +/- 0.014 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | dynamic | 8 | 2.78 +/- 0.000 | 11.69 +/- 0.132 | 18.23 +/- 0.343 | 1.56 +/- 0.012 | NO | 3 |
| official EAGLE | static geometric-256 | 4 | 1.75 +/- 0.000 | 14.31 +/- 0.017 | 18.10 +/- 0.048 | 1.26 +/- 0.003 | NO | 3 |
| official EAGLE | static geometric-256 | 8 | 1.81 +/- 0.000 | 15.37 +/- 0.079 | 17.98 +/- 0.056 | 1.17 +/- 0.005 | NO | 3 |
| official EAGLE | dynamic | 4 | 2.41 +/- 0.000 | 10.79 +/- 0.238 | 18.20 +/- 0.523 | 1.69 +/- 0.012 | NO | 3 |
| official EAGLE | dynamic | 8 | 2.53 +/- 0.000 | 12.16 +/- 0.181 | 18.10 +/- 0.388 | 1.49 +/- 0.012 | NO | 3 |

### humaneval

| Drafter | Tree | K | accept | ms/token | AR ms/token | speedup | output-equivalent | n |
|---|---|---|---|---|---|---|---|---|
| v1.0 baseline (det-20k) | static geometric-256 | 4 | 1.95 +/- 0.000 | 13.55 +/- 0.158 | 18.03 +/- 0.145 | 1.33 +/- 0.006 | NO | 3 |
| v1.0 baseline (det-20k) | static geometric-256 | 8 | 2.04 +/- 0.000 | 14.84 +/- 0.290 | 18.25 +/- 0.391 | 1.23 +/- 0.003 | NO | 3 |
| v1.0 baseline (det-20k) | dynamic | 4 | 2.88 +/- 0.000 | 9.61 +/- 0.138 | 18.02 +/- 0.323 | 1.87 +/- 0.009 | NO | 3 |
| v1.0 baseline (det-20k) | dynamic | 8 | 3.15 +/- 0.000 | 10.58 +/- 0.026 | 17.84 +/- 0.183 | 1.69 +/- 0.018 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | static geometric-256 | 4 | 2.30 +/- 0.000 | 12.33 +/- 0.191 | 18.25 +/- 0.244 | 1.48 +/- 0.004 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | static geometric-256 | 8 | 2.54 +/- 0.000 | 12.86 +/- 0.110 | 18.17 +/- 0.155 | 1.41 +/- 0.001 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | dynamic | 4 | 3.23 +/- 0.000 | 8.83 +/- 0.114 | 18.06 +/- 0.362 | 2.04 +/- 0.015 | NO | 3 |
| scaled (sw, 80k rows, 20k steps) | dynamic | 8 | 3.79 +/- 0.000 | 9.33 +/- 0.104 | 18.19 +/- 0.341 | 1.95 +/- 0.017 | NO | 3 |
| official EAGLE | static geometric-256 | 4 | 2.14 +/- 0.000 | 12.63 +/- 0.435 | 18.20 +/- 0.594 | 1.44 +/- 0.003 | NO | 3 |
| official EAGLE | static geometric-256 | 8 | 2.34 +/- 0.000 | 13.16 +/- 0.418 | 18.19 +/- 0.516 | 1.38 +/- 0.005 | NO | 3 |
| official EAGLE | dynamic | 4 | 2.95 +/- 0.000 | 9.36 +/- 0.352 | 17.95 +/- 0.183 | 1.92 +/- 0.052 | NO | 3 |
| official EAGLE | dynamic | 8 | 3.40 +/- 0.000 | 9.75 +/- 0.049 | 17.97 +/- 0.059 | 1.84 +/- 0.009 | NO | 3 |

### Factorial decomposition (drafter x tree)

| Benchmark | K | metric | base+static | scaled+dynamic | joint | drafter effect | tree effect | interaction |
|---|---|---|---|---|---|---|---|---|
| mtbench | 4 | accept | 1.64 | 2.57 | +0.93 | +0.26 | +0.67 | +0.11 |
| mtbench | 4 | speedup | 1.20 | 1.76 | +0.56 | +0.14 | +0.43 | +0.07 |
| mtbench | 8 | accept | 1.67 | 2.78 | +1.11 | +0.37 | +0.75 | +0.17 |
| mtbench | 8 | speedup | 1.08 | 1.56 | +0.48 | +0.15 | +0.32 | +0.07 |
| humaneval | 4 | accept | 1.95 | 3.23 | +1.28 | +0.35 | +0.93 | +0.00 |
| humaneval | 4 | speedup | 1.34 | 2.06 | +0.72 | +0.17 | +0.55 | +0.05 |
| humaneval | 8 | accept | 2.04 | 3.79 | +1.75 | +0.57 | +1.19 | +0.14 |
| humaneval | 8 | speedup | 1.23 | 1.97 | +0.73 | +0.24 | +0.49 | +0.12 |

A POSITIVE interaction means the two contributions reinforce rather than substitute: dynamic reallocation pays MORE with a better drafter, which is what the mechanism predicts -- global parent competition reallocates budget on the drafter's own signal, so a sharper drafter gives it more to work with. The effect is consistent on SPEEDUP (positive in every cell) and on acceptance in 3 of 4; HumanEval K=4 acceptance is exactly additive (+0.001), so the claim is 'reinforcing, never substituting' rather than 'always super-additive'.

### Protocol-breadth extras

| Setting | Drafter | K | accept | speedup |
|---|---|---|---|---|
| 512-token generation | v1.0 baseline | 4 | 1.70 | 1.28 |
| 512-token generation | scaled | 4 | 1.93 | 1.38 |
| temperature 1.0 (sampling) | v1.0 baseline | 4 | 1.29 | 0.16 |
| temperature 1.0 (sampling) | scaled | 4 | 1.49 | 0.19 |

vLLM reference row: not present (the chain treats it as failure-tolerant and never blocks on it).

All M5 cells present.

## M5 latency decomposition (K=8, sync-bracketed timers ON)

| Drafter | Tree | Benchmark | draft ms/step | tree-fwd ms/step | verify ms/step | commit ms/step |
|---|---|---|---|---|---|---|
| v1.0 baseline | static | mtbench | 18.06 | 20.86 | 0.15 | 2.48 |
| v1.0 baseline | dynamic | mtbench | 19.22 | 20.74 | 0.16 | 2.49 |
| scaled | static | mtbench | 18.52 | 21.35 | 0.15 | 2.55 |
| scaled | dynamic | mtbench | 18.93 | 20.57 | 0.15 | 2.51 |
| official EAGLE | static | mtbench | 17.21 | 21.03 | 0.14 | 2.45 |
| official EAGLE | dynamic | mtbench | 17.72 | 20.90 | 0.15 | 2.44 |
| v1.0 baseline | static | humaneval | 18.13 | 20.70 | 0.14 | 2.50 |
| v1.0 baseline | dynamic | humaneval | 19.04 | 20.83 | 0.15 | 2.58 |
| scaled | static | humaneval | 20.18 | 22.06 | 0.14 | 2.69 |
| scaled | dynamic | humaneval | 19.76 | 21.79 | 0.15 | 2.67 |
| official EAGLE | static | humaneval | 17.24 | 21.13 | 0.14 | 2.48 |
| official EAGLE | dynamic | humaneval | 17.69 | 20.93 | 0.15 | 2.49 |

Absolute values are inflated by the timers; read them as a COST DECOMPOSITION, not as wall-clock. The clean wall-clock numbers are in the factorial table above.
