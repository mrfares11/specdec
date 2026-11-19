# RESULTS.md — v1.0 frozen baseline & traceability map

Milestone 0 artifact (publication roadmap). Every number reported in
`docs/rq-summary-2026-08-11.pdf` traces to a CSV in this repo, a checkpoint,
and a command below. Tag: **v1.0**.

## Frozen artifacts

| Artifact | Location |
|---|---|
| Code | github.com/mrfares11/specdec |
| Drafter checkpoints | Not included in this repository. |
| Result CSVs | `results/bench/` (this repo, this tag) |
| Coauthor summary | `docs/rq-summary-2026-08-11.pdf` |
| Pod job scripts (as run) | `scripts/pod-jobs/` |

## Environment

CPU test environment (laptop, 136 tests passing at this tag):
Python 3.11.4, torch 2.3.1+cpu, transformers 4.57.6, omegaconf 2.3.1, Windows 11.

H100 runtime manifest: **captured** -> `results/env/h100-manifest.txt`
(H100 80GB HBM3, driver 580.126.09, torch 2.4.1+cu124 / cuDNN 9.1.0,
transformers 4.57.6, omegaconf 2.3.1, Python 3.11.10, Ubuntu 22.04.5,
224 vCPU / 2015 GB RAM, full pip freeze). The volume's
`/workspace/specdec-runs/logs/` also holds the tee'd invocation of every
benchmark run for cross-checking this map.

## Fixed evaluation settings (all rows)

Verifier `meta-llama/Meta-Llama-3-8B-Instruct` frozen; greedy; lossless
verification (exact greedy-equivalence in CI); draft tree **geometric
allocation, 256 nodes**; `eval.num_prompts=80`, `eval.max_new_tokens=128`,
`eval.K_sweep=[4,8]`; `runtime.seed=0`; single H100. Benchmark prompts are
pulled at run time by `src/eval/benchmark.py` from the pinned HF datasets
(HumanFaceH4/mt_bench_prompts, openai/openai_humaneval,
google-research-datasets/mbpp, openai/gsm8k, tatsu-lab/alpaca).

## Command template (ours)

```bash
python scripts/run_eval.py --config configs/base.yaml --set \
  eval.drafter_ckpt=<CKPT> eval.prompt_source=<BENCH> \
  eval.num_prompts=80 eval.max_new_tokens=128 'eval.K_sweep=[4,8]' \
  tree.allocation=geometric tree.num_nodes=256 eval.out_dir=<OUT>
```

EAGLE rows additionally merge `configs/eagle.yaml` (after
`scripts/import_eagle_weights.py --src yuhuili/EAGLE-LLaMA3-Instruct-8B`).

## Traceability table

| CSV (`results/bench/`) | System / checkpoint | Timing validity |
|---|---|---|
| `humaneval-solo.csv` | ours, `drafter/step_20000.pt` | **solo — citable wall-clock** |
| `mbpp.csv` | ours, `drafter/step_20000.pt` | **solo — citable wall-clock** |
| `mtbench-solo.csv` | ours, `drafter/step_20000.pt` | **solo — citable wall-clock** |
| `humaneval.csv` `mtbench.csv` `gsm8k.csv` `alpaca.csv` | ours, `drafter/step_20000.pt` | 3-way concurrent GPU — **acceptance columns only**; ignore ms/speedup |
| `rq2-det10.csv` | detached arm @ 10k steps, MT-Bench | concurrent — acceptance only |
| `rq2-bp10.csv` | BP-uniform arm @ 10k steps, MT-Bench | concurrent — acceptance only |
| `rq2-sw10.csv` | step-weighted (decay 0.5) arm, `drafter-sw/step_10000.pt`, MT-Bench | solo |
| `eagle-humaneval.csv` `eagle-mbpp.csv` `eagle-mtbench.csv` | official EAGLE weights imported, `eagle-import/eagle.pt` | **solo — citable wall-clock** |

RQ2 arm checkpoint paths for det10/bp10 to be re-confirmed against the pod's
`logs/bench-*.log` next session (recorded commands are ground truth).

Headline numbers these CSVs support: ours K=4 speedups 1.19×/1.19×/1.09×
(HumanEval/MBPP/MT-Bench); EAGLE-in-harness K=4 1.31×/1.28×/1.15×; RQ2
acceptance det 1.50/1.53, BP 1.35/1.35, step-weighted 1.58/1.62 (K=4/K=8).

## Known gaps at v1.0 (Milestone 1 backlog)

- ~~GSM8K and Alpaca **solo** timings not yet run~~ — CLOSED by M1:
  `results/m1/{ours,eagle}-{gsm8k,alpaca}-r{1,2,3}`, n=3 with mean ± std
  in the headline table. (This bullet was stale; M1's exit criteria are met.)
- ~~Headline timings are single runs~~ — CLOSED by M1: n=3 repeats,
  mean ± std, in `results/m1/` and the headline table.
- ~~H100 environment manifest not yet captured~~ — CLOSED:
  `results/env/h100-manifest.txt`.
- ~~RQ2 arms: confirm shared init seed for the "matched" claim~~ — CLOSED by
  the M3 re-certification below.

## RQ2 arm provenance (audit, M2)

Matched across all three arms and verifiable today: training data (same
extraction, stream seed 0), step count (10k), batch 4 x accum 4, eval tree
(geometric-256, K=[4,8]) and prompts (MT-Bench 80). Fully recorded: the
step-weighted arm's exact invocation (`scripts/pod-jobs/mbpp-then-train.sh`)
and its checkpoint-embedded train_cfg. PARTIALLY recorded: det/bp arms —
their launch scripts were on the destroyed pod-A volume and the safetensors
conversion stripped checkpoint-embedded configs; HF `train_log*.csv`
corroborate schedule/steps but init-seed identity is reconstructed, not
certifiable. Remedy: the M3 retraining re-runs all objectives on pod B under
fully version-controlled invocations; pod-A arms are reported as consistent
preliminary evidence.

## M3 re-certification (COMPLETE) — supersedes the pod-A RQ2 arms

All six arms retrained on pod B from `scripts/pod-jobs/m3-train.sh` (in git,
every invocation version-controlled), matched on 80k-row features (stream seed
0), batch 4 x accum 4, and 10k steps (the step-weighted flagship additionally
ran to 20k). Scored two ways: MT-Bench/HumanEval acceptance, and an
**arm-independent held-out rollout CE** (`scripts/heldout_ce.py`, 400 dev
samples from feature rows >= 100k, identical batching and shard order for every
checkpoint) that removes the per-arm step-weighting confound in the pod-A CE
column.

**The pod-A RQ2 conclusion did not reproduce.** BP-uniform is worse than
detached on both acceptance and held-out CE, not better on CE — see
`docs/tables/m3-findings.md` for the full analysis and the replacement claim.
Treat the pod-A RQ2 rows (`results/bench/rq2-*.csv`) as superseded.

| Artifact | Location |
|---|---|
| Arm checkpoints | `/workspace/specdec-runs/checkpoints/m3-*/` + HF repo (auto-synced) |
| Acceptance CSVs | `results/m3/m3-*-{mtbench,humaneval}/benchmark.csv` |
| Held-out CE | `results/m3/heldout-ce/*.json` |
| Campaign logs | `results/m3-logs/` |
| Analysis | `docs/tables/m3-findings.md` |

Determinism check: M5's `base-static-mtbench` cell reproduces M1's
`ours-mtbench` acceptance exactly to four decimals across sessions, so
acceptance numbers are comparable across milestones.
