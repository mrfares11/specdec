# Pod-A results — superseded, read before citing

These are single runs from pod A (the v1.0 era). They are kept for provenance
and for the RQ1 grid, but three cautions apply.

**1. The RQ2 rows (`rq2-det10.csv`, `rq2-bp10.csv`, `rq2-sw10.csv`) are
superseded and their conclusion did not reproduce.** They supported the claim
that BP-uniform *improves* cross-entropy while *degrading* acceptance. On the
arm-independent held-out CE (`results/m3/heldout-ce/`) BP-uniform is worse than
detached on **both** axes, and detached dominates it at every one of the 8
rollout steps. The pod-A CE column was each arm's own step-weighted TRAINING
CE — exactly the confound `scripts/heldout_ce.py` was written to remove.
Use `results/m3/` instead.

**2. Several of these were measured under 3-way concurrent GPU use.** For
`humaneval.csv`, `mtbench.csv`, `gsm8k.csv`, `alpaca.csv`: acceptance columns
only — ignore ms/token and speedup. See `RESULTS.md` for the per-file map.

**3. Arm provenance is partially reconstructed.** The det/bp launch scripts
lived on the destroyed pod-A volume and the safetensors conversion stripped the
checkpoint-embedded configs. The M3 re-certification on pod B re-ran every
objective under fully version-controlled invocations.

The acceptance matrix in the paper now draws from `results/m1/` (pod B, n=3)
and falls back here only where M1 has no cell.
