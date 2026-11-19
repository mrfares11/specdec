# FACT: Fixed-Budget Allocation and Cross-Step Training for Speculative Decoding

Code and measured results for the paper by Hadi Fares, Mayssa Hajj Hassan, Ali Salman and
Ammar Mohanna (American University of Beirut), [arXiv:2611.23463](https://arxiv.org/abs/2611.23463).

The paper studies how drafter training, candidate allocation and execution conditions together
decide whether speculative decoding is fast. A one-layer feature-prediction drafter proposes a
256-node candidate tree for a frozen Llama-3-8B-Instruct target, which verifies it under greedy
decoding on a single H100.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Drafter model, training loop, draft trees, verification and generation, benchmark harness |
| `scripts/` | Entry points for feature extraction, training, evaluation, timing campaigns and analysis |
| `scripts/pod-jobs/` | The command sequences used for the GPU campaigns, kept as they were run |
| `configs/` | OmegaConf configuration (`base.yaml` holds the schema and defaults, `tiny.yaml` small CPU test models) |
| `results/` | Per-prompt CSV files for every measured run; `results/README.md` describes each directory |
| `docs/tables/` | Summary tables written from `results/` by the analysis scripts, plus analysis notes |
| `tests/` | CPU tests that need no model downloads |

`RESULTS.md` maps the earlier milestone numbers to their files and commands.

## Reproducing the tables and figures

Every table and figure is computed from the committed files in `results/`, so no GPU is needed:

```bash
pip install -e .
python scripts/make_fact_tables.py              # paper tables, written to paper/tables-fact/
python scripts/make_arxiv_figures.py figures    # figures 4, 6, 8, 9 and 11
bash scripts/reproduce_protocol.sh              # correctness, interaction and host-thread analyses
```

Bootstrap intervals use 10,000 resamples over prompts with a fixed seed, so the numbers match
the paper exactly.

## Running the models

Training and timing need a CUDA GPU and access to the gated
`meta-llama/Meta-Llama-3-8B-Instruct` weights. `configs/base.yaml` points all data, checkpoints
and results at `/workspace`; override it on the command line, for example
`--set paths.root=./runs`. Each script documents its arguments at the top of the file, and
`scripts/pod-jobs/` shows the exact sequences behind the reported runs. Trained drafter
checkpoints are not included in this repository.

## Tests

```bash
pip install -e .[dev]
pytest
```

## Citation

```bibtex
@article{fares2026fact,
  title   = {{FACT}: Fixed-Budget Allocation and Cross-Step Training for Speculative Decoding},
  author  = {Fares, Hadi and Hajj Hassan, Mayssa and Salman, Ali and Mohanna, Ammar},
  journal = {arXiv preprint arXiv:2611.23463},
  year    = {2026}
}
```
