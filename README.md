# graphrag-certified

Experiment code for a study of certified defenses against structural poisoning of GraphRAG.

Everything here is CPU-only: linear programming, max-flow, combinatorics and Monte Carlo over
integer sets. The accompanying write-up is not part of this repository.

## Run

```bash
pip install -r experiments/requirements.txt
make verify      # ~5s    check the committed results against the recorded claims
make all         # ~4min  regenerate all results, then check
make help        # list the individual experiment targets
```

`make verify` runs two gates. Gate 1 (`verify_claims.py`) checks the results and always runs.
Gate 2 (`audit_paper.py`) cross-checks a write-up against those results and skips unless you
point it at one:

```bash
PAPER=/path/to/main.tex make verify
```

`pandapower` is optional; without it the witness experiments fall back to a degree-matched
synthetic host and record which host was used. `run_on_gpu.py` is the only script that wants a
GPU and is not part of `make all`.

## Layout

| path | what |
|---|---|
| `experiments/lp.py` | the two linear programs the rest builds on |
| `experiments/packing.py` | motif invariants, cell designs, skew, all connected patterns |
| `experiments/attack.py` | attack calculus and five implemented adversaries |
| `experiments/realdata.py` | a public query workload and a public network topology |
| `experiments/multitarget.py` | amplification, multi-target, full-knowledge, tightness |
| `experiments/witness_packing.py` | reachable damage, and three designs compared |
| `experiments/menger_check.py` | fractional packing, and the flow that computes it |
| `experiments/nu_check.py` | integral against fractional packing |
| `experiments/verify_claims.py` | gate 1 |
| `experiments/audit_paper.py` | gate 2 |
| `experiments/EXPERIMENTS.md` | each experiment name mapped to its script and results file |
| `experiments/results_*.json` | committed results the gates check against |

`run_all.py`, `stier.py`, `masc.py` and `theta_lowerbound.py` are earlier retrieval-side
experiments, superseded and kept for provenance.
