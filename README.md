# graphrag-certified

Experiment code for a study of certified defenses against structural poisoning of GraphRAG.

`full-version.pdf` is the full version of the paper: the submission's appendix carries only the
main proofs, and results it marks with an F (the HyperCube analysis, the security measurements
and the falsification ledger) are in this file.

Everything here is CPU-only: linear programming, max-flow, combinatorics and Monte Carlo over
integer sets, except `e2e_factiso.py`, which calls an LLM API.

## Run

```bash
pip install -r experiments/requirements.txt
make verify      # ~5s    check the committed results against the recorded claims
make all         # ~9min  regenerate all results, then check
make help        # list the individual experiment targets
```

`make verify` runs two gates. Gate 1 (`verify_claims.py`) checks the results and always runs.
Gate 2 (`audit_paper.py`) cross-checks a write-up against those results and skips unless you
point it at one:

```bash
PAPER=/path/to/main.tex make verify
```

`e2e_factiso.py` is the one experiment that calls an LLM API and is not part of `make all`; its
committed `results_e2e.json` comes from `qwen-122b` (vLLM, reasoning disabled), and a rerun with
the same model and the response cache in `experiments/e2e_cache/` reproduces it without new calls.
`pandapower` is optional; without it the witness experiments fall back to a degree-matched
synthetic host and record which host was used. `run_on_gpu.py` is the only script that wants a
GPU and is not part of `make all`.

`kg_omega.py` reads five public files under `experiments/data/` (MetaQA's WikiMovies KB and
its 2- and 3-hop test questions; Hetionet v1.0). They are ~25 MB and are not committed; run
`experiments/data/fetch_data.sh` once to download them, and `experiments/data/README.md`
records where each came from. `make kgomega` takes about two minutes, `make witness` about
ninety seconds.

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
| `experiments/e2e_factiso.py` | end to end: fact isolation with an LLM reader on MetaQA and Hetionet against four attacks, beside an undefended baseline; responses cached in `experiments/e2e_cache/` |
| `experiments/e2e_table.py` | the end-to-end attack table, generated from `results_e2e.json` and checked by gate 2 |
| `experiments/kg_claims.py` | corroboration of Hetionet treatment claims under seven metapaths (`experiments/data/`) |
| `experiments/kg_omega.py` | corroboration of real KGQA answers: MetaQA and Hetionet (`experiments/data/`) |
| `experiments/verify_claims.py` | gate 1 |
| `experiments/audit_paper.py` | gate 2 |
| `experiments/EXPERIMENTS.md` | each experiment name mapped to its script and results file |
| `experiments/results_*.json` | committed results the gates check against |

`run_all.py`, `stier.py`, `masc.py` and `theta_lowerbound.py` are earlier retrieval-side
experiments, superseded and kept for provenance.
