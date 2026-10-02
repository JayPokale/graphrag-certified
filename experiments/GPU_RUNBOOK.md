# GPU runbook — the one experiment that needs an A100

Everything else in this repository is CPU-only and reproduces in under 30 seconds per script.
This is the single compute-bound experiment, and the paper currently ships the harness rather
than the result. §10 ("What is not covered") says so explicitly:

> We do not measure `kappa` or the per-cell error rate `eps'` for a real verdict model; that is
> the one compute-bound experiment in the programme, and the artifact ships the harness rather
> than the result.

Filling this gap is the highest-value experimental addition available before submission.
Reviewers will ask what `eps'` and `kappa` actually are for a real reader, because Theorem 8.3
prices a fallible reader in terms of both.

## What it measures

| Symbol | Meaning |
|---|---|
| `eps'` | marginal per-cell verdict error rate of a real LLM reader |
| `kappa` | correlation budget — how much worse than independent the joint cell errors are |

Why a real model is required: one prompt-injection payload is written once and replicated by
the partition into `rho` cells. Whether the readers that see it fail *together* is a fact about
the model, not about the mathematics. Simulation cannot answer it. The harness builds a genuine
anchored HyperCube partition, plants a real motif, injects a real payload into the cells the
partition actually routes it to, and records the full **joint** error vector per query — which
cells erred together — not just the marginal rate.

Measuring these does **not** rescue the certificate; it is already sound with nothing measured,
because Cor. A.4 bounds `kappa <= rho * m_s` from the threat model. Measuring **tightens** the
Hoeffding fee from `sqrt(rho*m_s*K*ln(1/delta)/2)` down to `sqrt(kappa*K*ln(1/delta)/2)`.

## Setup

```bash
pip install torch transformers accelerate numpy scipy
```

An A100 (40GB or 80GB) is ample — a 7–8B model in bf16 needs roughly 16GB. The docstring was
written against an H200; nothing about the experiment requires one.

## What was wrong with the first run (2026-09-09), and what changed

The 2026-09-09 API run (`qwen-122b`, 28,800 calls) reported `kappa = 1.0` in every
configuration.  That was a **clamp, not a measurement**: `build_query` planted ONE witness,
which lands in ONE cell (Prop. margin), so exactly one cell could ever err and there was
nothing for a correlation to show up in.  The harness now plants `--witnesses` (default 6)
witnesses that share the anchor's first relation, so several cells fire, and the payload
rides a **duplicate of that shared relation** (`--attack dup`), which A1 routes into every
firing cell: one payload, several readers, the event kappa prices.  `--attack fresh` keeps
the old behaviour (a new edge whose cells the partition draws on its own) for comparison.
The result file now records `firing_mean`, `payload_hits_firing_mean` and
`kappa_is_measurable`; do not quote kappa from a run where the last is false.

The same run had 462 truncated replies (1.60%).  Use `--probe 10` first, and either raise
`--api-max-tokens` or disable thinking with
`--api-extra '{"chat_template_kwargs":{"enable_thinking":false}}'` until the probe parses
cleanly.

## Verify plumbing first (no GPU, ~1 minute)

```bash
python3 run_on_gpu.py --dry-run
```

This has already been run and passes end to end. It uses a **synthetic** reader, writes
`results_llm.json` with `dry_run: true` and `model: null`, and its numbers are meaningless as
science. Delete that file before committing — it must never be mistaken for a measurement.

## The real run

Start small to confirm the model loads and the prompts behave:

```bash
python3 run_on_gpu.py --model Qwen/Qwen2.5-7B-Instruct --queries 20 --batch 32 --witnesses 6 --attack dup
```

Then the full sweep (2 cell counts x 2 loads x 3 payloads x 60 queries), roughly 1–3 GPU-hours
for a 7–8B model:

```bash
python3 run_on_gpu.py --model meta-llama/Llama-3.1-8B-Instruct
```

Useful knobs: `--queries` (shortens the run linearly), `--batch` (cells per forward pass),
`--cells`, `--loads`, `--cycle`, `--payloads`, `--witnesses`, `--attack`, `--dtype`, `--out`,
`--seed`.  Run both `--attack dup` and `--attack fresh`: the paper's escapes appendix says a
private hash blinds only the second.

Note `meta-llama/*` on HuggingFace is gated — accept the licence and `huggingface-cli login`
first, or use the Qwen model, which is not gated.

## Interpreting the result

The script prints `kappa_max`, and `eps'` and `kappa` per payload class. All three outcomes are
publishable, which is why this is worth running regardless of how it lands:

- **`kappa ~ 1` even under a targeted payload** — independence is tenable; Theorem 8.3 applies
  as written with the independent form.
- **`1 < kappa <= rho*m_s`** — the correlated form applies with a *measured* budget; quote
  `fee_ratio_measured_vs_free` as the tightening obtained.
- **`kappa` near `rho*m_s`** — the free bound from the threat model is already tight, so
  nothing was lost by not measuring. That is itself a finding, and it closes the gap §10
  currently leaves open.

## Reporting it in the paper

Whichever outcome occurs, §10 must be updated — the sentence quoted at the top of this file
claims the measurement was not made. If you run this, that sentence becomes false and has to
change. Add the numbers to the scorecard (Table 6) and to `audit_paper.py` so the new claim is
gated like every other number in the paper.

Do **not** report a `--dry-run` result as a measurement under any circumstances.
