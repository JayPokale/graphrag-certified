# Experiment index

The paper names each experiment for the quantity it measures. This table maps those names to
the script that produces the number and the results file the reproduction gate asserts against.
Run `make verify` to check the paper against all of them, or `make all` to regenerate first.

| Name in the paper | What it establishes | Script | Results file |
|---|---|---|---|
| `share-lp` | share LP value `= 1/tau*_A`, 15 motif families, duality gap `< 1e-15` | `packing.py` | `results_packing.json` (`E12_invariants`) |
| `hypercube-exponent` | anchored HyperCube attains `K^(1-1/tau*_A)` with `P_det = 1` | `packing.py` | `results_packing.json` (`E13_*`) |
| `all-patterns` | `tau*_A < d` on 53/58 shape–anchor pairs up to 5 edges | `packing.py` | `results_packing.json` (`E27_all_patterns`) |
| `skew-and-pinning` | `rho` invariant to the data; anchoring never lowers `tau*_A` (400/400) | `packing.py` | `results_packing.json` (`E24_skew_separation`) |
| `tail-bound` | certified success-probability bound never violated (0/50) | `attack.py` | `results_attack.json` (`X1_*`) |
| `surface-identity` | `S = K B̄` with gap exactly 0 on PEGASE | `attack.py` | `results_attack.json` (`X4_*`) |
| `channel-choice` | attacker-mix threshold predicts the cheaper channel, 20/20 | `attack.py` | `results_attack.json` (`X5_*`) |
| `blinding-gain` | blinding buys `<= 2x` with margin, up to `63.6x` at onset | `attack.py` | `results_attack.json` (`X2_*`, `X2b_*`) |
| `adaptivity` | adaptive attacker gains `1.011x` over blind | `attack.py` | `results_attack.json` (`X2_*`) |
| `slice-alignment` | slice variance is an alignment artifact, `1.62x` generic | `attack.py` | `results_attack.json` (`X3_*`) |
| `bernstein-gain` | `min(Bernstein, Hoeffding)` beats either, 18/18 | `attack.py` | `results_attack.json` (`X6_*`) |
| `correlation-chaining` | `kappa <= rho m_s` holds 36/36; chaining reaches `7.6 rho` | `attack.py` | `results_attack.json` (`X8_*`) |
| `end-to-end` | certificate holds 89/89 under correlated verdict failure | `attack.py` | `results_attack.json` (`X9_*`) |
| `occupancy` | observed amplification meets `rho_max` exactly, ratio 1.000 | `multitarget.py` | `results_multitarget.json` (`E28_*`) |
| `multi-target` | per-target amplification constant; `Gamma` slope 1.00 | `multitarget.py` | `results_multitarget.json` (`E29_*`) |
| `full-knowledge` | full-knowledge attacker reaches `b rho_max`, never exceeds | `multitarget.py` | `results_multitarget.json` (`E30_*`) |
| `tightness` | `b_break / b_cert` in `[1.02, 1.33]` | `multitarget.py` | `results_multitarget.json` (`E31_*`) |
| `tpc-h` | `tau* < d` on 18/20 real queries, median gain `3.2x` | `realdata.py` | `results_realdata.json` (`RD1_tpch`) |
| `pegase` | `P_det = 1` on a real network; exact slot constant | `realdata.py` | `results_realdata.json` (`RD2_*`) |
| `reachable-damage` | exact `rho_F` (two endpoint hash vectors) equals `rho_max` at 27 of 32 anchor–cell points; the earlier single-vector count was 1.7–2.8x too small | `witness_packing.py` | `results_witness.json` |
| `duplicate-attack` | a blind copy of the most-shared honest relation reaches `rho_F/2` firing cells with no hash knowledge; 16x a fresh content's expectation at `K=256` | `witness_packing.py` | `results_witness.json` (`dup_*`) |
| `three-designs` | majority vs instance vs witness certificate on one instance | `witness_packing.py` | `results_witness.json` |
| `fractional-gain` | exact `omega <= nu* <= tau`; C4: `nu*/omega` max 1.50 (strict 7/12), `tau/nu*` max 1.85, transversal rule certifies median 5 vs witness packing 2 | `menger_check.py` | `results_menger.json` (`E34_*`) |
| `flow-check` | `omega_a = |N(v) ∩ N(a)|` on 464/464 two-hop answers | `menger_check.py` | `results_menger.json` (`E35_*`) |
| `metaqa` | median `omega_a = 1` on 92,323 two-hop and 191,110 three-hop gold answers; 96%/92% single-chain | `kg_omega.py` | `results_kgomega.json` (`metaqa`) |
| `hetionet` | median `omega_a = 1` on four metapaths of Hetionet (GragPoison's Medical domain) | `kg_omega.py` | `results_kgomega.json` (`hetionet`) |
| `claim-check` | 755 Hetionet CtD claims and 755 non-claims under a 7-metapath support class: tau >= 3 on 72.3% against 11.7%; omega = nu* = tau on every pair | `kg_claims.py` | `results_kgclaims.json` |
| `e2e` | fact isolation with qwen-122b on 200 MetaQA questions and 150 Hetionet claims, 4 attacks x budgets {1,3}, undefended baseline: 0 report changes within the certified budget over 864 attacked runs; injection 0.0% vs 8.1% | `e2e_factiso.py` | `results_e2e.json` |
| (supporting) | `nu = tau*_A` on 35/35 families and anchored TPC-H queries | `nu_check.py` | `results_nu.json` |

JSON keys still carry the historical `E`-numbers; the paper does not. Superseded retrieval-side
experiments (`run_all.py`, `stier.py`, `theta_lowerbound.py`) are kept for provenance and are
not cited by the current paper.
