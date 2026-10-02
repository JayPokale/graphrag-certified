"""
verify_claims.py — self-audit: assert every load-bearing claim in findings/*.md against
the canonical experiments/results.json. Exit non-zero if any claim fails.

    python3 verify_claims.py

This is the reproducibility gate: if the numbers in the markdown drift from the data file,
this fails loudly. Run it after any re-run of run_all.py.
"""
from __future__ import annotations
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
ok = True


def check(name, cond, detail=""):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))


print(f"results.json: quick={R['config']['quick']}, runtime={R['runtime_sec']:.0f}s\n")

# --- C2: sigma_M grows with n on scale-free (polylog premise FALSIFIED) ---
sf = R["E2_spread"]["scale_free"]
ns = sorted(int(n) for n in sf)
s_first, s_last = sf[str(ns[0])]["sigma_mean"], sf[str(ns[-1])]["sigma_mean"]
check("C2 sigma_M grows >=4x over n-range (polylog falsified)",
      s_last > 4 * s_first, f"{s_first:.0f}->{s_last:.0f} for n={ns[0]}..{ns[-1]}")

# --- C3: single-source PPR is the WORST sampler at large k (original Step 2 false) ---
rad = R["E3"]["radial"]["samples"]
kmax = max(int(k) for k in rad["uniform"])
ppr_k, uni_k = rad["ppr"][str(kmax)]["mean"], rad["uniform"][str(kmax)]["mean"]
check("C3 single-source PPR worse than uniform at k_max",
      ppr_k > uni_k, f"ppr {ppr_k:.0f} > uniform {uni_k:.0f} at k={kmax}")

# --- C3': bippr halves the exponent — >=100x fewer samples than uniform at k_max ---
bip_k = rad["bippr"][str(kmax)]["mean"]
check("C3' bippr >=100x fewer samples than uniform at k_max",
      uni_k / bip_k >= 100, f"{uni_k/bip_k:.0f}x (uniform {uni_k:.0f} / bippr {bip_k:.0f})")
# bippr near-flat: bippr(k_max)/bippr(k_min) much smaller than uniform's blowup
kmin = min(int(k) for k in rad["uniform"])
bip_growth = rad["bippr"][str(kmax)]["mean"] / rad["bippr"][str(kmin)]["mean"]
uni_growth = rad["uniform"][str(kmax)]["mean"] / rad["uniform"][str(kmin)]["mean"]
check("C3' bippr grows far slower in k than uniform",
      bip_growth < uni_growth / 20, f"bippr x{bip_growth:.0f} vs uniform x{uni_growth:.0f}")

# --- C7: detection onset tracks rho = K^{1-1/d} ---
d = R["config"]["d"]
for K, ser in R["E4a_detection"].items():
    onset = float(K) ** (1 - 1.0 / d)
    rhos = sorted(int(x) for x in ser)
    below = [r for r in rhos if r <= onset]
    above = [r for r in rhos if r >= 2 * onset]
    lo = max((ser[str(r)]["detect"] for r in below), default=0.0)
    hi = min((ser[str(r)]["detect"] for r in above), default=1.0)
    check(f"C7 K={K}: detection low below onset({onset:.1f}), high well above",
          lo < 0.7 and hi > 0.7, f"det@<=onset={lo:.2f}, det@>=2onset={hi:.2f}")

# --- C6: certificate tight — optimal adversary is a step exactly at budget = mu ---
mu = R["config"]["fmu"]
opt = {r["budget"]: r["flip"] for r in R["E4b_flip"]["optimal"] if r["m_s"] == 0}
below_mu = [opt[b] for b in opt if b < mu]
at_above = [opt[b] for b in opt if b >= mu]
check("C6 optimal adversary cannot flip below budget=mu",
      all(f == 0.0 for f in below_mu), f"max flip below mu={mu}: {max(below_mu, default=0):.2f}")
check("C6 optimal adversary always flips at/above budget=mu (tight)",
      all(f == 1.0 for f in at_above), f"min flip at>=mu={mu}: {min(at_above, default=1):.2f}")
# soundness: random adversary needs MORE than mu for reliable flip
rnd = {r["budget"]: r["flip"] for r in R["E4b_flip"]["random"] if r["m_s"] == 0}
check("C6 random adversary is weaker (conservative cert) at budget=mu",
      rnd.get(mu, 1.0) < 0.9, f"random flip at budget=mu: {rnd.get(mu, float('nan')):.2f}")

# =========================================================================== #
# T2/E5 — theta gadget: bippr cannot beat w^{(k-1)/2} polynomially            #
# =========================================================================== #
theta_path = os.path.join(HERE, "results_theta.json")
if os.path.exists(theta_path):
    T = json.load(open(theta_path))
    print()
    for w, ser in T.items():
        kmax = str(max(int(k) for k in ser))
        r = ser[kmax].get("bippr_over_ref")
        check(f"E5 w={w}: bippr/ref stays >=1 at k={kmax} (no poly beat of w^(k-1)/2)",
              r is not None and r >= 1.0, f"bippr/ref={r:.2f}" if r else "capped")

# =========================================================================== #
# Breakthrough legs — results_stier.json (E6 covering, E7 private, E8 anchor) #
# =========================================================================== #
stier_path = os.path.join(HERE, "results_stier.json")
if os.path.exists(stier_path):
    S = json.load(open(stier_path))
    print()
    # --- E6 / Leg 3: covering-design optimality ---
    rows = S["e6"]["per_q"]
    check("E6 PG(2,q) verified exact Steiner 2-design for every q",
          all(r["steiner"] and r["all_pairs_covered"] and r["P_det"] == 1.0 for r in rows),
          f"q={[r['q'] for r in rows]}")
    ratios = [r["rho_over_sqrtK"] for r in rows]
    check("E6 rho/sqrt(K) decreases toward 1 (design meets covering bound)",
          all(a > b for a, b in zip(ratios, ratios[1:])) and ratios[-1] < 1.05,
          f"{ratios[0]:.3f} -> {ratios[-1]:.3f}")
    seps = [r["ratio_allpairs"] for r in rows]
    check("E6 random/design all-pairs ratio GROWS (sqrt(log N) separation)",
          seps[-1] > seps[0] and seps[-1] >= 2.5,
          f"{seps[0]:.2f} -> {seps[-1]:.2f}")
    check("E6 amplification identity A=rho under both designs (coupling unbroken)",
          all(r["amplification_pg"] == r["rho"] == r["q"] + 1 for r in rows))
    last = rows[-1]
    check("E6 pencil concurrency << rho vs general-position firing set",
          last["pencil_mean_concurrency"] < last["rho"] / 2,
          f"conc {last['pencil_mean_concurrency']:.1f} vs rho {last['rho']}")
    # --- E7 / Leg 4: private-hash decoupling ---
    maj = S["e7"]["majority"]
    check("E7 majority regime: blind gain is a small constant (<=3x)",
          1.0 <= maj["gain"] <= 3.0, f"gain x{maj['gain']:.2f}")
    for d in (2, 3):
        rs = sorted((r for r in S["e7"]["or_regime"] if r["d"] == d), key=lambda r: r["K"])
        grow = all(a["sep"] <= b["sep"] for a, b in zip(rs, rs[1:]))
        big = rs[-1]
        near = 0.5 <= big["sep"] / big["pred"] <= 1.5
        check(f"E7 OR d={d}: separation grows with K and tracks K^(1/d) at K={big['K']}",
              grow and near, f"sep x{big['sep']:.1f} vs pred {big['pred']:.1f}")
    # --- E8 / Leg 2: anchor-count law ---
    for w, ser in S["e8"].items():
        rats = [ser[a]["bippr_over_ref_thm"] for a in ser if ser[a]["bippr_over_ref_thm"]]
        check(f"E8 w={w}: bippr/(a*(w/(1-alpha))^depth) bounded in [1,20] for all a",
              all(1.0 <= r <= 20.0 for r in rats),
              f"ratios {['%.1f' % r for r in rats]}")
        drop = ser["1"]["bippr"] / ser["2"]["bippr"]
        check(f"E8 w={w}: a=1 -> a=2 collapses samples >=10x (exponent halves)",
              drop >= 10, f"x{drop:.0f}")
    # --- E9: anchor pinning under structure-aware designs ---
    if "e9" in S:
        print()
        rows9 = S["e9"]
        check("E9 every design achieves exhaustive single-cycle coverage",
              all(r["coverage_ok"] for r in rows9),
              f"{len(rows9)} designs")
        check("E9 pinning theorem holds: rho_min(first-hop)*B/w^(k-2) >= 1 everywhere",
              all(r["ratio"] >= 1.0 for r in rows9),
              f"min ratio {min(r['ratio'] for r in rows9):.2f}")
        pencil = [r for r in rows9 if r["design"] == "pencil"]
        check("E9 pencil design meets the pinning bound within 3x (tight)",
              all(r["ratio"] <= 3.0 for r in pencil),
              f"ratios {['%.2f' % r['ratio'] for r in pencil]}")
        check("E9 average replication stays tiny (<5) while first-hop is pinned",
              all(r["avg_rho"] < 5.0 for r in rows9),
              f"max avg_rho {max(r['avg_rho'] for r in rows9):.2f}")
    # --- E10: adaptive adversary vs private hash ---
    if "e10" in S:
        print()
        pers = S["e10"]["persistent"]
        check("E10a adaptivity buys nothing in persistent model (gain in [0.8, 1.25])",
              all(0.8 <= r["gain"] <= 1.25 for r in pers),
              f"gains {['%.2f' % r['gain'] for r in pers]}")
        pr = S["e10"]["probes"]
        fx = pr["fixed"]
        err = max(abs(r["p_hitter"] - r["p_hitter_pred"]) for r in fx)
        check("E10b probe leak tracks 1-(1-rho/K)^T within 0.08 at every T",
              err <= 0.08, f"max |measured-pred| = {err:.3f}")
        t0 = next(r for r in fx if r["T"] == 0)["median_budget"]
        thl = min((r for r in fx if r["median_budget"] <= t0 / 2),
                  key=lambda r: r["T"], default=None)
        check("E10b fixed-hash half-life within 2x of ln2*K/rho",
              thl is not None and pr["halflife_pred"] / 2 <= thl["T"] <= 2 * pr["halflife_pred"],
              f"measured T_1/2 ~ {thl['T'] if thl else '?'} vs pred {pr['halflife_pred']:.1f}")
        rh = S["e10"]["probes"]["rehash"]
        first, last = rh[0]["median_budget"], rh[-1]["median_budget"]
        check("E10c re-hash per query is FLAT (last-T budget >= 0.7x first)",
              last >= 0.7 * first, f"T=0: {first:.0f} -> T_max: {last:.0f}")
    # --- E11: noisy per-cell verdicts ---
    if "e11" in S:
        print()
        cur = S["e11"]["majority_noise"]["curves"]
        tols = [c["tolerated"] for c in cur]
        check("E11 tolerated budget decreases monotonically with noise",
              all(a >= b for a, b in zip(tols, tols[1:])) and tols[0] > tols[-1],
              f"{tols}")
        check("E11 Hoeffding certificate NEVER violated (measured >= floor)",
              all(c["tolerated"] >= c["predicted"] for c in cur),
              f"floors {['%.1f' % c['predicted'] for c in cur]}")
        scal = S["e11"]["scaling"]
        fracs = [r["frac_of_margin"] for r in scal]
        check("E11 tolerated/margin climbs toward 1 (noise is second-order)",
              all(a < b for a, b in zip(fracs, fracs[1:])) and fracs[-1] >= 0.9,
              f"{['%.2f' % f for f in fracs]}")
        big = scal[-1]
        check("E11 CLT prediction within 3 poisons of measurement at largest K",
              abs(big["tolerated"] - big["clt_pred"]) <= 3.0,
              f"measured {big['tolerated']} vs CLT {big['clt_pred']:.0f} at K={big['K']}")

# ---------------------------------------------------------------------------
# E12-E17: the packing law (Section "The Packing Law", results_packing.json)
# ---------------------------------------------------------------------------
_pk = os.path.join(HERE, "results_packing.json")
if not os.path.exists(_pk):
    print("\n[skip] results_packing.json absent — run packing.py")
else:
    P = json.load(open(_pk))
    print("\n--- E12-E17: packing law ---")

    inv = P["E12_invariants"]
    check("E12 share-LP value equals 1/tau*_v on every motif (zero duality gap)",
          all(r["lp_duality_gap"] < 1e-9 for r in inv),
          f"max gap {max(r['lp_duality_gap'] for r in inv):.2e}")
    check("E12 tau*_v <= d always",
          all(r["tau_star_v"] <= r["d"] + 1e-9 for r in inv))
    check("E12 tau*_v == d IFF free vertices form a matching",
          all(r["tau_equals_d"] == r["matching_on_free"] for r in inv),
          f"{sum(r['tau_equals_d'] for r in inv)}/{len(inv)} at equality")
    byname = {r["motif"]: r for r in inv}
    check("E12 separation (i): star cheap to find, expensive to protect (r_v=1, tau*=d)",
          all(byname[f"star d={k}"]["r_v"] == 1 and
              abs(byname[f"star d={k}"]["tau_star_v"] - k) < 1e-9 for k in (3, 5, 8)))
    check("E12 separation (ii): directed path expensive to find, cheap to protect",
          byname["path k=6"]["r_v"] == 6 and abs(byname["path k=6"]["tau_star_v"] - 3) < 1e-9,
          f"r_v={byname['path k=6']['r_v']}, tau*={byname['path k=6']['tau_star_v']}")
    check("E12 fixed point: cycles have r_v == tau*_v",
          all(byname[c]["r_v"] == byname[c]["tau_star_v"] for c in ("C4", "C6", "C8", "C12")))
    check("E12 bippr has no forward/backward route on bipartite clusters (r_v = inf)",
          byname["bipartite K_{2,2}"]["r_v"] is None
          and byname["bipartite K_{2,2}"]["r_v_mixed"] == 2)
    check("E12 scatter-gather: tau*_v = 2 independent of fan width b",
          all(abs(byname[f"scatter-gather b={b}"]["tau_star_v"] - 2) < 1e-9 for b in (2, 3)))

    hc = P["E13_hypercube"]
    check("E13 anchored HyperCube achieves DETERMINISTIC detection (P_det = 1) everywhere",
          all(r["P_det"] == 1.0 for o in hc.values() for r in o["records"]))
    check("E13 measured rho_max exponent == 1 - 1/tau*_v (<=0.01)",
          all(abs(o["measured_exponent"] - o["predicted_exponent"]) <= 0.01 for o in hc.values()),
          ", ".join(f"{n}: {o['measured_exponent']:.4f} vs {o['predicted_exponent']:.4f}"
                    for n, o in hc.items()))
    viol = [(n, r["K"], r["rho_max"], r["paper_floor"])
            for n, o in hc.items() for r in o["records"]
            if o["tau_star_v"] < o["d"] - 1e-9 and r["rho_max"] < r["paper_floor"]]
    check("E13 K^{1-1/d} is NOT a floor: design beats it at every K with P_det=1",
          len(viol) == sum(len(o["records"]) for o in hc.values()
                           if o["tau_star_v"] < o["d"] - 1e-9),
          f"{len(viol)} witnesses, e.g. {viol[-1]}")

    dv = P["E14_design_vs_random"]
    check("E14 design/random gap exponent matches 1/tau*_v - 1/d (<=0.01)",
          all(abs(o["measured_gap_exponent"] - o["predicted_gap_exponent"]) <= 0.01
              for o in dv.values()),
          ", ".join(f"{n}: {o['measured_gap_exponent']:.4f}" for n, o in dv.items()))
    check("E14 gap is polynomial in K (grows, not O(sqrt(log N)) constant)",
          all(o["records"][-1]["gap"] > o["records"][0]["gap"] for o in dv.values()))

    cr = P["E15_certified_robustness"]
    check("E15 certified m_s* improves >= 3x at K = 16384",
          all(o["records"][-1]["improvement"] >= 3.0 for o in cr.values()),
          ", ".join(f"{n}: {o['records'][-1]['improvement']:.1f}x" for n, o in cr.items()))

    pb = P["E16_packing_counting_bound"]
    check("E16 packing counting bound holds: embeddings per cell <= b^{tau*_v}",
          all(o["packing_bound_holds"] for o in pb.values()),
          f"max ratio {max(o['max_ratio_to_b_pow_tau'] for o in pb.values()):.4f}")
    check("E16 binom(b,d) is tight only on the matching case (star)",
          abs(pb["star d=3"]["max_ratio_to_binom_bd"] - 1.0) < 1e-9
          and all(pb[m]["max_ratio_to_binom_bd"] < 0.2
                  for m in ("triangle (C3)", "C4", "C5", "scatter-gather b=2")))

    sk = P["E17_skew"]["records"]
    check("E17 skew inflates max/mean cell load (constants, not exponent)",
          all(1.0 < r["skew_ratio"] < 5.0 for r in sk),
          f"ratios {['%.2f' % r['skew_ratio'] for r in sk]}")

    qa = P["E18_offgraph_multihop_qa"]
    check("E18 OFF-GRAPH query: tau*_A < d (2 vs 3) on ternary relations, no graph",
          qa["tau_star_A"] < qa["d"] - 1e-9,
          f"tau*_A={qa['tau_star_A']:g}, d={qa['d']}")
    check("E18 share-LP identity holds off-graph (zero duality gap)",
          qa["lp_duality_gap"] < 1e-9, f"gap {qa['lp_duality_gap']:.2e}")
    check("E18 HyperCube: deterministic detection off-graph (P_det = 1)",
          all(r["P_det"] == 1.0 for r in qa["records"]))
    check("E18 measured exponent == 1 - 1/tau*_A off-graph",
          abs(qa["measured_exponent"] - qa["predicted_exponent"]) <= 0.01,
          f"{qa['measured_exponent']:.4f} vs {qa['predicted_exponent']:.4f}")
    check("E18 arity floor beaten off-graph at every K",
          all(r["rho_max"] < r["paper_floor"] for r in qa["records"]))
    check("E18 certificate is TIGHT off-graph: flip budget == ceil(mu/rho_max)",
          all(r["flip_budget"] == r["cert_predicted_budget"] for r in qa["records"]),
          "pred/meas " + ", ".join(f"{r['cert_predicted_budget']}/{r['flip_budget']}"
                                   for r in qa["records"]))

# ---------------------------------------------------------------------------
# RD1-RD2: real data (results_realdata.json)
# ---------------------------------------------------------------------------
_rd = os.path.join(HERE, "results_realdata.json")
if not os.path.exists(_rd):
    print("\n[skip] results_realdata.json absent — run realdata.py")
else:
    D = json.load(open(_rd))
    print("\n--- RD1-RD2: real data ---")

    r1 = D["RD1_tpch"]
    check("RD1 TPC-H: tau* < d on >= 85% of multi-relation queries",
          r1["frac_overpriced"] >= 0.85,
          f"{r1['n_tau_lt_d']}/{r1['n_multi_relation']} = {100*r1['frac_overpriced']:.0f}%")
    check("RD1 share-LP identity holds on every real query (zero duality gap)",
          r1["max_lp_duality_gap"] < 1e-9, f"max gap {r1['max_lp_duality_gap']:.1e}")
    check("RD1 median certified-budget gain >= 3x at K=1024",
          r1["median_budget_ratio"] >= 3.0, f"{r1['median_budget_ratio']:.2f}x")
    check("RD1 max certified-budget gain is 32x (two-table queries need NO replication)",
          abs(r1["max_budget_ratio"] - 32.0) < 0.5,
          f"{r1['max_budget_ratio']:.1f}x on {r1['argmax_query']}")
    twotable = [r for r in r1["rows"] if r["d"] == 2 and r.get("join_vars")]
    check("RD1 every 2-table FK join has tau* = 1 (rho = K^0: no replication at all)",
          all(abs(r["tau_star"] - 1.0) < 1e-9 for r in twotable),
          f"{len(twotable)} queries")
    check("RD1 context-window rule flips on >= half the multi-relation queries",
          r1["n_context_rule_flipped"] >= r1["n_multi_relation"] / 2,
          f"{r1['n_context_rule_flipped']}/{r1['n_multi_relation']}: "
          f"{r1['context_rule_flipped']}")
    anch = [r for r in r1["rows"] if r["d"] > 1 and r.get("anchor")]
    check("RD1 anchoring never lowers tau* (it removes a packing constraint)",
          all(r["tau_star_anchored"] >= r["tau_star"] - 1e-9 for r in anch),
          f"{len(anch)} anchored queries")

    for key, label in (("RD2_pegase_2path", "2-path"), ("RD2_pegase_C4", "C4")):
        o = D[key]
        check(f"RD2 [{label}] deterministic detection on the REAL network (P_det = 1)",
              all(r["min_P_det"] == 1.0 for r in o["records"]),
              f"{o['network']}, n={o['n_nodes']}, m={o['n_edges']}")
        check(f"RD2 [{label}] real-topology exponent stays below the arity reading",
              o["measured_exponent"] < o["paper_exponent"] - 0.05,
              f"measured {o['measured_exponent']:.3f} vs arity {o['paper_exponent']:.3f} "
              f"vs packing {o['predicted_exponent']:.3f}")
    o2 = D["RD2_pegase_2path"]
    check("RD2 [2-path] tau*_v = 1: replication is O(1), flat in K on real data",
          abs(o2["tau_star_v"] - 1.0) < 1e-9
          and len({r["max_rho_max"] for r in o2["records"]}) == 1,
          f"rho_max {[r['max_rho_max'] for r in o2['records']]} "
          f"vs arity floor {[round(r['paper_floor'],1) for r in o2['records']]}")
    o4 = D["RD2_pegase_C4"]
    for key in ("RD2_pegase_2path", "RD2_pegase_C4", "RD3_skew_aware_C4"):
        o = D[key]
        d_, t_ = o["d"], o["tau_star_v"]
        exact = [d_ * (r["K"] ** (1 - 1 / t_) - 1) if t_ > 1 else float(d_)
                 for r in o["records"]]
        check(f"RD [{key}] rho_max matches the EXACT slot formula d(K^(1-1/tau*)-1)",
              all(r["max_rho_max"] == round(e) for r, e in zip(o["records"], exact)),
              f"rho_max {[r['max_rho_max'] for r in o['records']]} vs "
              f"{[round(e) for e in exact]}")
        check(f"RD [{key}] rho_max <= d * K^(1-1/tau*) (slot multiplicity <= d)",
              all(r["max_rho_max"] <= d_ * r["K"] ** (1 - 1 / t_) + 1e-9
                  for r in o["records"]))
    check("RD2 the 0.580 log-log slope is the -1 in the exact formula, not skew: "
          "rho_max/(d K^(1-1/tau*)) rises toward 1",
          [r["max_rho_max"] / (o4["d"] * r["K"] ** (1 - 1 / o4["tau_star_v"]))
           for r in o4["records"]] == sorted(
              [r["max_rho_max"] / (o4["d"] * r["K"] ** (1 - 1 / o4["tau_star_v"]))
               for r in o4["records"]]),
          "0.750 -> 0.875 -> 0.938")
    sk = D["RD3_skew_aware_C4"]
    check("RD3 frequency-balanced bucketing cuts LOAD skew but leaves rho_max alone "
          "(load skew and amplification are different quantities)",
          sk["records"][-1]["mean_skew"] < o4["records"][-1]["mean_skew"] - 0.5
          and all(a["max_rho_max"] == b["max_rho_max"]
                  for a, b in zip(sk["records"], o4["records"])),
          f"skew {o4['records'][-1]['mean_skew']:.2f} -> "
          f"{sk['records'][-1]['mean_skew']:.2f}, rho_max unchanged")

# ---------------------------------------------------------------------------
# E19-E20: the attack calculus vs implemented adversaries (results_attack.json)
# ---------------------------------------------------------------------------
_at = os.path.join(HERE, "results_attack.json")
if not os.path.exists(_at):
    print("\n[skip] results_attack.json absent — run attack.py")
else:
    A = json.load(open(_at))
    print("\n--- E19-E20: attack calculus ---")

    x1 = A["X1_calculus_vs_montecarlo"]
    check("E19a Hoeffding success bound NEVER violated (this is the certified one)",
          x1["chernoff_violations"] == 0,
          f"0/{x1['n_points']} configurations")
    check("E19a mean-field estimate accurate (MAE <= 0.02) but two-sided",
          x1["mean_abs_error_meanfield"] <= 0.02,
          f"MAE {x1['mean_abs_error_meanfield']:.4f}, max {x1['max_abs_error_meanfield']:.4f}")

    x4 = A["X4_surface_identity_real"]
    check("E19b surface identity S = sum_e rho(e) = K*Bbar holds EXACTLY on a real network",
          all(r["identity_max_gap"] == 0 for r in x4["records"]),
          f"{x4['network']}, n={x4['n_nodes']}, gaps "
          f"{[r['identity_max_gap'] for r in x4['records']]}")
    check("E19b attack surface is not concentrated on one edge (top-1 share < 20%)",
          all(r["max_top1_share"] < 0.2 for r in x4["records"]),
          f"max top-1 share {max(r['max_top1_share'] for r in x4['records']):.3f}")

    x5 = A["X5_attacker_mix"]
    check("E19c attacker-mix threshold rule predicts the cheaper channel exactly",
          x5["threshold_rule_holds"] and x5["exact_le_rho"],
          f"{len(x5['rows'])}/{len(x5['rows'])} configurations; exact threshold <= rho")

    x2 = A["X2_budget_separation"]
    check("E20a majority regime: blinding buys <= 2x (matches the theorem's ceiling)",
          all(1.0 <= r["separation"] <= 2.05 for r in x2["rows"]),
          f"separations {[round(r['separation'],1) for r in x2['rows']]}")
    check("E20a closed-form blind budget matches the implemented attacker (<= 1 poison)",
          all(abs(r["blind_budget_measured"] - r["blind_budget_formula"]) <= 1.0
              for r in x2["rows"]),
          "max gap %.2f" % max(abs(r["blind_budget_measured"] - r["blind_budget_formula"])
                               for r in x2["rows"]))
    check("E20c adaptivity buys at most a few percent vs the blind attacker",
          x2["adaptive_over_blind"] <= 1.10,
          f"{x2['adaptive_over_blind']:.3f}x")

    x2b = A["X2b_onset_regime"]
    check("E20b onset regime: blinding buys a POLYNOMIAL factor, fitted ~ K^{1/tau*}",
          abs(x2b["fitted_exponent_tau2"] - 0.5) <= 0.06
          and abs(x2b["fitted_exponent_tau3"] - 1 / 3) <= 0.06,
          f"tau=2: {x2b['fitted_exponent_tau2']:.3f} vs 0.500; "
          f"tau=3: {x2b['fitted_exponent_tau3']:.3f} vs 0.333")
    check("E20b separation grows to >= 50x at the largest K",
          max(r["separation"] for r in x2b["rows"]) >= 50,
          f"max {max(r['separation'] for r in x2b['rows']):.1f}x")
    zs = [abs(r["z_vs_exact"]) for r in x2b["rows"] if r.get("z_vs_exact") is not None]
    check("E20b measured blind budget matches the exact geometric mean K/rho (|z| < 3)",
          max(zs) < 3.0, f"max |z| = {max(zs):.2f} over {len(zs)} settings")

    x3 = A["X3_slice_penalty"]
    check("E20d slice and uniform rho-subset have the SAME mean damage",
          x3["max_mean_gap"] <= 0.5, f"max |mean - predicted| = {x3['max_mean_gap']:.3f} hits")
    check("E20d uniform rho-subset variance matches the exact hypergeometric",
          x3["uniform_matches_hypergeom"], "all 12 settings within 15%")
    check("E20d GENERIC firing set: private-HyperCube ~ idealised model (inflation < 2x)",
          x3["generic_max_inflation"] < 2.0,
          f"max inflation {x3['generic_max_inflation']:.2f}x, budget ratio "
          f"{[round(r,3) for r in x3['generic_budget_ratio_range']]}")
    check("E20d ALIGNED firing set is the worst case and obeys the rho bound",
          x3["aligned_within_bound"] and x3["aligned_max_inflation"] > 5,
          f"max inflation {x3['aligned_max_inflation']:.1f}x, all <= rho")

    x6 = A["X6_bernstein_vs_hoeffding"]
    check("E20e min(Bernstein, Hoeffding) is never worse than Hoeffding alone",
          x6["min_never_worse"], f"{len(x6['rows'])} configurations")
    check("E20e neither tail bound dominates in general (Bernstein can lose near q=1/2)",
          "n_bernstein_alone_worse" in x6,
          f"Bernstein alone looser in {x6['n_bernstein_alone_worse']}/{len(x6['rows'])} "
          "of the tested settings; 4.1% over the wider random sweep")
    check("E20e Bernstein buys extra certified budget (mean > 0, max >= 25%)",
          x6["mean_gain_pct"] > 0 and x6["max_gain_pct"] >= 25,
          f"mean +{x6['mean_gain_pct']:.1f}%, max +{x6['max_gain_pct']:.0f}%")

    x7 = A.get("X7_kappa_block_certificate")
    if x7:
        check("E23 kappa estimator (Var ratio) recovers the block size to within 15%",
              x7["kappa_estimator_accurate"],
              ", ".join(f"kappa={r['kappa']}->{r['var_ratio']:.2f}" for r in x7["rows"]))
        check("E23 block-Hoeffding bound of Theorem kappa never violated",
              x7["block_hoeffding_never_violated"],
              f"{x7['n_tail_points']} tail points across 5 settings")

# --- E27: the law is not cycle-specific --------------------------------------
_e27 = json.load(open(os.path.join(HERE, "results_packing.json"))).get("E27_all_patterns")
if _e27:
    print("\n--- E27: every connected pattern, not a curated list ---")
    check("E27 tau* < d on the overwhelming majority of ALL patterns",
          _e27["frac_strict"] > 0.85,
          f"{_e27['n_strict']}/{_e27['n_cases']} shape-anchor pairs, "
          f"{_e27['n_graphs']} shapes")
    check("E27 cycles are not the beneficiaries (trees gain at least as much)",
          _e27["by_family"]["tree"]["median_gain"] >= _e27["by_family"]["cycle"]["median_gain"],
          f"tree {_e27['by_family']['tree']['median_gain']:.2f}x vs "
          f"cycle {_e27['by_family']['cycle']['median_gain']:.2f}x")
    check("E27 the exceptions are exactly the matchings (stars at the centre)",
          _e27["by_family"]["star"]["n"] - _e27["by_family"]["star"]["n_strict"]
          == _e27["n_cases"] - _e27["n_strict"],
          f"{_e27['n_cases'] - _e27['n_strict']} non-strict, all stars")
    check("E27 single atom reproduces the disjoint-partition regime (tau*=1, rho=1)",
          abs(_e27["corner_single_atom"] - 1.0) < 1e-9)
    check("E27 a matching reproduces the arity rate (tau*=d)",
          abs(_e27["corner_matching"] - 4.0) < 1e-9)
    check("E27 the anchor matters, not only the shape (same star, tau* 4 -> 1)",
          abs(_e27["corner_star_centre"] - 4.0) < 1e-9
          and abs(_e27["corner_star_leaf"] - 1.0) < 1e-9)
    check("E27 tau* monotone under adding an atom (a class is bounded by its largest)",
          _e27["mono_violations"] == 0,
          f"{_e27['mono_trials']} random queries, strict in {_e27['mono_strict']}")
    check("E27 two-hop entity QA needs no replication at all",
          _e27["qclass"]["2hop_entity_qa"]["rho"] == 1.0
          and _e27["qclass"]["2hop_entity_qa"]["gain"] == 32.0)
    check("E27 one shared partition for a class costs more than re-drawing per query",
          _e27["qclass_pooled_rho"] > _e27["qclass_per_query_max_rho"],
          f"{_e27['qclass_pooled_rho']:.0f} vs {_e27['qclass_per_query_max_rho']:.0f}")

# --- E25: kappa is bounded by the damage the certificate already permits ---
_x8 = A.get("X8_kappa_is_bounded")
if _x8:
    print("\n--- E25: the correlation budget is not a free parameter ---")
    check("E25 max component <= min(K, rho*m_s) in every configuration",
          _x8["component_bound_holds"], f"{_x8['n_settings']} settings")
    check("E25 chaining pushes kappa strictly above rho (single-payload reading unsafe)",
          _x8["max_kappa_over_rho"] > 1.0,
          f"max kappa/rho = {_x8['max_kappa_over_rho']:.2f}")
    check("E25 certified regime forces kappa <= rho*m_s < mu <= K",
          _x8["certified_regime_violations"] == 0,
          f"{_x8['certified_regime_draws']} random draws")
    check("E25 block-Hoeffding at kappa = largest component never violated",
          _x8["block_hoeffding_never_violated"],
          f"{_x8['tail_points']} tail points, max excess {_x8['max_excess']:.2e}")

# --- E26: end-to-end test of the kappa-free certificate --------------------
_x9 = A.get("X9_end_to_end_kappa_free")
if _x9:
    print("\n--- E26: the certificate, end to end ---")
    check("E26 certificate never breached in a certified setting",
          _x9["n_violations"] == 0,
          f"{_x9['n_certified']} settings, max excess {_x9['max_excess']:+.3f}")
    check("E26 empirical failure stays strictly below delta",
          _x9["max_excess"] < 0, f"max(empirical - delta) = {_x9['max_excess']:+.3f}")
    check("E26 negative control breaks the verdict below the margin (test has power)",
          _x9["control_is_informative"],
          f"{_x9['control_failures']}/{_x9['n_control']} control settings fail")

# --- E24: skew separation -------------------------------------------------
_pk = json.load(open(os.path.join(HERE, "results_packing.json")))
if "E24_skew_separation" in _pk:
    e24 = _pk["E24_skew_separation"]
    print("\n--- E24: skew separation ---")
    check("E24 rho is invariant to the data distribution", e24["rho_invariant"])
    check("E24 predicted rho matches measured rho in every skew regime",
          all(r["rho_meas"] == r["rho_pred"] for r in e24["rows"]))
    check("E24 single-coordinate load lower bound holds", e24["lb_single_holds"])
    check("E24 product load lower bound holds", e24["lb_product_holds"])
    check("E24 load DOES move under skew (else the separation is vacuous)",
          e24["max_inflation"] > 3.0, f"max/mean = {e24['max_inflation']:.2f}")
    check("E24 anchoring never decreases tau*_A", e24["anchor_monotone_violations"] == 0,
          f"{e24['anchor_monotone_trials']} random queries")
    check("E24 anchoring strictly increases tau*_A sometimes (non-vacuous)",
          e24["anchor_strict_increases"] > 0,
          f"strict in {e24['anchor_strict_increases']}")
    check("E24 pinning a triangle variable raises tau* from 3/2 to 2",
          abs(e24["triangle_tau_free"] - 1.5) < 1e-9
          and abs(e24["triangle_tau_pinned"] - 2.0) < 1e-9)
    check("E24 pinning costs the 3.2x in rho_max the paper quotes",
          abs(e24["pinning_cost"] - 3.2) < 0.05,
          f"{e24['pinning_cost']:.2f}x")


# --- E28-E31: the security experiments -------------------------------------
_mt_path = os.path.join(HERE, "results_multitarget.json")
# --- E32/E33/E36: exact rho_F, maximum packing, the duplicate attack ------------------
_W = os.path.join(HERE, "results_witness.json")
if os.path.exists(_W):
    print("\n--- E32/E33/E36: reachable damage, witness packing, duplicate attack ---")
    W = json.load(open(_W))
    _recs = [r for k in ("E32_E33_2path", "E32_E33_C4") for r in W[k]["records"]]
    _rows = [row for r in _recs for row in r["rows"]]
    check("E32 rho_F never exceeds rho_max (the certificate is a bound)",
          all(r["rho_F_never_exceeds_rho_max"] for r in _recs))
    check("E32 exact rho_F equals rho_max at the median on both motifs",
          all(r["med_rho_F"] == r["max_rho_max"] for r in _recs),
          "so the instance certificate is sound but not tighter here")
    check("E33 the exact packing is never smaller than greedy",
          all(row["omega"] >= row["omega_greedy"] for row in _rows))
    check("E33 greedy is not always maximum (else the ILP is pointless)",
          any(r["greedy_is_maximum"] < r["n_anchors"] for r in _recs),
          "greedy = max at %s" % [(r["greedy_is_maximum"], r["n_anchors"]) for r in _recs])
    check("E36 a copy of any witness edge reaches at least one firing cell, every anchor",
          all(r["dup_min_at_least_1"] for r in _recs))
    check("E36 the blind copy never beats the white-box rho_F",
          all(row["dup_blind"] <= row["rho_F"] for row in _rows))
    check("E36 the blind copy reaches about half of rho_F at the median",
          all(0.4 <= r["dup_blind_over_rho_F"] <= 0.7 for r in _recs),
          "medians %s" % [round(r["dup_blind_over_rho_F"], 2) for r in _recs])
    _p256 = [r for r in W["E32_E33_2path"]["records"] if r["K"] == 256][0]
    check("E36 on the sparse 2-path at K=256 the copy beats a fresh content by > 10x",
          _p256["dup_blind_over_fresh"] > 10, "%.1fx" % _p256["dup_blind_over_fresh"])

# --- real KGQA workloads ------------------------------------------------------------
_KG = os.path.join(HERE, "results_kgomega.json")
if os.path.exists(_KG):
    print("\n--- kg-omega: MetaQA and Hetionet ---")
    KG = json.load(open(_KG))
    _all = [KG["metaqa"]["2-hop"], KG["metaqa"]["3-hop"], *KG["hetionet"]["metapaths"].values()]
    check("kg every extracted witness family is physically edge-disjoint",
          all(s["disjointness_violations"] == 0 for s in _all))
    check("kg no answer was skipped as a hub", all(s["answers_skipped_hub"] == 0 for s in _all))
    check("kg every MetaQA test question was scored",
          all(KG["metaqa"][h]["n_questions_scored"] == KG["metaqa"][h]["n_questions_total"] for h in ("2-hop", "3-hop")))
    check("kg every MetaQA template path is inferred with Jaccard >= 0.9 against the gold answers",
          all(p["support"] >= 0.9 for h in ("2-hop", "3-hop") for p in KG["metaqa"][h]["templates"].values()),
          "min %.3f" % min(p["support"] for h in ("2-hop", "3-hop") for p in KG["metaqa"][h]["templates"].values()))
    check("kg the median answer rests on one chain in every workload",
          all(s["median"] == 1 for s in _all))
    check("kg fewer than 2% of MetaQA answers certify even one forged relation",
          all(KG["metaqa"][h]["frac_certify_ge1"] < 0.02 for h in ("2-hop", "3-hop")))
    check("kg well-corroborated answers exist (the ceiling is not vacuous everywhere)",
          max(s["max"] for s in _all) >= 10, "max omega %d" % max(s["max"] for s in _all))

# --- the corroboration limit: omega <= nu* <= tau, and what the transversal rule buys --------
_MGp = os.path.join(HERE, "results_menger.json")
if os.path.exists(_MGp):
    print("\n--- E34: omega, nu*, tau on PEGASE (Thm limit) ---")
    _S = json.load(open(_MGp))["E34_fractional"]["summary"]
    _p2, _c4 = _S["2-path (v-x-y)"], _S["C4 (v-x-y-z)"]
    check("E34 omega <= nu* <= tau at every anchor of both motifs",
          _p2["chain_holds"] and _c4["chain_holds"])
    check("E34 on the 2-path (a path) all three coincide, as Menger says",
          _p2["n_tau_eq_omega"] == _p2["n_anchors"] and _p2["max_gap"] == 1.0)
    check("E34 on the C4 the integral gap tau/nu* is real (> 1.5)",
          _c4["max_tau_over_nu"] > 1.5, "max %.2f" % _c4["max_tau_over_nu"])
    check("E34 the transversal rule certifies more than twice witness packing on the C4 (median)",
          _c4["med_b_tau"] > 2 * _c4["med_b_int"], "%g against %g" % (_c4["med_b_tau"], _c4["med_b_int"]))

_KC = os.path.join(HERE, "results_kgclaims.json")
if os.path.exists(_KC):
    print("\n--- claim-check: Hetionet treatment claims under seven metapaths ---")
    KC = json.load(open(_KC))
    _t, _n = KC["true_claims"]["summary"], KC["non_claims"]["summary"]
    check("claims every CtD edge and an equal number of non-claims scored",
          _t["n"] == 755 and _n["n"] == 755)
    check("claims no witness enumeration hit the cap", _t["n_capped"] == 0 and _n["n_capped"] == 0)
    check("claims omega <= nu* <= tau on every pair",
          _t["omega_le_nu"] and _t["nu_le_tau"] and _n["omega_le_nu"] and _n["nu_le_tau"])
    check("claims most true claims certify one forged relation (tau >= 3)",
          _t["pct_neut_t1"] > 50, "%.1f%%" % _t["pct_neut_t1"])
    check("claims the threshold separates claims from non-claims by > 5x at t = 1",
          _t["pct_neut_t1"] > 5 * _n["pct_neut_t1"], "%.1f%% vs %.1f%%" % (_t["pct_neut_t1"], _n["pct_neut_t1"]))

# --- end to end: fact isolation with a real LLM reader ----------------------------------
_E2E = os.path.join(HERE, "results_e2e.json")
if os.path.exists(_E2E):
    print("\n--- e2e: fact isolation, real reader, four attacks ---")
    E = json.load(open(_E2E))
    _m, _s = E["meta"], E["summary"]
    check("e2e was run against a real model, not the offline stand-in",
          not _m["dry_run"] and _m["model"] not in ("", "mock"), _m["model"])
    check("e2e every reply was complete (no reply cut at max_tokens)", _m["truncated"] == 0,
          "%d truncated" % _m["truncated"])
    check("e2e workload sizes as stated: 200 MetaQA questions, 150 Hetionet claims",
          _s["clean"]["metaqa"]["n"] == 200 and _s["clean"]["hetionet"]["n"] == 150)
    check("e2e all four attacks ran at both budgets on both datasets' applicable sides",
          all(any(k.split("/")[1] == a for k in _s["attacks"]) for a in ("gragpoison", "kepo", "mincut", "injection")))
    check("e2e SOUND: no attack within the clean certified budget changed a defended report",
          _s["sound"] and all(v["success_within_cert"] == 0 for v in _s["attacks"].values()))
    _rows = E["attack_rows"]
    check("e2e every attacked report change spent more operations than the certificate allowed",
          all(r["ops"] > r["clean_cert"] for r in _rows if r["defended_success"]),
          "%d report changes" % sum(r["defended_success"] for r in _rows))
    check("e2e the reader's honest false-negative rate is measured on both datasets",
          any(k.startswith("metaqa") for k in _s["reader"]["honest_fn"]) and
          any(k.startswith("hetionet") for k in _s["reader"]["honest_fn"]))

if os.path.exists(_mt_path):
    M = json.load(open(_mt_path))
    print("\n--- E28-E31: security experiments ---")
    e28 = M["E28_observed_amplification"]
    check("E28 observed amplification never exceeds the bound",
          e28["bound_never_exceeded"], f"{len(e28['rows'])} configs")
    check("E28 the bound is attained, not merely respected (else uninformative)",
          e28["bound_attained"], f"ratio = {e28['max_ratio']:.3f} everywhere")
    e29 = M["E29_multi_target"]
    check("E29 multi-target corruption stays within the linear bound",
          e29["all_within_linear"])
    check("E29 per-target amplification is independent of the number of targets",
          e29["per_target_constant_in_m"],
          f"= {e29['per_target_value']} for every m")
    check("E29 Gamma grows linearly in m, not faster",
          abs(e29["gamma_slope"] - 1.0) < 0.05, f"slope {e29['gamma_slope']:.2f}")
    e30 = M["E30_adaptive"]
    check("E30 adaptive full-knowledge attacker never beats b*rho_max",
          e30["violations"] == 0, f"{e30['n']} configs, max ratio {e30['max_ratio']:.3f}")
    e31 = M["E31_tightness"]
    check("E31 certificate is sound (break strictly above cert everywhere)",
          e31["certificate_sound"])
    check("E31 certificate is non-vacuous (b_cert >= 1)", e31["certificate_nonvacuous"])
    check("E31 certificate is tight (break within 1.5x of cert)",
          e31["max_tightness"] < 1.5,
          f"T in [{e31['min_tightness']:.2f}, {e31['max_tightness']:.2f}]")

print()
if ok:
    print("ALL CLAIMS VERIFIED against results files ✓")
    sys.exit(0)
else:
    print("SOME CLAIMS FAILED — findings and results files disagree ✗")
    sys.exit(1)
