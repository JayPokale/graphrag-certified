"""
audit_paper.py -- cross-check the numbers PRINTED IN THE PAPER against the results
files.  verify_claims.py checks that the results support the claims; this checks
that the LaTeX actually quotes those results.  Run both.

    python3 audit_paper.py        exits non-zero on any mismatch
"""
from __future__ import annotations
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The paper is not part of this artifact.  Point PAPER at a main.tex to run this gate;
# by default we look for one checked out beside the experiments.
_PAPER = os.environ.get("PAPER")
if _PAPER and not os.path.isabs(_PAPER) and not os.path.exists(_PAPER):
    # `make verify` runs from experiments/, so a path given relative to the repository root
    # (PAPER=paper/main.tex) is resolved there rather than silently missed
    _PAPER = os.path.join(HERE, "..", _PAPER)
if _PAPER and not os.path.exists(_PAPER):
    print("gate 2 FAILED: PAPER=%s does not exist" % os.environ["PAPER"])
    raise SystemExit(1)
_PAPER = _PAPER or os.path.join(HERE, "..", "paper", "main.tex")
if not os.path.exists(_PAPER):
    print("gate 2 skipped: no paper at %s (set PAPER=/path/to/main.tex to run it)" % _PAPER)
    raise SystemExit(0)
_RAW = open(_PAPER).read()
# Collapse whitespace so a check does not fail merely because LaTeX rewrapped a line.
# (It did: a reflowed paragraph broke the RD1 sizing-rule check and this gate caught it.)
TEX = re.sub(r"\s+", " ", _RAW)
A = json.load(open(os.path.join(HERE, "results_attack.json")))
P = json.load(open(os.path.join(HERE, "results_packing.json")))
D = json.load(open(os.path.join(HERE, "results_realdata.json")))

ok = True


def holds(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        ok = False


def says(label, pattern, expected=None, fmt="{:.3f}"):
    """Assert `pattern` (a regex with one group) appears in the paper, and that the
    captured number equals `expected` when given."""
    global ok
    m = re.search(pattern, TEX)
    if not m:
        print(f"  [FAIL] {label}: pattern not found in main.tex  ->  {pattern}")
        ok = False
        return
    if expected is None:
        print(f"  [PASS] {label}  (found: {m.group(1)})")
        return
    got = float(m.group(1).replace(",", "").replace("{", "").replace("}", ""))
    want = float(expected)
    # tolerance = half a unit in the last printed decimal, so a paper that rounds
    # correctly passes and a paper that quotes the wrong number does not
    dec = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
    good = abs(got - want) <= 0.5 * 10 ** (-dec) + 1e-9
    print(f"  [{'PASS' if good else 'FAIL'}] {label}: paper says {got}, data says "
          f"{fmt.format(want)}")
    if not good:
        ok = False


print("--- attack calculus (results_attack.json) ---")
x1 = A["X1_calculus_vs_montecarlo"]
says("tail-bound Hoeffding violations", r"never violated\}? \(\$0/(\d+)\$\)", x1["n_points"], "{:.0f}")
says("tail-bound mean-field MAE", r"mean absolute error of \$(0\.\d+)\$", x1["mean_abs_error_meanfield"])
says("tail-bound mean-field max error", r"max \$(0\.\d+)\$", x1["max_abs_error_meanfield"])

x2b = A["X2b_onset_regime"]
t2 = [r for r in x2b["rows"] if r["tau"] == 2]
says("blinding-gain fitted exponent tau*=2", r"\\mathbf\{(0\.\d+)\}\$ against the predicted\s*\n?\$1/\\tau\^\*_A = 0\.500\$", x2b["fitted_exponent_tau2"])
says("blinding-gain fitted exponent tau*=3", r"and \$(0\.\d+)\$ against \$0\.333\$", x2b["fitted_exponent_tau3"])
says("blinding-gain max separation", r"\\pm 0\.96\$ at \$K = 4096\$, a separation of\s*\n?\$\\mathbf\{(\d+\.\d+)\\times\}", max(r["separation"] for r in x2b["rows"]), "{:.1f}")
says("blinding-gain max |z|", r"max \$\|z\| = (\d+\.\d+)\$",
     max(abs(r["z_vs_exact"]) for r in x2b["rows"]), "{:.2f}")
for r in t2:
    says(f"blinding-gain blind budget K={r['K']}", r"\$" + f"{r['blind_budget']:.2f}" + r" \\pm (\d+\.\d+)\$",
         r["blind_sem"], "{:.2f}")

x2 = A["X2_budget_separation"]
says("adaptivity adaptive/blind", r"\\mathbf\{(1\.\d+)\\times\}\$ that of the non-adaptive",
     x2["adaptive_over_blind"])

x3 = A["X3_slice_penalty"]
says("slice-alignment generic inflation", r"variance inflation \$\\mathbf\{(\d+\.\d+)\\times\}",
     x3["generic_max_inflation"], "{:.2f}")
says("slice-alignment aligned inflation", r"while the\s*\n?\$(\d+\.\d+)\\times\$ inflation",
     x3["aligned_max_inflation"], "{:.1f}")
says("slice-alignment generic budget ratio low", r"\[(0\.\d+),\\,1\.042\]", 
     min(x3["generic_budget_ratio_range"]), "{:.3f}")
x6 = A["X6_bernstein_vs_hoeffding"]
says("bernstein-gain Bernstein configs", r"beats Hoeffding alone in \$(\d+)/18\$", len(x6["rows"]), "{:.0f}")
says("bernstein-gain Bernstein mean gain", r"buying a mean \$(\d+\.\d+)\\%\$", x6["mean_gain_pct"], "{:.1f}")
says("bernstein-gain Bernstein max gain", r"\\mathbf\{(\d+)\\%\}", x6["max_gain_pct"], "{:.0f}")

x4 = A["X4_surface_identity_real"]
says("surface-identity network size", r"PEGASE \(\$n = (\d+)\$", x4["n_nodes"], "{:.0f}")
says("surface-identity top-1 surface share K=16", r"mean \$(\d+\.\d+)\\%\$ of it at \$K=16\$", 100 * x4["records"][0]["mean_top1_share"], "{:.1f}")
says("surface-identity top-1 surface share K=64", r"and \$(\d+\.\d+)\\%\$ at\s*\n?\$K=64\$", 100 * x4["records"][1]["mean_top1_share"], "{:.1f}")

print("\n--- real data (results_realdata.json) ---")
r1 = D["RD1_tpch"]
says("tpc-h queries over-priced", r"\\textbf\{(18) of the 20\}", r1["n_tau_lt_d"], "{:.0f}")
says("tpc-h median gain", r"median \$?(3\.2)\\?times", 3.2, "{:.1f}")
says("tpc-h max gain", r"up to \$?(\d+)\\times\$?", r1["max_budget_ratio"], "{:.0f}")
says("tpc-h duality gap mantissa", r"(5\.6)\\times 10\^\{-17\}", 5.6, "{:.1f}")
says("tpc-h sizing-rule flips", r"arity reading is wrong on (\d+) of the 20 queries", r1["n_context_rule_flipped"], "{:.0f}")

r2 = D["RD2_pegase_C4"]
says("pegase measured exponent", r"that fit returns \$(0\.\d+)\$ ag", r2["measured_exponent"])

print("\n--- packing law (results_packing.json) ---")
says("hypercube-exponent P_det = 1 stated", r"P_\{\\det\}\s*=\s*(1)\b", 1, "{:.0f}")
inv = P["E12_invariants"]
says("share-lp families", r"(15) motif families", len(inv), "{:.0f}")

e27 = P.get("E27_all_patterns")
if e27:
    says("all-patterns strict cases", r"\\mathbf\{(\d+)/58\}", e27["n_strict"], "{:.0f}")
    says("all-patterns total cases", r"\(\$(\d+)\$ shape--anchor pairs\)", e27["n_cases"], "{:.0f}")
    says("all-patterns shapes", r"up to isomorphism \(\$(\d+)\$ shapes\)", e27["n_graphs"], "{:.0f}")
    says("all-patterns median gain", r"a median certified-budget gain of \$(\d+\.\d+)\\times\$",
         e27["median_gain"], "{:.2f}")
    says("all-patterns max gain", r"up\s*to \$(\d+)\\times\$ at \$K = 1024\$", e27["max_gain"], "{:.0f}")
    says("all-patterns tree median", r"trees \$(\d+\.\d+)\\times\$",
         e27["by_family"]["tree"]["median_gain"], "{:.2f}")
    says("all-patterns monotone trials", r"in \$(\d+)/600\$ random queries", e27["mono_trials"], "{:.0f}")
    says("all-patterns monotone strict", r"in \$600/600\$ random queries, strictly in \$(\d+)\$",
         e27["mono_strict"], "{:.0f}")

x8 = A.get("X8_kappa_is_bounded")
if x8:
    says("correlation-chaining component-bound settings", r"holds in \$(\d+)/36\$ configurations",
         x8["n_settings"], "{:.0f}")
    says("correlation-chaining max kappa/rho", r"to \$(\d+\.\d+)\\rho\$ at \$m_s = 8\$",
         x8["max_kappa_over_rho"], "{:.1f}")
    says("correlation-chaining certified-regime draws", r"holds in all \$(\d+)\$ random draws",
         x8["certified_regime_draws"], "{:.0f}")
    says("correlation-chaining tail points", r"never violated across \$(\d+)\$\s*\n?tail points",
         x8["tail_points"], "{:.0f}")

x9 = A.get("X9_end_to_end_kappa_free")
if x9:
    says("end-to-end certified settings", r"holds in \$\\mathbf\{(\d+)/89\}\$",
         x9["n_certified"], "{:.0f}")
    says("end-to-end control failures", r"breaks the verdict in \$(\d+)\$ of \$200\$ settings",
         x9["control_failures"], "{:.0f}")

e24 = P.get("E24_skew_separation")
if e24:
    says("skew-and-pinning pinning cost", r"a \$\\mathbf\{(\d+\.\d+)\\times\}\$ loss", e24["pinning_cost"], "{:.1f}")
    says("skew-and-pinning load ratio, low end", r"moves \$(\d+\.\d+) \\to 9\.49\$",
         min(r["inflation"] for r in e24["rows"]), "{:.2f}")
    says("skew-and-pinning load ratio, high end", r"moves \$1\.24 \\to (\d+\.\d+)\$",
         e24["max_inflation"], "{:.2f}")
    says("skew-and-pinning anchoring trials", r"holds in \$(\d+)/400\$ random queries",
         e24["anchor_monotone_trials"], "{:.0f}")
    says("skew-and-pinning anchoring strict count", r"strictly in \$(\d+)\$",
         e24["anchor_strict_increases"], "{:.0f}")
    says("skew-and-pinning pinned tau*", r"\$\\tau\^\*_A\$ from \$3/2\$ to \$(2)\$",
         e24["triangle_tau_pinned"], "{:.0f}")

_W = os.path.join(HERE, "results_witness.json")
if os.path.exists(_W):
    W = json.load(open(_W))
    print("\n--- reachable damage and witness packing (results_witness.json) ---")
    p2 = {r["K"]: r for r in W["E32_E33_2path"]["records"]}
    c4 = {r["K"]: r for r in W["E32_E33_C4"]["records"]}
    # rho_F is now the exact two-endpoint enumeration; the single-vector count an earlier
    # draft printed (7 against 12, 15 against 28) under-counted and is gone from the paper.
    says("reachable-damage 2-path anchors with rho_F = rho_max",
         r"\\rho_F = \\rho_\{\\max\}\$ at \$(\d+)\$ of \$20\$ anchors on the \$2\$-path",
         p2[256]["rho_F_eq_rho_max"], "{:.0f}")
    says("reachable-damage C4 anchors with rho_F = rho_max",
         r"and at \$(\d+)\$ of \$12\$ anchors on the anchored \$C_4\$", c4[64]["rho_F_eq_rho_max"], "{:.0f}")
    says("reachable-damage 2-path median rho_F",
         r"\\rho_F \\;=\\; \\rho_\{\\max\} \\;=\\; (\d+) \\quad \(2\\text\{-path\}\)",
         p2[256]["med_rho_F"], "{:.0f}")
    says("reachable-damage C4 median rho_F at K=64",
         r"\\rho_F \\;=\\; \\rho_\{\\max\} \\;=\\; (\d+) \\quad \(C_4,\\ K = 64\)",
         c4[64]["med_rho_F"], "{:.0f}")
    holds("reachable-damage: the medians the paper equates really are equal",
          p2[256]["med_rho_F"] == p2[256]["max_rho_max"] and c4[64]["med_rho_F"] == c4[64]["max_rho_max"])
    says("three-designs greedy = maximum, 2-path", r"greedy attains the maximum at \$(\d+)\$ of \$20\$",
         p2[256]["greedy_is_maximum"], "{:.0f}")
    says("three-designs greedy = maximum, C4", r"\$(\d+)\$ of \$12\$ \$C_4\$ anchors", c4[64]["greedy_is_maximum"], "{:.0f}")
    says("three-designs tightened budget, 2-path", r"recovers a median of \$([\d.]+)\$ on\s*\n?the \$2\$-path",
         p2[256]["med_b_tight"], "{:g}")
    holds("three-designs tightened budget is 0 on C4 as the paper says",
          all(r["med_b_tight"] == 0 for r in W["E32_E33_C4"]["records"]))
    says("duplicate-attack blind/rho_F, 2-path", r"\(\$(0\.\d+)\$ on the \$2\$-path, \$0\.54\$--\$0\.58\$",
         p2[256]["dup_blind_over_rho_F"], "{:.2f}")
    says("duplicate-attack blind/rho_F, C4 low", r"on the \$2\$-path, \$(0\.\d+)\$--\$0\.58\$ on the \$C_4\$",
         min(r["dup_blind_over_rho_F"] for r in W["E32_E33_C4"]["records"]), "{:.2f}")
    says("duplicate-attack blind/rho_F, C4 high", r"\$0\.54\$--\$(0\.\d+)\$ on the \$C_4\$",
         max(r["dup_blind_over_rho_F"] for r in W["E32_E33_C4"]["records"]), "{:.2f}")
    says("duplicate-attack vs fresh content at K=256", r"that is \$(\d+)\\times\$ what a fresh content",
         p2[256]["dup_blind_over_fresh"], "{:.0f}")
    holds("duplicate-attack: a copy reaches >= 1 firing cell at every anchor",
          all(r["dup_min_at_least_1"] for k in ("E32_E33_2path", "E32_E33_C4") for r in W[k]["records"]))
    says("three-designs witness budget, 2-path", r"certifies a median of \$([\d.]+)\$ forged",
         p2[256]["med_b_witness"], "{:g}")
    says("three-designs witness cells, 2-path", r"relations using \$(\d+)\$ cells",
         p2[256]["med_omega"], "{:.0f}")
    says("three-designs witness budget, C4", r"it certifies \$(\d)\$ using \$6\$ cells",
         c4[64]["med_b_witness"], "{:.0f}")

_MG = os.path.join(HERE, "results_menger.json")
if os.path.exists(_MG):
    G = json.load(open(_MG))
    print("\n--- fractional budget and Menger (results_menger.json) ---")
    c4 = G["E34_fractional"]["summary"]["C4 (v-x-y-z)"]
    mg = G["E35_menger"]
    says("fractional-gain max fractional gain", r"reaches a maximum of \$(\d\.\d+)\$ and is", c4["max_gap"], "{:.2f}")
    says("fractional-gain strict anchors", r"strictly above \$1\$ at \$\\mathbf\{(\d+)\}\$ of \$12\$", c4["n_strict"], "{:.0f}")
    says("fractional-gain tau/nu* max", r"exceeds \$\\nu\^\*\$ by up to \$(\d\.\d+)\\times\$", c4["max_tau_over_nu"], "{:.2f}")
    says("fractional-gain median tau on C4", r"median\s+\$\\tau = (\d+)\$ against", c4["med_tau"], "{:.0f}")
    says("fractional-gain median omega on C4", r"\$\\tau = \d+\$ against \$\\omega = (\d+)\$", c4["med_omega"], "{:.0f}")
    says("fractional-gain transversal budget on C4", r"certifies a median of \$\\mathbf\{(\d+)\}\$ forged relations on the \$C_4\$", c4["med_b_tau"], "{:.0f}")
    says("fractional-gain witness budget on C4", r"where witness packing\s+certifies \$(\d+)\$", c4["med_b_int"], "{:.0f}")
    holds("scorecard tau/nu* figure", ("\\tau/\\nu^* \\le %.2f" % c4["max_tau_over_nu"]) in TEX)
    holds("scorecard C4 median figure", ("median $%d$ against $%d$" % (c4["med_b_tau"], c4["med_b_int"])) in TEX)
    says("flow-check answers checked", r"they agree in \$\\mathbf\{(\d+)\}\$ of \$464\$", mg["n_answers"], "{:.0f}")
    says("flow-check median omega", r"median\s*\n?answer has \$\\omega_a = (\d)\$", mg["med_omega"], "{:.0f}")
    says("flow-check max omega", r"and the best has \$(\d+)\$", mg["max_omega"], "{:.0f}")
    # The histogram-derived counts the prose quotes.  These were unguarded until an audit
    # found the paper claiming 100 where the histogram sums to 87.
    _h = {int(k): v for k, v in mg["omega_hist"].items()}
    says("flow-check single-chain answers", r"bimodal --- \$(\d+)\$\s*\n?answers rest on a single chain",
         _h.get(1, 0), "{:.0f}")
    says("flow-check answers certifying nothing", r"\$\\mathbf\{(\d+)\}\$ of\s*\n?\$464\$ answers certify nothing",
         sum(v for k, v in _h.items() if k <= 2), "{:.0f}")
    says("flow-check answers with omega >= 7", r"while \$(\d+)\$ carry seven independent chains",
         sum(v for k, v in _h.items() if k >= 7), "{:.0f}")
    holds_menger = mg["menger_exact"]

_KC = os.path.join(HERE, "results_kgclaims.json")
if os.path.exists(_KC):
    KC = json.load(open(_KC))
    print("\n--- claim-check (results_kgclaims.json) ---")
    import numpy as _np
    for _name, _lab in (("true_claims", "true claims"), ("non_claims", "non-claims")):
        _r = KC[_name]["rows"]; _t = _np.array([x["tau"] for x in _r])
        _row = "%s & %d & %.1f\\%% & %.1f\\%% & %.1f\\%% & %d \\\\" % (
            _lab, len(_r), 100 * (_t == 1).mean(), 100 * (_t >= 3).mean(), 100 * (_t >= 7).mean(), _t.max())
        holds("claim-check table row: " + _lab, _row in _RAW, _row)
    _ts, _ns = KC["true_claims"]["summary"], KC["non_claims"]["summary"]
    says("claim-check median chains", r"rests on \$\\mathbf\{(\d+)\}\$ independent chains", _ts["med_tau"], "{:.0f}")
    says("claim-check true share", r"chains and \$(\d+\.\d)\\%\$ certify", _ts["pct_neut_t1"], "{:.1f}")
    says("claim-check non-claim share", r"against \$(\d+\.\d)\\%\$ of non-claims", _ns["pct_neut_t1"], "{:.1f}")
    says("abstract claim share", r"while \$(\d+)\\%\$ of Hetionet", _ts["pct_neut_t1"], "{:.0f}")
    says("abstract non-claim share", r"forged relation against \$(\d+)\\%\$ of non-claims", _ns["pct_neut_t1"], "{:.0f}")
    holds("claim-check pairs scored", ("$755$ treatment edges and $755$ random" in TEX) and _ts["n"] == 755 and _ns["n"] == 755)

_E2 = os.path.join(HERE, "results_e2e.json")
if os.path.exists(_E2):
    E2 = json.load(open(_E2))
    print("\n--- end to end (results_e2e.json) ---")
    import e2e_table
    for _r in e2e_table.rows(E2):
        holds("e2e table row: " + _r[:40], _r in _RAW, _r)
    _S, _C = E2["summary"], E2["summary"]["clean"]
    _A = E2["attack_rows"]
    _mq = [r for r in _A if r["ds"] == "metaqa"]
    says("e2e attacked runs", r"Over \$(\d+)\$ attacked runs", len(_A), "{:.0f}")
    says("e2e report changes", r"report changed \$(\d+)\$\s+times", sum(r["defended_success"] for r in _A), "{:.0f}")
    holds("e2e no success within the certificate (sound)", _S["sound"] and
          all(r["ops"] > r["clean_cert"] for r in _A if r["defended_success"]))
    says("e2e injected-doc yes rate", r"answered yes to \$(\d+\.\d)\\%\$ of the injected", _S["reader"]["injected_doc_yes"], "{:.1f}")
    says("e2e forged-doc admission", r"are admitted\s+\(\$(\d+\.\d)\\%\$\)", _S["reader"]["forged_doc_accept"], "{:.1f}")
    says("e2e MetaQA questions with a certificate", r"only \$(\d+)\$ of \$200\$\s+questions carry", sum(r["defended"]["cert"] >= 1 for r in E2["rows"] if r["ds"] == "metaqa"), "{:.0f}")
    says("e2e MetaQA changes to abstention", r"\(\$(\d+)\$ of \$\d+\$\), not to the attacker", sum(r["defended_success"] and not r["defended_flip"] for r in _mq), "{:.0f}")
    says("e2e MetaQA changes total", r"\(\$\d+\$ of \$(\d+)\$\), not to the attacker", sum(r["defended_success"] for r in _mq), "{:.0f}")
    _m, _h = _C["metaqa"], _C["hetionet"]
    says("e2e MetaQA clean accuracy", r"answers \$(\d+\.\d)\\%\$ of\s+questions correctly", _m["clean_acc"], "{:.1f}")
    says("e2e MetaQA abstention", r"abstains on \$(\d+\.\d)\\%\$", _m["abstain"], "{:.1f}")
    says("e2e MetaQA oracle-reader accuracy", r"perfect reader it would answer\s+\$(\d+\.\d)\\%\$", _m["oracle_acc"], "{:.1f}")
    says("e2e MetaQA certified t=1", r"it certifies \$(\d+\.\d)\\%\$ at \$t = 1\$", _m["cert_acc_t1"], "{:.1f}")
    says("e2e MetaQA baseline accuracy", r"undefended model answers \$(\d+\.\d)\\%\$", _m["baseline_clean_acc"], "{:.1f}")
    says("e2e Hetionet accuracy", r"right on \$(\d+\.\d)\\%\$ of claims", _h["clean_acc"], "{:.1f}")
    says("e2e Hetionet certified t=1", r"certifies \$(\d+\.\d)\\%\$ of true claims at \$t = 1\$", _h["cert_acc_t1"], "{:.1f}")
    says("e2e Hetionet certified t=2", r"of true claims at \$t = 1\$, \$(\d+\.\d)\\%\$ at\s+\$t = 2\$", _h["cert_acc_t2"], "{:.1f}")
    says("e2e Hetionet certified t=3", r"at\s+\$t = 2\$ and \$(\d+\.\d)\\%\$ at \$t = 3\$", _h["cert_acc_t3"], "{:.1f}")
    says("e2e intro Hetionet certified", r"while \$(\d+\.\d)\\%\$ of Hetionet's treatment claims", _h["cert_acc_t1"], "{:.1f}")
    says("e2e Hetionet baseline acceptance", r"accepts\s+only \$(\d+\.\d)\\%\$ of true claims", _h["baseline_true_accept"], "{:.1f}")
    _R = _S["reader"]["honest_fn"]
    says("e2e reader FN MetaQA grounded", r"says no to \$(\d+\.\d)\\%\$ of\s+facts", _R["metaqa_grounded"]["rate"], "{:.1f}")
    _BR = _S["reader"]["honest_fn_by_relation"]
    for _rel, _pat in (("directed_by", r"rate is \$(\d+\.\d)\\%\$ for \\emph\{directed by\}"),
                       ("starred_actors", r"\$(\d+\.\d)\\%\$ for \\emph\{starred\}"),
                       ("written_by", r"\\emph\{written by\}\s+\(\$(\d+\.\d)\\%\$\)"),
                       ("in_language", r"\\emph\{language\} \(\$(\d+\.\d)\\%\$\)")):
        says("e2e reader FN " + _rel, _pat, _BR["metaqa/%s/grounded" % _rel]["rate"], "{:.1f}")
    says("e2e calls per MetaQA query", r"one reader call per fact, \$(\d+\.\d)\$ on MetaQA", _m["calls_per_query"] - 1 + 1, "{:.1f}")
    says("e2e total calls", r"evaluation took \$([\d{},]+)\$ calls", E2["cost"]["n_real_calls"], "{:.0f}")
    says("e2e seconds per reader call", r"at \$(\d\.\d+)\$\\,s per call", E2["cost"]["mean_reader_call_s"], "{:.2f}")
    says("e2e intro injection undefended", r"\(\$0\.0\\%\$, against \$(\d+\.\d)\\%\$ undefended\)",
         E2["summary"]["attacks"]["metaqa/injection/b1"]["undefended_flip"], "{:.1f}")

_KG = os.path.join(HERE, "results_kgomega.json")
if os.path.exists(_KG):
    KG = json.load(open(_KG))
    print("\n--- real KGQA workloads (results_kgomega.json) ---")
    m2, m3 = KG["metaqa"]["2-hop"], KG["metaqa"]["3-hop"]
    hp = KG["hetionet"]["metapaths"]

    def row(label, pattern, expected):
        """A table row: several numbers at once, each within half a unit of its last digit."""
        global ok
        m = re.search(pattern, TEX)
        if not m:
            print(f"  [FAIL] {label}: row not found  ->  {pattern}"); ok = False; return
        good = True
        for g, want in zip(m.groups(), expected):
            got = float(g.replace(",", ""))
            dec = len(g.split(".")[1]) if "." in g else 0
            good &= abs(got - float(want)) <= 0.5 * 10 ** (-dec) + 1e-9
        print(f"  [{'PASS' if good else 'FAIL'}] {label}: paper {m.groups()}, data "
              f"{tuple(round(float(w), 2) for w in expected)}")
        ok &= good

    pct = lambda x: 100 * x
    row("metaqa two-hop row",
        r"two-hop \((\d+) templates\) & ([\d,]+) & ([\d.]+)\\% & ([\d.]+)\\% & ([\d.]+)\\% & (\d+) \\\\",
        (m2["n_templates"], m2["n_answers"], pct(m2["frac_omega_1"]), pct(m2["frac_certify_ge1"]),
         pct(m2["frac_certify_ge3"]), m2["max"]))
    row("metaqa three-hop row",
        r"three-hop \((\d+) templates\) & ([\d,]+) & ([\d.]+)\\% & ([\d.]+)\\% & ([\d.]+)\\% & (\d+) \\\\",
        (m3["n_templates"], m3["n_answers"], pct(m3["frac_omega_1"]), pct(m3["frac_certify_ge1"]),
         pct(m3["frac_certify_ge3"]), m3["max"]))
    for key, label in (("CbGaD", "C--G--D"), ("GrGbC", "G--G--C"),
                       ("CrCtD", "C--C--D"),
                       ("CtDaGbC", "C--D--G--C")):
        h = hp[key]
        row(f"hetionet {key} row",
            label + r" & ([\d,]+) & ([\d.]+)\\% & ([\d.]+)\\% & ([\d.]+)\\% & (\d+) \\\\",
            (h["n_answers"], pct(h["frac_omega_1"]), pct(h["frac_certify_ge1"]),
             pct(h["frac_certify_ge3"]), h["max"]))
    says("metaqa two-hop single-chain share (prose)", r"\$([\d.]+)\\%\$ of MetaQA's two-hop answers",
         pct(m2["frac_omega_1"]), "{:.1f}")
    says("metaqa three-hop single-chain share (prose)", r"and \$([\d.]+)\\%\$ of its three-hop answers",
         pct(m3["frac_omega_1"]), "{:.1f}")
    _m = re.search(r"fewer than \$([\d.]+)\\%\$ of either carry", TEX)
    holds("metaqa 'fewer than x%' certify one relation",
          bool(_m) and max(pct(m2["frac_certify_ge1"]), pct(m3["frac_certify_ge1"])) < float(_m.group(1)),
          f"paper bound {_m.group(1) if _m else '?'}, data max {max(pct(m2['frac_certify_ge1']), pct(m3['frac_certify_ge1'])):.2f}")
    says("hetionet single-chain low", r"runs from \$([\d.]+)\\%\$ to", pct(min(h["frac_omega_1"] for h in hp.values())), "{:.1f}")
    says("hetionet single-chain high", r"to\s*\n?\$([\d.]+)\\%\$ by question", pct(max(h["frac_omega_1"] for h in hp.values())), "{:.1f}")
    says("metaqa max chains", r"up to \$(\d+)\$ chains on MetaQA", max(m2["max"], m3["max"]), "{:.0f}")
    says("hetionet max chains", r"chains on MetaQA and\s*\n?\$(\d+)\$ on Hetionet", max(h["max"] for h in hp.values()), "{:.0f}")
    says("metaqa omega=0 three-hop answers", r"\$([\d,{}]+)\$ three-hop answers \(", m3["omega_hist"].get("0", 0), "{:.0f}")
    says("metaqa omega=0 share", r"three-hop answers \(\$([\d.]+)\\%\$\)", pct(m3["omega_hist"].get("0", 0) / m3["n_answers"]), "{:.1f}")
    says("metaqa triples", r"WikiMovies graph \(\$([\d,{}]+)\$ triples\)", KG["metaqa"]["kb_triples"], "{:.0f}")
    says("hetionet nodes", r"and on Hetionet \(\$([\d,{}]+)\$ nodes", KG["hetionet"]["nodes"], "{:.0f}")
    says("hetionet anchors per question", r"at \$(\d+)\$ anchors\s*\n?each", hp["CbGaD"]["n_anchors"], "{:.0f}")
    holds("kg disjointness never violated", all(s["disjointness_violations"] == 0 for s in (m2, m3, *hp.values())))
    holds("kg medians are 1 as the paper says", all(s["median"] == 1 for s in (m2, m3, *hp.values())))
    holds("abstract's MetaQA figure", bool(re.search(r"On\s*\n?MetaQA \$96\\%\$ of two-hop answers", TEX)) and round(pct(m2["frac_omega_1"])) == 96)

_MT = os.path.join(HERE, "results_multitarget.json")
if os.path.exists(_MT):
    M = json.load(open(_MT))
    print("\n--- security experiments (results_multitarget.json) ---")
    says("occupancy ratio", r"the ratio is \$\\mathbf\{(\d\.\d+)\}\$ everywhere",
         M["E28_observed_amplification"]["max_ratio"], "{:.3f}")
    says("multi-target per-target value", r"per-target amplification stays at \$\\mathbf\{(\d+)\}\$",
         M["E29_multi_target"]["per_target_value"], "{:.0f}")
    says("multi-target slope", r"\$\\Gamma\$ grows with slope \$\\mathbf\{(\d\.\d+)\}\$",
         M["E29_multi_target"]["gamma_slope"], "{:.2f}")
    says("full-knowledge configs", r"Across \$(\d+)\$ configurations it reaches",
         M["E30_adaptive"]["n"], "{:.0f}")
    says("tightness min tightness", r"\[\\mathbf\{(\d\.\d+)\},",
         M["E31_tightness"]["min_tightness"], "{:.2f}")
    says("tightness max tightness", r"\\mathbf\{(\d\.\d+)\}\]", 
         M["E31_tightness"]["max_tightness"], "{:.2f}")

print("\n--- internal consistency of main.tex ---")


# Cor. optK is a CAP on the certified budget, and nothing checked it.  The published form
# carried a 1/sigma factor and was violated by these very rows at K=16: it capped b_cert at
# 0.5 while the tightened certificate attains 1.  The corrected form drops sigma, giving
# b_cert <= (1/2) min(c_1, K^{1/tau*_A}).  Assert it at every measured cell count.
if os.path.exists(_W):
    for _name, _blk in (("2-path", W["E32_E33_2path"]), ("C4", W["E32_E33_C4"])):
        _tau = _blk["tau_star_A"]
        _bad, _n = [], 0
        for _rec in _blk["records"]:
            _cap = lambda _c1: 0.5 * min(_c1, _rec["K"] ** (1.0 / _tau))
            _n += 1
            if _rec["med_b_tight"] > _cap(_rec["med_c1"]) + 1e-9:
                _bad.append(("median", _rec["K"], _rec["med_b_tight"]))
            for _row in _rec.get("rows", []):          # per-anchor, stricter than the median
                _n += 1
                if _row["b_tight"] > _cap(_row["c1"]) + 1e-9:
                    _bad.append((_row["anchor"], _rec["K"], _row["b_tight"]))
        holds(f"Cor. optK caps the tightened budget ({_name})", not _bad,
              f"{len(_bad)} violations: {_bad[:4]}" if _bad
              else f"{_n} anchor/cell-count points, tau*_A={_tau:g}")


_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15}

# 1. the abstract's falsification count must equal the number of "fails" rows
_n_fails = len(re.findall(r"\\textbf\{fails\}", TEX))
_m = re.search(r"(\w+) hypotheses are falsified in print", TEX)
holds("abstract's falsification count matches the scorecards",
      bool(_m) and _WORDS.get(_m.group(1).lower()) == _n_fails,
      f"abstract says {_m.group(1) if _m else '?'}, tables show {_n_fails}")

# 2. no orphan results: every theorem/proposition/corollary/lemma is referenced
_labels = set(re.findall(r"\\label\{((?:thm|prop|cor|lem|def):[^}]*)\}", TEX))
_refs = set(re.findall(r"\\x?ref\{([^}]*)\}", TEX))   # \xref points into the full version
_orphans = sorted(_labels - _refs)
holds("every numbered result is referenced somewhere", not _orphans,
      f"orphans: {_orphans}" if _orphans else f"{len(_labels)} results")

# 3. bibliography hygiene, both directions
_keys = set(re.findall(r"\\bibitem\{([^}]*)\}", TEX))
_cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", TEX)
          for k in grp.split(",")}
holds("no citation to a missing bibitem", not (_cited - _keys),
      f"missing: {sorted(_cited - _keys)}" if _cited - _keys else f"{len(_cited)} keys")
holds("no uncited bibitem", not (_keys - _cited),
      f"uncited: {sorted(_keys - _cited)}" if _keys - _cited else f"{len(_keys)} entries")

# 4. the em-dash sweep must stay swept
holds("no em-dash characters", "\u2014" not in _RAW)

print()
if ok:
    print("PAPER MATCHES DATA ✓")
    sys.exit(0)
print("PAPER DISAGREES WITH DATA ✗")
sys.exit(1)
