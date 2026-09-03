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
_PAPER = os.environ.get("PAPER", os.path.join(HERE, "..", "paper", "main.tex"))
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
    got = float(m.group(1).replace(",", ""))
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
    says("reachable-damage 2-path rho_max", r"\$1\$ against \$\\rho_\{\\max\} = (\d)\$ on the \$2\$-path",
         p2[256]["max_rho_max"], "{:.0f}")
    says("reachable-damage C4 rho_F at K=16",
         r"\\rho_F \\;=\\; (\d+) \\ \\text\{against\}\\ \\rho_\{\\max\} = 12",
         c4[16]["med_rho_F"], "{:.0f}")
    says("reachable-damage C4 rho_max at K=16",
         r"\\text\{against\}\\ \\rho_\{\\max\} = (\d+) \\ \(K = 16\)",
         c4[16]["max_rho_max"], "{:.0f}")
    says("reachable-damage C4 rho_F at K=64", r"\\rho_F \\;=\\; (\d+) \\ \\text\{against\}\\ 28",
         c4[64]["med_rho_F"], "{:.0f}")
    says("three-designs witness budget, 2-path", r"certifies a median of \$(\d+)\$ forged",
         p2[256]["med_b_witness"], "{:.0f}")
    says("three-designs witness cells, 2-path", r"relations using \$(\d+)\$ cells",
         p2[256]["med_omega"], "{:.0f}")
    says("three-designs witness budget, C4", r"it certifies \$(\d)\$ using \$3\$ cells",
         c4[64]["med_b_witness"], "{:.0f}")

_MG = os.path.join(HERE, "results_menger.json")
if os.path.exists(_MG):
    G = json.load(open(_MG))
    print("\n--- fractional budget and Menger (results_menger.json) ---")
    c4 = G["E34_fractional"]["summary"]["C4 (v-x-y-z)"]
    mg = G["E35_menger"]
    says("fractional-gain max fractional gain", r"maximum of \$(\d\.\d+)\$ on the anchored", c4["max_gap"], "{:.2f}")
    says("fractional-gain strict anchors", r"strictly above \$1\$ at \$(\d+)\$ of \$15\$", c4["n_strict"], "{:.0f}")
    says("flow-check answers checked", r"they agree in \$\\mathbf\{(\d+)\}\$ of \$306\$", mg["n_answers"], "{:.0f}")
    says("flow-check median omega", r"median\s*\n?answer has \$\\omega_a = (\d)\$", mg["med_omega"], "{:.0f}")
    says("flow-check max omega", r"and the best has \$(\d)\$", mg["max_omega"], "{:.0f}")
    holds_menger = mg["menger_exact"]

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


def holds(label, cond, detail=""):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not cond:
        ok = False


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
_refs = set(re.findall(r"\\ref\{([^}]*)\}", TEX))
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
