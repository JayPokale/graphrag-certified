"""
attack.py -- the attack calculus: surface, success probability, required budget,
and five IMPLEMENTED adversaries against the cell-partitioned GraphRAG defense.

Everything here is about ONE thing: an adversary who forges edges in the retrieved
subgraph, and what it costs to stop them.

Quantities defined and measured
-------------------------------
A1  per-poison attack surface        A(e)      = |cells(e)|            = rho(e)
A2  aimed surface                    A(e | F)  = |cells(e) & F|        (F = firing cells)
A3  total corpus attack surface      S         = sum_e rho(e) = K * Bbar
                                                 ("surface = verdict compute")
A4  damage rate (blind)              E[hits per poison] = rho|F|/K
A5  white-box required budget        m_s* = ceil(mu / rho_max)
A6  blind required budget            m_s* = ln(c1 / floor(K/2)) / (-ln(1 - rho/K))
A7  success probability              P_succ(m_d, m_s) = P[#survivors <= floor(K/2)]
A8  attacker mix threshold           all-structural is optimal iff cost_s/cost_d < rho
A9  slice penalty (private HyperCube) Var inflation vs uniform rho-subsets

Adversaries implemented (not assumed)
-------------------------------------
ADV-WB     white-box greedy: sees the hash, aims at firing cells             (optimal)
ADV-BLIND  blind: uniform random contents against a private hash
ADV-ADAPT  adaptive: inject, read the 1-bit verdict, keep or retract
ADV-SKEW   degree-targeted: forges edges at the highest-replication positions
ADV-SLICE  slice-aware: exploits HyperCube geometry (reachable cell-sets are
           coordinate slices, not arbitrary rho-subsets)

Run:  python3 attack.py            (writes results_attack.json)
GPU:  not used. The workload is LP + combinatorics + Monte Carlo over integer
      sets; it is CPU- and memory-bound. See ATTACK_CALCULUS.md for the one place
      GPUs would matter (the LLM-verdict harness, which is not run here).
"""
from __future__ import annotations
import itertools, json, math, os, random
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260731
TRIALS = 1500


# ===========================================================================
# LP core (shared with packing.py; duplicated so this file runs standalone)
from lp import tau_star, share_lp



CYCLE = lambda k: [("R", ["v" if i == 0 else f"u{i}",
                          "v" if (i + 1) % k == 0 else f"u{(i+1)%k}"]) for i in range(k)]


# ===========================================================================
# Closed-form attack calculus
# ===========================================================================
def survival_prob(K, rho, m_d, m_s):
    """q = P[a given firing cell survives the whole attack], EXACT per cell.
    A document poison lands in 1 uniform cell; a structural poison lands in a
    uniform rho-subset, which contains a fixed cell with probability exactly
    rho/K."""
    return (1.0 - 1.0 / K) ** m_d * (1.0 - rho / K) ** m_s


def p_succ_meanfield(K, c1, rho, m_d, m_s):
    """P[verdict flips] under the mean-field (independent-cell) model:
    survivors ~ Bin(c1, q), flip iff survivors <= floor(K/2)."""
    from scipy.stats import binom
    return float(binom.cdf(K // 2, c1, survival_prob(K, rho, m_d, m_s)))


def p_succ_bernstein(K, c1, rho, m_d, m_s):
    """Bernstein/Bennett upper bound.  Survivors are NA Bernoulli(q), so
    Var[X] <= c1 q(1-q) -- strictly smaller than Hoeffding's implicit c1/4
    whenever q is away from 1/2, which is exactly the certified regime."""
    q = survival_prob(K, rho, m_d, m_s)
    t = c1 * q - K // 2
    if t <= 0:
        return 1.0
    var = c1 * q * (1.0 - q)
    return float(math.exp(-t * t / (2.0 * var + 2.0 * t / 3.0)))


def p_succ_certified(K, c1, rho, m_d, m_s):
    """The bound a defender should actually quote: Hoeffding and Bernstein are
    both valid upper bounds for negatively associated summands, so their MINIMUM
    is valid.  Neither dominates: Bernstein wins iff q(1-q) < 1/4 - t/(3 c1),
    which fails when q is close to 1/2 (measured: q in [0.51, 0.70], ~4% of the
    certified regime)."""
    return min(p_succ_bernstein(K, c1, rho, m_d, m_s),
               p_succ_chernoff(K, c1, rho, m_d, m_s))


def certified_budget(K, c1, rho, delta, m_d=0, bound="min"):
    """Largest m_s certifiable at confidence 1-delta."""
    f = {"bernstein": p_succ_bernstein, "hoeffding": p_succ_chernoff,
         "min": p_succ_certified}[bound]
    best = -1
    for m in range(0, 40 * K):
        if f(K, c1, rho, m_d, m) <= delta:
            best = m
        else:
            break
    return best


def p_succ_chernoff(K, c1, rho, m_d, m_s):
    """Hoeffding upper bound on the attacker's success probability."""
    q = survival_prob(K, rho, m_d, m_s)
    thr = K // 2
    if c1 * q <= thr:
        return 1.0
    return float(math.exp(-2.0 * (c1 * q - thr) ** 2 / c1))


_BB = {}


def blind_budget(K, c1, rho, delta=0.5, m_d=0):
    """A6: smallest m_s with mean-field success probability >= delta (bisection)."""
    key = (K, c1, rho, delta, m_d)
    if key in _BB:
        return _BB[key]
    hi = 1
    while p_succ_meanfield(K, c1, rho, m_d, hi) < delta and hi < 1 << 20:
        hi *= 2
    lo = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if p_succ_meanfield(K, c1, rho, m_d, mid) >= delta:
            hi = mid
        else:
            lo = mid + 1
    _BB[key] = lo
    return lo


def whitebox_budget(mu, rho_max, m_d=0):
    """A5: exact, deterministic (Theorem: certificate is tight to the integer)."""
    return int(math.ceil(max(0, mu - m_d) / rho_max))


# ===========================================================================
# The defense: anchored HyperCube over a host graph
# ===========================================================================
def build_hypercube(atoms, host_edges, K, rng, bound=frozenset({"v"}), private=True):
    """Returns (place: edge -> frozenset(cells), Kreal, shares, slices)
    `slices[u][val]` is the coordinate a value hashes to (the attacker's target
    handle under a public hash)."""
    _, sh = share_lp(atoms, bound)
    U = sorted(sh)
    p = {u: max(1, int(round(K ** sh[u]))) for u in U}
    Kreal = 1
    for u in U:
        Kreal *= p[u]
    coords = list(itertools.product(*[range(p[u]) for u in U]))
    ui = {u: i for i, u in enumerate(U)}
    h = {u: {} for u in U}

    def hv(u, x):
        if x not in h[u]:
            h[u][x] = rng.randrange(p[u])
        return h[u][x]

    place = {}
    for (a, b) in host_edges:
        cells = set()
        for _, vs in atoms:
            for (xa, xb) in ((a, b),):
                fixed, ok = {}, True
                for var, val in zip(vs, (xa, xb)):
                    if var in bound:
                        continue
                    fixed[ui[var]] = hv(var, val)
                if ok:
                    cells |= {c for c in coords
                              if all(c[i] == q for i, q in fixed.items())}
        place[(a, b)] = frozenset(cells)
    return place, Kreal, p, coords, h, ui, U


def random_slice(coords, p, U, ui, rng):
    """A cell-set reachable by ONE forged edge under the HyperCube: fix the
    coordinates of the variables the edge binds, leave the rest free."""
    fixed_vars = list(U)[:2] if len(U) >= 2 else list(U)
    fixed = {ui[u]: rng.randrange(p[u]) for u in fixed_vars}
    return frozenset(c for c in coords if all(c[i] == q for i, q in fixed.items()))


# ===========================================================================
# Adversaries.  Each returns the number of structural poisons needed to flip.
# ===========================================================================
def adv_whitebox(F, reachable, mu, rng, cap=10 ** 5):
    """Sees the hash; greedily picks the reachable cell-set covering the most
    still-firing cells."""
    alive = set(F)
    n = 0
    while len(F) - len(alive) < mu and n < cap:
        best = max(reachable, key=lambda S: len(S & alive))
        gain = len(best & alive)
        if gain == 0:
            return None
        alive -= best
        n += 1
    return n


def adv_blind(F, K, rho, mu, rng, cap=10 ** 5):
    """Private hash: each poison lands in a uniform rho-subset."""
    alive = set(F)
    cells = list(range(K))
    n = 0
    while len(F) - len(alive) < mu and n < cap:
        alive -= set(rng.sample(cells, rho))
        n += 1
    return n


def adv_slice(F, reachable_shapes, mu, rng, cap=10 ** 5):
    """Private HyperCube: the poison lands in a uniform random SLICE, not a
    uniform rho-subset.  Same marginal (rho/K per cell) but correlated."""
    alive = set(F)
    n = 0
    while len(F) - len(alive) < mu and n < cap:
        alive -= rng.choice(reachable_shapes)
        n += 1
    return n


def adv_adaptive(F, reachable_shapes, mu, rng, probes_per_round=1, cap=10 ** 5):
    """Injects, reads the 1-bit verdict (did the firing count drop?), retracts on
    no-op.  Under a private hash the verdict reveals only whether THIS content
    hit; it cannot be used to aim the next one."""
    alive = set(F)
    n = 0
    while len(F) - len(alive) < mu and n < cap:
        S = rng.choice(reachable_shapes)
        hit = len(S & alive)
        n += 1                     # the probe costs an injection
        if hit:
            alive -= S             # keep it
        # miss -> retract; the bit told us nothing about unqueried contents
    return n


def adv_skew(F, place, mu, rng, cap=10 ** 5):
    """Degree-targeted: forge at the positions with the LARGEST replication,
    i.e. exploit load skew.  This is the attacker RD2 says exists on real graphs."""
    ranked = sorted(place.values(), key=lambda S: -len(S))
    alive = set(F)
    n = 0
    for S in ranked:
        if len(F) - len(alive) >= mu or n >= cap:
            break
        if S & alive:
            alive -= S
            n += 1
    return n if len(F) - len(alive) >= mu else None


# ===========================================================================
# X1: closed-form calculus vs Monte Carlo (does the formula hold?)
# ===========================================================================
def X1(Ks=(64, 256), rhos=None, trials=TRIALS):
    rng = random.Random(SEED)
    out = []
    for K in Ks:
        for tau in (2, 3):
            rho = max(1, int(round(K ** (1 - 1.0 / tau))))
            c1 = int(0.75 * K)
            mu = c1 - K // 2
            for m_s in range(0, 4 * blind_budget(K, c1, rho, 0.5) + 1,
                             max(1, blind_budget(K, c1, rho, 0.5) // 3)):
                nprng = np.random.default_rng(SEED + K + tau + m_s)
                alive = np.ones((trials, c1), dtype=bool)
                for _ in range(m_s):
                    keys = nprng.random((trials, K))
                    sel = np.argpartition(keys, rho - 1, axis=1)[:, :rho]
                    ok = (sel < c1).ravel()
                    rows_ = np.repeat(np.arange(trials), rho)[ok]
                    alive[rows_, sel.ravel()[ok]] = False
                emp = int((alive.sum(axis=1) <= K // 2).sum())
                out.append(dict(K=K, tau=tau, rho=rho, c1=c1, mu=mu, m_s=m_s,
                                p_meanfield=p_succ_meanfield(K, c1, rho, 0, m_s),
                                p_chernoff=p_succ_chernoff(K, c1, rho, 0, m_s),
                                p_measured=emp / trials))
    err = [abs(r["p_meanfield"] - r["p_measured"]) for r in out]
    viol = [r for r in out if r["p_measured"] > r["p_chernoff"] + 0.02]
    return dict(max_abs_error_meanfield=float(max(err)),
                mean_abs_error_meanfield=float(np.mean(err)),
                chernoff_violations=len(viol), n_points=len(out), rows=out)


# ===========================================================================
# X2: budget separation -- white-box vs blind vs adaptive vs slice
# ===========================================================================
def X2(Ks=(64, 256, 1024), taus=(2, 3, 4), trials=120):
    rng = random.Random(SEED + 1)
    rows = []
    for K in Ks:
        for tau in taus:
            rho = max(1, int(round(K ** (1 - 1.0 / tau))))
            c1 = int(0.75 * K); mu = c1 - K // 2
            F = set(range(c1))
            wb = whitebox_budget(mu, rho)
            bl = [adv_blind(F, K, rho, mu, rng) for _ in range(trials // 4)]
            shapes = [frozenset(rng.sample(range(K), rho)) for _ in range(256)]
            ad = [adv_adaptive(F, shapes, mu, rng) for _ in range(trials // 4)]
            rows.append(dict(K=K, tau=tau, rho=rho, mu=mu,
                             wb_budget=wb,
                             blind_budget_measured=float(np.mean(bl)),
                             blind_budget_formula=blind_budget(K, c1, rho, 0.5),
                             adaptive_budget=float(np.mean(ad)),
                             separation=float(np.mean(bl)) / max(wb, 1),
                             predicted_separation=K / rho / max(wb, 1)))
    return dict(rows=rows,
                min_separation=float(min(r["separation"] for r in rows)),
                adaptive_over_blind=float(np.mean(
                    [r["adaptive_budget"] / r["blind_budget_measured"] for r in rows])))


# ===========================================================================
# X3: the slice penalty -- private HyperCube vs idealised uniform rho-subsets
#     (this is the gap the private-hash theorem left open)
# ===========================================================================
def X3(Ks=(64, 256, 1024), taus=(2, 3), trials=4000):
    """The private-HyperCube question, correctly posed.

    A blinded poison lands in a uniform rho-SUBSET under the idealised model and in
    a uniform coordinate SLICE under the deployed design.  Both have the same mean.
    The variance depends on how the firing set F sits relative to the hash
    coordinates, so ALIGNMENT is the variable:

      generic   F drawn uniformly at random  -- what a private hash produces, since
                the adversary cannot align F with a hash it cannot see
      aligned   F a union of slices          -- the worst case, reachable only if the
                hash is public or leaked

    Exact reference: for a uniform rho-subset, |S & F| ~ Hypergeometric(K, c1, rho),
    so Var = rho (c1/K)(1 - c1/K)(K - rho)/(K - 1).
    """
    rng = random.Random(SEED + 2)
    rows = []
    for K in Ks:
        for tau in taus:
            side = int(round(K ** (1.0 / tau)))
            Kr = side ** tau
            rho = Kr // side
            c1 = int(0.75 * Kr)
            mu = c1 - Kr // 2
            coords = list(itertools.product(*[range(side)] * tau))
            cid = {c: i for i, c in enumerate(coords)}
            shapes = [frozenset(cid[c] for c in coords if c[dim] == val)
                      for dim in range(tau) for val in range(side)]
            hyper_var = rho * (c1 / Kr) * (1 - c1 / Kr) * (Kr - rho) / (Kr - 1)
            for label, F in (("generic", set(rng.sample(range(Kr), c1))),
                             ("aligned", set(range(c1)))):
                hs = [len(rng.choice(shapes) & F) for _ in range(trials)]
                hu = [len(set(rng.sample(range(Kr), rho)) & F) for _ in range(trials)]
                bs = [adv_slice(F, shapes, mu, rng) for _ in range(200)]
                bu = [adv_blind(F, Kr, rho, mu, rng) for _ in range(200)]
                rows.append(dict(
                    K=Kr, tau=tau, side=side, rho=rho, c1=c1, mu=mu,
                    alignment=label,
                    mean_hits_slice=float(np.mean(hs)),
                    mean_hits_uniform=float(np.mean(hu)),
                    predicted_mean=rho * c1 / Kr,
                    var_slice=float(np.var(hs)), var_uniform=float(np.var(hu)),
                    var_hypergeometric=float(hyper_var),
                    uniform_matches_hypergeom=bool(
                        abs(np.var(hu) - hyper_var) <= 0.15 * hyper_var),
                    var_inflation=float(np.var(hs) / max(np.var(hu), 1e-9)),
                    worst_case_inflation_bound=float(rho),   # F = union of slices: Var_slice ~ (rho/tau) Var_unif
                    budget_slice=float(np.mean(bs)), budget_uniform=float(np.mean(bu)),
                    budget_ratio=float(np.mean(bs) / np.mean(bu))))
    gen = [r for r in rows if r["alignment"] == "generic"]
    ali = [r for r in rows if r["alignment"] == "aligned"]
    return dict(
        rows=rows,
        generic_max_inflation=float(max(r["var_inflation"] for r in gen)),
        generic_budget_ratio_range=[float(min(r["budget_ratio"] for r in gen)),
                                    float(max(r["budget_ratio"] for r in gen))],
        aligned_max_inflation=float(max(r["var_inflation"] for r in ali)),
        aligned_within_bound=all(r["var_inflation"] <= r["worst_case_inflation_bound"]
                                 for r in ali),
        uniform_matches_hypergeom=all(r["uniform_matches_hypergeom"] for r in rows),
        max_mean_gap=float(max(abs(r["mean_hits_slice"] - r["predicted_mean"])
                               for r in rows)))


# ===========================================================================
# X4: attack surface identity  S = sum_e rho(e) = K * Bbar, on a real graph
# ===========================================================================
def X4(net_name="case2869pegase", Ks=(16, 64), n_anchors=16):
    rng = random.Random(SEED + 3)
    import pandapower.networks as pn
    net = getattr(pn, net_name)()
    E = set()
    for _, r in net.line.iterrows():
        a, b = int(r.from_bus), int(r.to_bus)
        if a != b:
            E.add((min(a, b), max(a, b)))
    for _, r in net.trafo.iterrows():
        a, b = int(r.hv_bus), int(r.lv_bus)
        if a != b:
            E.add((min(a, b), max(a, b)))
    adj = defaultdict(set)
    for a, b in E:
        adj[a].add(b); adj[b].add(a)

    atoms = [("R", ["v", "x"]), ("R", ["x", "y"]), ("R", ["y", "z"]), ("R", ["z", "v"])]
    tau = tau_star(atoms, frozenset({"v"}))
    anchors = sorted(adj, key=lambda u: -len(adj[u]))[:n_anchors]
    rows = []
    for K in Ks:
        recs = []
        for v in anchors:
            emb = []
            for x in adj[v]:
                for y in adj[x]:
                    if y == v:
                        continue
                    for z in adj[y]:
                        if z in (v, x):
                            continue
                        if v in adj[z]:
                            emb.append(((min(v, x), max(v, x)), (min(x, y), max(x, y)),
                                        (min(y, z), max(y, z)), (min(z, v), max(z, v))))
            if len(emb) < 4:
                continue
            edges = sorted({e for et in emb for e in et})
            place, Kreal, p, coords, h, ui, U = build_hypercube(
                atoms, edges, K, rng, frozenset({"v"}))
            rho = [len(place[e]) for e in edges]
            loads = defaultdict(int)
            for e in edges:
                for c in place[e]:
                    loads[c] += 1
            S = sum(rho)
            recs.append(dict(anchor=v, deg=len(adj[v]), N=len(edges),
                             surface=S, K=Kreal, B_bar=sum(loads.values()) / Kreal,
                             identity_gap=abs(S - sum(loads.values())),
                             rho_max=max(rho), rho_bar=float(np.mean(rho)),
                             surface_top1_share=max(rho) / S))
        rows.append(dict(K_requested=K, n_anchors=len(recs),
                         identity_max_gap=max(r["identity_gap"] for r in recs),
                         mean_surface=float(np.mean([r["surface"] for r in recs])),
                         mean_top1_share=float(np.mean([r["surface_top1_share"]
                                                        for r in recs])),
                         max_top1_share=max(r["surface_top1_share"] for r in recs),
                         per_anchor=recs[:6]))
    return dict(network=net_name, n_nodes=len(adj), n_edges=len(E),
                motif="C4", d=4, tau_star_v=tau, records=rows)


# ===========================================================================
# X5: attacker mix threshold -- when does forging edges beat forging documents?
# ===========================================================================
def X5(K=1024, taus=(2, 3, 4, 6), cost_ratios=(0.5, 2, 8, 32, 128)):
    rows = []
    c1 = int(0.75 * K); mu = c1 - K // 2
    for tau in taus:
        rho = max(1, int(round(K ** (1 - 1.0 / tau))))
        for cr in cost_ratios:          # cost_s / cost_d
            n_struct = math.ceil(mu / rho)
            cost_all_struct = cr * n_struct
            cost_all_doc = 1.0 * mu
            thr_exact = mu / n_struct          # exact threshold on cost_s/cost_d
            rows.append(dict(K=K, tau=tau, rho=rho, mu=mu, cost_ratio=cr,
                             n_structural=n_struct,
                             cost_structural=cost_all_struct,
                             cost_document=cost_all_doc,
                             structural_wins=bool(cost_all_struct < cost_all_doc),
                             exact_threshold=thr_exact,
                             asymptotic_threshold=float(rho)))
    ok = all(r["structural_wins"] == (r["cost_ratio"] < r["exact_threshold"])
             for r in rows)
    tight = all(r["exact_threshold"] <= r["rho"] + 1e-9 for r in rows)
    return dict(rows=rows, threshold_rule_holds=ok, exact_le_rho=tight,
                rule=("all-structural is optimal iff cost_s/cost_d < mu/ceil(mu/rho), "
                      "which is rho up to the integrality of the last poison"))


# ===========================================================================
# X2b: the ONSET/OR regime -- where the private hash actually buys a polynomial
# factor.  X2 measures the majority regime, where the theorem predicts <= 2x.
# ===========================================================================
def X2b(Ks=(64, 256, 1024, 4096), taus=(2, 3), trials=4000):
    rng = random.Random(SEED + 5)
    rows = []
    for K in Ks:
        for tau in taus:
            rho = max(1, int(round(K ** (1 - 1.0 / tau))))
            t = max(1, int(round(K * (rho / K) ** tau)))   # E[firing cells] at onset
            F = set(rng.sample(range(K), t))
            mu = 1                                         # OR-aggregation: silence all
            wb = math.ceil(t / rho)                        # white-box: aims at F
            bl = []
            for _ in range(trials):
                alive = set(F); n = 0
                while alive and n < 50 * K:
                    alive -= set(rng.sample(range(K), rho))
                    n += 1
                bl.append(n)
            sem = float(np.std(bl) / math.sqrt(len(bl)))
            rows.append(dict(K=K, tau=tau, rho=rho, t=t, wb_budget=wb,
                             blind_budget=float(np.mean(bl)), blind_sem=sem,
                             separation=float(np.mean(bl)) / max(wb, 1),
                             exact_mean_t1=(K / rho if t == 1 else None),
                             z_vs_exact=((float(np.mean(bl)) - K / rho) / sem
                                         if t == 1 and sem > 0 else None),
                             predicted_separation=(K / rho) / max(wb, 1)))
    lk = np.log([r["K"] for r in rows if r["tau"] == 2])
    ls = np.log([r["separation"] for r in rows if r["tau"] == 2])
    lk3 = np.log([r["K"] for r in rows if r["tau"] == 3])
    ls3 = np.log([r["separation"] for r in rows if r["tau"] == 3])
    return dict(rows=rows,
                fitted_exponent_tau2=float(np.polyfit(lk, ls, 1)[0]),
                predicted_exponent_tau2=0.5,
                fitted_exponent_tau3=float(np.polyfit(lk3, ls3, 1)[0]),
                predicted_exponent_tau3=1.0 / 3)


# ===========================================================================
# X6: Bernstein vs Hoeffding -- what the tighter tail is worth in certified budget
# ===========================================================================
def X6(Ks=(256, 1024, 4096), taus=(2, 3), deltas=(0.05, 0.01, 1e-3)):
    rows = []
    for K in Ks:
        for tau in taus:
            rho = max(1, int(round(K ** (1 - 1.0 / tau))))
            c1 = int(0.75 * K)
            for dl in deltas:
                h = certified_budget(K, c1, rho, dl, bound="hoeffding")
                b = certified_budget(K, c1, rho, dl, bound="bernstein")
                m = certified_budget(K, c1, rho, dl, bound="min")
                rows.append(dict(K=K, tau=tau, rho=rho, c1=c1, delta=dl,
                                 budget_hoeffding=h, budget_bernstein=b,
                                 budget_min=m, gain=(m - h),
                                 gain_pct=(100.0 * (m - h) / max(h, 1)),
                                 bernstein_alone_worse=bool(b < h)))
    return dict(rows=rows,
                min_never_worse=all(r["budget_min"] >= r["budget_hoeffding"] for r in rows),
                n_bernstein_alone_worse=sum(r["bernstein_alone_worse"] for r in rows),
                bernstein_never_worse=all(r["budget_bernstein"] >= r["budget_hoeffding"]
                                          for r in rows),
                mean_gain_pct=float(np.mean([r["gain_pct"] for r in rows])),
                max_gain_pct=float(max(r["gain_pct"] for r in rows)))


# ===========================================================================
# X7: the kappa-block certificate.  Verdict errors are NOT independent across
# cells when one payload is replicated into rho of them.  Under a block model
# (blocks of size <= kappa, independent across blocks, arbitrary within)
# Hoeffding applies to the block sums with sum_g |g|^2 <= kappa K, giving a
# sqrt(kappa) fee.  With only a variance budget and no block structure, the
# best available is Chebyshev, which costs delta^{-1/2} instead of sqrt(ln 1/delta).
# ===========================================================================
def X7(settings=((64, 1, 0.10), (64, 4, 0.10), (64, 16, 0.10),
                 (256, 8, 0.20), (256, 32, 0.20)), trials=40000, seed=SEED + 7):
    rs = np.random.default_rng(seed)
    rows = []
    for K, kappa, eps in settings:
        G = K // kappa
        blocks = rs.random((trials, G)) < eps
        X = blocks.repeat(kappa, axis=1).sum(axis=1)
        var, indep = float(X.var()), K * eps * (1 - eps)
        mean = float(X.mean())
        tail = []
        for c in (0.5, 1.0, 1.5):
            t = c * math.sqrt(kappa * K)
            emp = float((X <= mean - t).mean())
            hoef_block = math.exp(-2 * t * t / (kappa * K))
            cheb = min(1.0, kappa * K * eps * (1 - eps) / (t * t))
            tail.append(dict(t=t, empirical=emp, hoeffding_block=hoef_block,
                             chebyshev=cheb, holds=bool(emp <= hoef_block + 1e-9)))
        rows.append(dict(K=K, kappa=kappa, eps=eps, var=var, indep_var=indep,
                         var_ratio=var / indep, tail=tail))
    return dict(rows=rows,
                kappa_estimator_accurate=all(abs(r["var_ratio"] - r["kappa"])
                                             <= 0.15 * r["kappa"] for r in rows),
                block_hoeffding_never_violated=all(t["holds"] for r in rows
                                                   for t in r["tail"]),
                n_tail_points=sum(len(r["tail"]) for r in rows))



# ==========================================================================
# X8 (E25): kappa is not a free parameter.  If per-cell verdict errors are
# independent EXCEPT that cells sharing a forged edge fail together, then the
# kappa-groups of Theorem kappa are the connected components of the cell-overlap
# graph, so kappa <= min(K, rho*m_s).  In the certified regime rho*m_s < mu <= K,
# so kappa < mu automatically and the certificate needs no measurement of kappa
# for a sound (conservative) guarantee.
# ==========================================================================
def _components(K, sets):
    par = list(range(K))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for S in sets:
        S = list(S)
        for b in S[1:]:
            ra, rb = find(S[0]), find(b)
            if ra != rb:
                par[ra] = rb
    groups = {}
    for i in range(K):
        groups.setdefault(find(i), []).append(i)
    return max(len(g) for g in groups.values()), groups


def x8_kappa_is_bounded(seed=SEED):
    rng = random.Random(seed)

    # (a) max component <= min(K, rho m_s)
    rows, bad = [], 0
    for K, rho, ms in itertools.product([64, 256, 1024], [4, 16, 64], [1, 2, 4, 8]):
        if rho > K:
            continue
        mx = 0
        for _ in range(200):
            mx = max(mx, _components(K, [rng.sample(range(K), rho)
                                         for _ in range(ms)])[0])
        ub = min(K, rho * ms)
        bad += mx > ub
        rows.append(dict(K=K, rho=rho, m_s=ms, max_component=mx, bound=ub,
                         holds=bool(mx <= ub)))

    # (b) the certified regime forces kappa <= rho m_s < mu <= K
    viol = tot = 0
    for _ in range(2000):
        K = rng.choice([64, 256, 1024])
        rho = rng.choice([2, 4, 8, 16, 32])
        mu = rng.randrange(2, K + 1)
        md = rng.randrange(0, mu)
        ms_max = (mu - md - 1) // rho
        if ms_max < 1:
            continue
        ms = rng.randrange(1, ms_max + 1)
        tot += 1
        k = _components(K, [rng.sample(range(K), min(rho, K)) for _ in range(ms)])[0]
        if not (k <= min(K, rho * ms) and rho * ms < mu <= K):
            viol += 1

    # (c) block-Hoeffding at kappa = max component, errors perfectly correlated
    #     inside each component
    tails, worst = [], -1.0
    for K, rho, ms, eps in [(64, 8, 2, 0.10), (256, 16, 3, 0.15),
                            (256, 32, 4, 0.20), (1024, 32, 6, 0.10)]:
        kap, groups = _components(K, [rng.sample(range(K), rho) for _ in range(ms)])
        T = 40000
        cnt = np.empty(T)
        gs = list(groups.values())
        for t in range(T):
            e = 0
            for g in gs:
                if rng.random() < eps:
                    e += len(g)
            cnt[t] = e
        for dev in (0.5, 1.0, 1.5, 2.0, 2.5):
            t_ = dev * math.sqrt(kap * K)
            emp = float(np.mean(cnt - K * eps >= t_))
            bd = math.exp(-2 * t_ ** 2 / (kap * K))
            worst = max(worst, emp - bd)
            tails.append(dict(K=K, kappa=kap, dev=dev, empirical=emp, bound=bd,
                              ok=bool(emp <= bd + 1e-12)))
        tails[-1]["rho_ms"] = rho * ms

    return dict(rows=rows, component_bound_holds=bad == 0, n_settings=len(rows),
                certified_regime_draws=tot, certified_regime_violations=viol,
                tail_points=len(tails), max_excess=worst,
                block_hoeffding_never_violated=bool(worst <= 0),
                max_kappa_over_rho=max(r["max_component"] / r["rho"] for r in rows))



# ==========================================================================
# X9 (E26): end-to-end test of Corollary kappa-free.  Run the WHOLE pipeline --
# worst-case adversary silencing m_d + rho*m_s firing cells, then correlated
# verdict noise in which cells sharing a forged edge flip together -- and check
# the aggregate verdict fails less often than delta.  Includes a negative
# control below the certified threshold, without which the test is vacuous.
# ==========================================================================
def _corr_fail_rate(K, rho, mu, md, ms, eps, rng, T=4000):
    """Vectorised: report count = sum over groups of (flipped ? |g|-fire_g : fire_g)."""
    c1 = K // 2 + mu
    if c1 > K:
        return None
    firing = max(0, c1 - min(md + rho * ms, c1))
    sets = [rng.sample(range(K), min(rho, K)) for _ in range(ms)]
    par = list(range(K))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for S in sets:
        for b in S[1:]:
            ra, rb = find(S[0]), find(b)
            if ra != rb:
                par[ra] = rb
    groups = {}
    for i in range(K):
        groups.setdefault(find(i), []).append(i)
    sizes = np.array([len(g) for g in groups.values()])
    fire = np.array([sum(1 for i in g if i < firing) for g in groups.values()])
    gen = np.random.default_rng(rng.randrange(2 ** 32))
    flip = gen.random((T, len(sizes))) < eps
    counts = np.where(flip, sizes - fire, fire).sum(axis=1)
    return float(np.mean(counts <= K // 2))


def x9_end_to_end_kappa_free(seed=SEED):
    rng = random.Random(seed)
    certified, control = [], []
    for _ in range(200):
        K = rng.choice([64, 128, 256])
        rho = rng.choice([2, 4, 8])
        eps = rng.choice([0.05, 0.10, 0.20])
        delta = rng.choice([0.05, 0.01])
        ms = rng.randrange(1, 4)
        md = rng.randrange(0, 4)
        fee = math.sqrt(rho * ms * K * math.log(1 / delta) / 2) / (1 - 2 * eps)
        mu_star = md + rho * ms + fee
        mu = math.ceil(mu_star) + 1
        e = _corr_fail_rate(K, rho, mu, md, ms, eps, rng)
        if e is not None:
            certified.append(dict(K=K, rho=rho, mu=mu, m_d=md, m_s=ms, eps=eps,
                                  delta=delta, empirical=e, holds=bool(e <= delta)))
        # negative control: well below the certified margin
        mu_lo = max(1, int(0.25 * mu_star))
        e2 = _corr_fail_rate(K, rho, mu_lo, md, ms, eps, rng)
        if e2 is not None:
            control.append(dict(K=K, rho=rho, mu=mu_lo, empirical=e2))
    n_bad = sum(not r["holds"] for r in certified)
    ctrl_fail = sum(r["empirical"] > 0.05 for r in control)
    return dict(n_certified=len(certified), n_violations=n_bad,
                max_excess=max((r["empirical"] - r["delta"] for r in certified),
                               default=0.0),
                n_control=len(control), control_failures=ctrl_fail,
                control_is_informative=ctrl_fail > 0,
                certified=certified[:20], control=control[:20])


if __name__ == "__main__":
    # Every key the paper quotes must be produced here.  An earlier version of this
    # file wrote only X1-X5, so `make all` silently deleted X2b/X6/X7 and then the
    # gate failed; the audit_paper gate is what caught it.
    res = {"seed": SEED, "trials": TRIALS,
           "X1_calculus_vs_montecarlo": X1(),
           "X2_budget_separation": X2(),
           "X2b_onset_regime": X2b(),
           "X3_slice_penalty": X3(),
           "X4_surface_identity_real": X4(),
           "X5_attacker_mix": X5(),
           "X6_bernstein_vs_hoeffding": X6(),
           "X7_kappa_block_certificate": X7(),
           "X8_kappa_is_bounded": x8_kappa_is_bounded(),
           "X9_end_to_end_kappa_free": x9_end_to_end_kappa_free()}
    with open(os.path.join(HERE, "results_attack.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    x1 = res["X1_calculus_vs_montecarlo"]
    print("X1  success-probability formula vs Monte Carlo (%d points)" % x1["n_points"])
    print("    max |mean-field - measured| = %.4f   mean %.4f   Chernoff violations %d"
          % (x1["max_abs_error_meanfield"], x1["mean_abs_error_meanfield"],
             x1["chernoff_violations"]))
    print("\nX2  required budget by adversary")
    print("    %-6s %-4s %-6s %-7s %-9s %-9s %-9s" %
          ("K", "tau", "rho", "wb", "blind", "adaptive", "separation"))
    for r in res["X2_budget_separation"]["rows"]:
        print("    %-6d %-4d %-6d %-7d %-9.1f %-9.1f %-9.1f" %
              (r["K"], r["tau"], r["rho"], r["wb_budget"],
               r["blind_budget_measured"], r["adaptive_budget"], r["separation"]))
    print("    adaptive/blind = %.3f (1.0 => adaptivity buys nothing)"
          % res["X2_budget_separation"]["adaptive_over_blind"])
    print("\nX3  slice penalty (private HyperCube vs uniform rho-subsets)")
    for r in res["X3_slice_penalty"]["rows"]:
        print("    K=%-6d tau=%d rho=%-5d  mean hits slice %.2f / unif %.2f "
              "(pred %.2f)  var x%.2f  budget ratio %.3f"
              % (r["K"], r["tau"], r["rho"], r["mean_hits_slice"],
                 r["mean_hits_uniform"], r["predicted_mean"],
                 r["var_inflation"], r["budget_ratio"]))
    x4 = res["X4_surface_identity_real"]
    print("\nX4  attack surface on %s (n=%d, m=%d), C4, tau*=%g"
          % (x4["network"], x4["n_nodes"], x4["n_edges"], x4["tau_star_v"]))
    for r in x4["records"]:
        print("    K=%-5d anchors=%-3d identity gap=%-3d mean surface=%-9.1f "
              "top-1 edge share: mean %.3f max %.3f"
              % (r["K_requested"], r["n_anchors"], r["identity_max_gap"],
                 r["mean_surface"], r["mean_top1_share"], r["max_top1_share"]))
    print("\nX5  attacker mix: %s -> rule holds = %s"
          % (res["X5_attacker_mix"]["rule"], res["X5_attacker_mix"]["threshold_rule_holds"]))
