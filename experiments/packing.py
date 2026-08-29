"""
packing.py -- E12-E18: the anchored fractional edge PACKING number tau*_A(Q) is the
true certificate exponent, not the conjunction arity d.

Claims tested
-------------
C-P1  For every motif M anchored at v, the optimal anchored-HyperCube share LP has value
      exactly 1/tau*_v(M), so min-over-designs max-replication exponent = 1 - 1/tau*_v.
C-P2  tau*_v(M) <= d with equality iff M's non-anchor-incident edges form a matching.
      => the d-based floor rho >= K^{1-1/d} of the compression law is NOT a floor for
      structured motifs; anchored HyperCube violates it while keeping P_det = 1.
C-P3  Separation: the retrieval exponent r_v(M) (frontier radius) and the robustness
      exponent tau*_v(M) are incomparable. Cycles are the fixed point where they agree.
C-P4  Random rho-replication needs rho ~ K^{1-1/d}; designs need K^{1-1/tau*_v}.
      The design/random gap is POLYNOMIAL in K (K^{1/tau*_v - 1/d}), not Theta(sqrt(log N)).
C-P5  Certified structural robustness m_s* improves by the same polynomial factor.
C-P6  The theory is about CONJUNCTIVE QUERIES, not graphs: E18 runs the whole pipeline on
      an off-graph multi-hop-QA query over ternary relations (documents as first-class
      values) and reproduces tau*_A < d, P_det = 1, the exponent, and certificate tightness.

Run:  python3 packing.py            (writes results_packing.json)
"""
from __future__ import annotations
import itertools, json, math, os, random
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

from lp import tau_star as _tau_star

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20260729


# ---------------------------------------------------------------------------
# Motif library.  A motif = (anchor 'v', list of directed edges over labels).
# Free vertices are every label != 'v'.
# ---------------------------------------------------------------------------
def cycle(k):
    lab = ["v"] + [f"u{i}" for i in range(1, k)]
    return [(lab[i], lab[(i + 1) % k]) for i in range(k)]


def path(k):
    """Directed path of k edges leaving v (one-sided: no return to v)."""
    lab = ["v"] + [f"u{i}" for i in range(1, k + 1)]
    return [(lab[i], lab[i + 1]) for i in range(k)]


def star(d):
    """Fan-out of arity d from the anchor."""
    return [("v", f"u{i}") for i in range(d)]


def scatter_gather(b):
    """v -> m ; m -> s_i (b sinks) ; s_i -> t  (layered fan-out/fan-in flow)."""
    E = [("v", "m")] + [("m", f"s{i}") for i in range(b)] + [(f"s{i}", "t") for i in range(b)]
    return E


def bipartite_cluster(a, b):
    """K_{a,b} with v one of the left vertices."""
    left = ["v"] + [f"l{i}" for i in range(1, a)]
    right = [f"r{j}" for j in range(b)]
    return [(x, y) for x in left for y in right]


def theta(k):
    """Balanced theta: forward path of a edges, backward path of b edges, closer."""
    a = (k - 1) // 2
    b = (k - 1) - a
    E = []
    fwd = ["v"] + [f"f{i}" for i in range(1, a + 1)]
    bwd = ["v"] + [f"b{i}" for i in range(1, b + 1)]
    E += [(fwd[i], fwd[i + 1]) for i in range(a)]
    E += [(bwd[i + 1], bwd[i]) for i in range(b)]
    E += [(fwd[-1], bwd[-1])]
    return E


MOTIFS = {
    "triangle (C3)": cycle(3),
    "C4": cycle(4),
    "C5": cycle(5),
    "C6": cycle(6),
    "C8": cycle(8),
    "C12": cycle(12),
    "star d=3": star(3),
    "star d=5": star(5),
    "star d=8": star(8),
    "path k=4": path(4),
    "path k=6": path(6),
    "scatter-gather b=2": scatter_gather(2),
    "scatter-gather b=3": scatter_gather(3),
    "bipartite K_{2,2}": bipartite_cluster(2, 2),
    "bipartite K_{3,3}": bipartite_cluster(3, 3),
}


def free_vertices(E):
    return sorted({x for e in E for x in e if x != "v"})


# ---------------------------------------------------------------------------
# tau*_v : anchored fractional edge packing number.
#   max sum_e x_e   s.t.  for each FREE vertex u:  sum_{e ni u} x_e <= 1,  x >= 0
# (The anchor carries no constraint: its share is pinned to 1.)
# ---------------------------------------------------------------------------
def tau_star_anchored(E):
    U = free_vertices(E)
    A = np.zeros((len(U), len(E)))
    for j, e in enumerate(E):
        for i, u in enumerate(U):
            if u in e:
                A[i, j] = 1.0
    res = linprog(c=-np.ones(len(E)), A_ub=A, b_ub=np.ones(len(U)),
                  bounds=[(0, None)] * len(E), method="highs")
    assert res.success, res.message
    return float(-res.fun), res.x


# ---------------------------------------------------------------------------
# Share LP: the anchored-HyperCube optimum.
#   maximize  t  s.t.  sum_{u in e, u != v} a_u >= t  for every motif edge e,
#                      sum_u a_u = 1, a >= 0.
# Value should equal 1/tau*_v  (LP duality).  Max replication exponent = 1 - t.
# ---------------------------------------------------------------------------
def share_lp(E):
    U = free_vertices(E)
    n = len(U)
    idx = {u: i for i, u in enumerate(U)}
    # vars: [a_1..a_n, t];  maximize t  ->  minimize -t
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub, b_ub = [], []
    for e in E:
        row = np.zeros(n + 1)
        for x in e:
            if x != "v":
                row[idx[x]] = -1.0        # -sum a_u + t <= 0
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert res.success, res.message
    return float(res.x[-1]), {u: float(res.x[idx[u]]) for u in U}


# ---------------------------------------------------------------------------
# r_v(M): frontier radius = max over motif edges of the min number of FREE vertices
# that must be fixed to pin that edge, going forward from v or backward into v.
# (This is the exponent bippr pays: w^{r_v}.)
# ---------------------------------------------------------------------------
def frontier_radius(E):
    fwd, bwd = defaultdict(list), defaultdict(list)
    for (a, b) in E:
        fwd[a].append(b)
        bwd[b].append(a)

    def bfs(adj):
        dist = {"v": 0}
        q = ["v"]
        while q:
            x = q.pop(0)
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)
        return dist

    df, db = bfs(fwd), bfs(bwd)
    # mixed frontier: alternating walk direction is allowed (symmetrized PPR)
    mix = defaultdict(list)
    for (a, b) in E:
        mix[a].append(b)
        mix[b].append(a)
    dm = bfs(mix)

    INF = float("inf")
    worst_dir = worst_mix = 0
    for (a, b) in E:
        # cost to pin edge (a,b): reach a from v forward (then b is one hop, free),
        # or reach b from v backward (then a is one hop back).
        cf = 0 if a == "v" else df.get(a, INF)
        cb = 0 if b == "v" else db.get(b, INF)
        worst_dir = max(worst_dir, min(cf, cb) + 1)
        cm = min(0 if a == "v" else dm.get(a, INF),
                 0 if b == "v" else dm.get(b, INF))
        worst_mix = max(worst_mix, cm + 1)
    return worst_dir, worst_mix


# ---------------------------------------------------------------------------
# E12: invariant table
# ---------------------------------------------------------------------------
def E12():
    rows = []
    for name, E in MOTIFS.items():
        d = len(E)
        tau, _ = tau_star_anchored(E)
        t, shares = share_lp(E)
        r, rmix = frontier_radius(E)
        # matching test: free vertices each covered by exactly one motif edge
        U = free_vertices(E)
        deg = {u: sum(1 for e in E if u in e) for u in U}
        is_matching_on_free = all(v_ == 1 for v_ in deg.values())
        rows.append(dict(
            motif=name, d=d, tau_star_v=round(tau, 6), share_lp_value=round(t, 6),
            one_over_tau=round(1.0 / tau, 6),
            lp_duality_gap=round(abs(t - 1.0 / tau), 9),
            r_v=(None if r == float("inf") else r), r_v_mixed=rmix,
            bippr_applicable=(r != float("inf")),
            tau_equals_d=(abs(tau - d) < 1e-9),
            matching_on_free=is_matching_on_free,
            paper_exponent=round(1 - 1.0 / d, 6),
            true_exponent=round(1 - 1.0 / tau, 6),
            overprice=round((1 - 1.0 / d) - (1 - 1.0 / tau), 6),
            shares={k: round(v, 4) for k, v in shares.items()},
        ))
    return rows


# ---------------------------------------------------------------------------
# E13: explicit anchored-HyperCube design on a host graph.
# Host: anchor v with out/in degree w, random layered graph carrying many motif copies.
# Verify P_det = 1 (every embedding lands wholly in one cell) and measure rho.
# ---------------------------------------------------------------------------
def build_host(E, w, rng):
    """Materialize every embedding of the motif with branching factor w.
    Returns (edges set, list of embeddings-as-edge-tuples)."""
    U = free_vertices(E)
    # each free slot gets w candidate host vertices, namespaced per slot
    cand = {u: [f"{u}#{i}" for i in range(w)] for u in U}
    edges, embeds = set(), []
    for combo in itertools.product(*[cand[u] for u in U]):
        assign = {"v": "V", **dict(zip(U, combo))}
        et = tuple((assign[a], assign[b]) for (a, b) in E)
        embeds.append(et)
        edges.update(et)
    return sorted(edges), embeds


def hypercube_assign(E, edges, shares_exp, K, rng):
    """Anchored HyperCube. shares_exp: dict free-vertex -> exponent a_u (sum = 1).
    p_u = round(K^{a_u}); cell id = tuple of coords.  Returns dict edge -> set(cells)."""
    U = free_vertices(E)
    p = {}
    for u in U:
        p[u] = max(1, int(round(K ** shares_exp[u])))
    Kreal = 1
    for u in U:
        Kreal *= p[u]
    h = {u: {} for u in U}

    def hv(u, x):
        if x not in h[u]:
            h[u][x] = rng.randrange(p[u])
        return h[u][x]

    # slot list: which motif edges a host edge could fill.  We know the namespacing,
    # so a host edge (a,b) fills exactly the slots whose labels match its namespaces.
    def slots_of(a, b):
        out = []
        for (la, lb) in E:
            ok_a = (a == "V" and la == "v") or (la != "v" and a.startswith(la + "#"))
            ok_b = (b == "V" and lb == "v") or (lb != "v" and b.startswith(lb + "#"))
            if ok_a and ok_b:
                out.append((la, lb))
        return out

    coords = list(itertools.product(*[range(p[u]) for u in U]))
    ui = {u: i for i, u in enumerate(U)}
    place = {}
    for (a, b) in edges:
        cells = set()
        for (la, lb) in slots_of(a, b):
            fixed = {}
            if la != "v":
                fixed[ui[la]] = hv(la, a)
            if lb != "v":
                fixed[ui[lb]] = hv(lb, b)
            for c in coords:
                if all(c[i] == val for i, val in fixed.items()):
                    cells.add(c)
        place[(a, b)] = cells
    return place, Kreal, p


def E13(w=3, Ks=(8, 27, 64, 125, 216, 343)):
    rng = random.Random(SEED)
    out = {}
    for name in ["triangle (C3)", "C4", "C5", "scatter-gather b=2"]:
        E = MOTIFS[name]
        d = len(E)
        tau, _ = tau_star_anchored(E)
        _, shares = share_lp(E)
        edges, embeds = build_host(E, w, rng)
        N = len(edges)
        recs = []
        for K in Ks:
            place, Kreal, p = hypercube_assign(E, edges, shares, K, rng)
            # deterministic detection check
            covered = 0
            for et in embeds:
                inter = None
                for e in et:
                    inter = place[e] if inter is None else (inter & place[e])
                    if not inter:
                        break
                if inter:
                    covered += 1
            rho = [len(place[e]) for e in edges]
            loads = defaultdict(int)
            for e in edges:
                for c in place[e]:
                    loads[c] += 1
            recs.append(dict(
                K_requested=K, K=Kreal, N=N, shares=p,
                P_det=covered / len(embeds),
                rho_bar=float(np.mean(rho)), rho_max=int(max(rho)),
                B=int(max(loads.values())) if loads else 0,
                paper_floor=Kreal ** (1 - 1.0 / d),
                new_floor=Kreal ** (1 - 1.0 / tau),
            ))
        # fit measured exponent of rho_max vs K
        lk = np.log([r["K"] for r in recs])
        lr = np.log([max(r["rho_max"], 1) for r in recs])
        slope = float(np.polyfit(lk, lr, 1)[0])
        out[name] = dict(d=d, tau_star_v=tau, w=w,
                         predicted_exponent=1 - 1.0 / tau,
                         paper_exponent=1 - 1.0 / d,
                         measured_exponent=slope, records=recs)
    return out


# ---------------------------------------------------------------------------
# E14: random rho-replication threshold vs the design.
# Detection prob for random replication: 1 - (1 - (rho/K)^d)^K  (indep cells).
# Find the smallest rho with P_det >= 1/2; compare with the design's rho at P_det = 1.
# ---------------------------------------------------------------------------
def E14(Ks=(64, 256, 1024, 4096, 16384)):
    out = {}
    for name in ["triangle (C3)", "C4", "C5", "C6"]:
        E = MOTIFS[name]
        d = len(E)
        tau, _ = tau_star_anchored(E)
        recs = []
        for K in Ks:
            rho_rand = None
            for rho in range(1, K + 1):
                pdet = 1 - (1 - (rho / K) ** d) ** K
                if pdet >= 0.5:
                    rho_rand = rho
                    break
            rho_design = K ** (1 - 1.0 / tau)
            recs.append(dict(K=K, rho_random_half=rho_rand,
                             rho_random_pred=K ** (1 - 1.0 / d),
                             rho_design=rho_design,
                             gap=rho_rand / rho_design))
        lk = np.log([r["K"] for r in recs])
        lg = np.log([r["gap"] for r in recs])
        out[name] = dict(d=d, tau_star_v=tau,
                         predicted_gap_exponent=1.0 / tau - 1.0 / d,
                         measured_gap_exponent=float(np.polyfit(lk, lg, 1)[0]),
                         records=recs)
    return out


# ---------------------------------------------------------------------------
# E15: certified structural robustness m_s* = (mu - m_d)/rho, mu = K/4.
# ---------------------------------------------------------------------------
def E15(Ks=(64, 256, 1024, 4096, 16384)):
    out = {}
    for name in ["triangle (C3)", "C5", "C6", "C8"]:
        E = MOTIFS[name]
        d = len(E)
        tau, _ = tau_star_anchored(E)
        recs = []
        for K in Ks:
            mu = K / 4.0
            ms_paper = mu / K ** (1 - 1.0 / d)
            ms_new = mu / K ** (1 - 1.0 / tau)
            recs.append(dict(K=K, mu=mu, ms_random=ms_paper, ms_design=ms_new,
                             improvement=ms_new / ms_paper))
        out[name] = dict(d=d, tau_star_v=tau,
                         predicted_improvement_exponent=1.0 / tau - 1.0 / d,
                         records=recs)
    return out


# ---------------------------------------------------------------------------
# E16: the packing counting bound.  A cell holding b edges contains at most
# b^{tau*_v} anchored embeddings (Friedgut / Bollobas-Thomason).  This is the
# counting lemma the lower bound needs; the paper used binom(b,d) instead, which
# is the matching-only special case.  Verified by sampling random cells.
# ---------------------------------------------------------------------------
def E16(w=4, trials=400):
    rng = random.Random(SEED + 1)
    out = {}
    for name in ["triangle (C3)", "C4", "C5", "scatter-gather b=2", "star d=3"]:
        E = MOTIFS[name]
        d = len(E)
        tau, _ = tau_star_anchored(E)
        edges, embeds = build_host(E, w, rng)
        eset = {e: i for i, e in enumerate(edges)}
        worst_pack = worst_binom = 0.0
        for _ in range(trials):
            b = rng.randrange(d, len(edges) + 1)
            cell = set(rng.sample(edges, b))
            cnt = sum(1 for et in embeds if all(e in cell for e in et))
            if cnt:
                worst_pack = max(worst_pack, cnt / (b ** tau))
                worst_binom = max(worst_binom, cnt / math.comb(b, d))
        out[name] = dict(d=d, tau_star_v=tau, w=w,
                         max_ratio_to_b_pow_tau=worst_pack,
                         packing_bound_holds=bool(worst_pack <= 1.0 + 1e-9),
                         max_ratio_to_binom_bd=worst_binom)
    return out


# ---------------------------------------------------------------------------
# E17: skew.  BKS's tau*-optimal load holds for skew-free instances.  A hub of
# degree h in the anchored triangle forces the HyperCube cell holding the hub's
# coordinate slice to absorb h edges, degrading the effective exponent.  Real
# graphs are skewed (this paper's own E2: sigma_M ~ n^0.7), so the fee is real.
# ---------------------------------------------------------------------------
def E17(K=64, base=8, hubs=(1, 2, 4, 8, 16, 32)):
    rng = random.Random(SEED + 2)
    E = MOTIFS["triangle (C3)"]
    tau, _ = tau_star_anchored(E)
    _, shares = share_lp(E)
    recs = []
    for h in hubs:
        # host: u1 has `base` candidates; one designated u1-vertex is a hub with
        # `h*base` outgoing middle edges, the rest have `base`.
        u1s = [f"u1#{i}" for i in range(base)]
        u2s = [f"u2#{j}" for j in range(base * h)]
        edges = set()
        for a in u1s:
            edges.add(("V", a))
        for b_ in u2s:
            edges.add((b_, "V"))
        for i, a in enumerate(u1s):
            tgt = u2s if i == 0 else u2s[:base]      # vertex 0 is the hub
            for b_ in tgt:
                edges.add((a, b_))
        edges = sorted(edges)
        place, Kreal, p = hypercube_assign(E, edges, shares, K, rng)
        loads = defaultdict(int)
        for e in edges:
            for c in place[e]:
                loads[c] += 1
        L = max(loads.values())
        avg = sum(loads.values()) / len(loads)
        recs.append(dict(hub_factor=h, N=len(edges), K=Kreal,
                         max_load=L, avg_load=avg, skew_ratio=L / avg,
                         ideal_load=len(edges) / Kreal * (Kreal ** (1 - 1.0 / tau))))
    return dict(tau_star_v=tau, K=K, records=recs)


# ---------------------------------------------------------------------------
# E18: OFF-GRAPH instantiation.  A multi-hop QA query over ternary relations --
# no graph, no edges, no anchor neighbourhood.  Demonstrates that the Part-I
# theory (packing law + certificate) is a statement about conjunctive queries.
#
#   Q(ans) :- Cite(a, d1, e1), Cite2(e1, d2, e2), Assert(e2, d3, ans)
#
# `a` is bound (the entity asked about); d1,d2,d3 are DOCUMENTS (first-class
# values, not vertices); e1,e2 intermediate entities; ans the answer.
# Free vars {d1,e1,d2,e2,d3,ans}; d = 3 atoms; tau*_A = 2.
# ---------------------------------------------------------------------------
QA_QUERY = [
    ("Cite",   ["a", "d1", "e1"]),
    ("Cite2",  ["e1", "d2", "e2"]),
    ("Assert", ["e2", "d3", "ans"]),
]
QA_BOUND = {"a"}


def cq_free_vars(atoms, bound):
    seen = []
    for _, vs in atoms:
        for v_ in vs:
            if v_ not in bound and v_ not in seen:
                seen.append(v_)
    return seen


def tau_star_cq(atoms, bound):
    """Anchored fractional edge packing number.  Delegates to the canonical LP in
    lp.py; kept as a name because the rest of this file predates that module.

    The x <= 1 upper bound matters: without it a query in which some atom has no
    free variable (every one of its variables anchored) makes the LP unbounded.
    That is not hypothetical -- it is what pinning a heavy hitter to a constant
    does (E24), and an earlier local copy of this function omitted the bound.
    """
    return _tau_star(atoms, frozenset(bound))


def share_lp_cq(atoms, bound):
    U = cq_free_vars(atoms, bound)
    n = len(U)
    idx = {u: i for i, u in enumerate(U)}
    c = np.zeros(n + 1); c[-1] = -1.0
    A_ub, b_ub = [], []
    for _, vs in atoms:
        row = np.zeros(n + 1)
        for v_ in vs:
            if v_ in idx:
                row[idx[v_]] = -1.0
        row[-1] = 1.0
        A_ub.append(row); b_ub.append(0.0)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert res.success
    return float(res.x[-1]), {u: float(res.x[idx[u]]) for u in U}


def E18(atoms=QA_QUERY, bound=QA_BOUND, w=8, Ks=(4, 9, 16, 25, 36, 64)):
    """Build the corpus, run anchored HyperCube, verify P_det=1 and rho_max,
    then simulate the Stackelberg certificate off-graph."""
    rng = random.Random(SEED + 3)
    U = cq_free_vars(atoms, bound)
    d = len(atoms)
    tau = tau_star_cq(atoms, bound)
    t, shares = share_lp_cq(atoms, bound)

    # corpus: each free var ranges over w namespaced constants; bound var is fixed
    dom = {u: [f"{u}#{i}" for i in range(w)] for u in U}
    dom.update({b_: [f"{b_}*"] for b_ in bound})
    witnesses, tuples = [], set()
    for combo in itertools.product(*[dom[u] for u in U]):
        assign = dict(zip(U, combo))
        assign.update({b_: dom[b_][0] for b_ in bound})
        wt = tuple((rel, tuple(assign[v_] for v_ in vs)) for rel, vs in atoms)
        witnesses.append((assign, wt))
        tuples.update(wt)
    tuples = sorted(tuples)

    recs = []
    for K in Ks:
        p = {u: max(1, int(round(K ** shares[u]))) for u in U}
        Kreal = 1
        for u in U:
            Kreal *= p[u]
        # balanced hash: cycle values through buckets under a random permutation,
        # so every bucket is occupied whenever |domain| >= p_u (what a real
        # deployment does; uniform-random hashing only adds collision noise).
        h = {}
        for u in U:
            vals = list(dom[u])
            rng.shuffle(vals)
            perm = list(range(p[u]))
            rng.shuffle(perm)
            h[u] = {v_: perm[i % p[u]] for i, v_ in enumerate(vals)}

        def hv(u, x):
            return h[u][x]

        coords = list(itertools.product(*[range(p[u]) for u in U]))
        ui = {u: i for i, u in enumerate(U)}
        place = {}
        for tup in tuples:
            rel, vals = tup
            vs = dict(atoms)[rel]
            fixed = {ui[v_]: hv(v_, val) for v_, val in zip(vs, vals) if v_ in ui}
            place[tup] = {c for c in coords
                          if all(c[i] == val for i, val in fixed.items())}
        covered = sum(1 for _, wt in witnesses
                      if set.intersection(*[place[t_] for t_ in wt]))
        rho = [len(place[t_]) for t_ in tuples]
        loads = defaultdict(int)
        for t_ in tuples:
            for c in place[t_]:
                loads[c] += 1

        # --- certificate simulation, off-graph ---
        # clean firing cells: those containing a full witness
        firing = set()
        for _, wt in witnesses:
            firing |= set.intersection(*[place[t_] for t_ in wt])
        c1 = len(firing)
        mu = c1 - Kreal // 2
        rmax = max(rho)
        flip_budget = None
        if mu > 0:
            # adversary silences rmax firing cells per structural poison (assumption A3)
            for ms in range(0, c1 + 2):
                if c1 - rmax * ms <= Kreal // 2:
                    flip_budget = ms
                    break
        recs.append(dict(K_requested=K, K=Kreal, n_tuples=len(tuples),
                         n_witnesses=len(witnesses), shares=p,
                         P_det=covered / len(witnesses),
                         rho_bar=float(np.mean(rho)), rho_max=rmax,
                         B=int(max(loads.values())),
                         paper_floor=Kreal ** (1 - 1.0 / d),
                         new_floor=Kreal ** (1 - 1.0 / tau),
                         c1=c1, mu=mu, flip_budget=flip_budget,
                         cert_predicted_budget=(None if mu <= 0
                                                else math.ceil(mu / rmax))))
    lk = np.log([r["K"] for r in recs])
    lr = np.log([max(r["rho_max"], 1) for r in recs])
    return dict(query=[[rel, vs] for rel, vs in atoms], bound=sorted(bound),
                d=d, tau_star_A=tau, share_lp_value=t,
                lp_duality_gap=abs(t - 1.0 / tau),
                free_vars=U, w=w,
                predicted_exponent=1 - 1.0 / tau,
                paper_exponent=1 - 1.0 / d,
                measured_exponent=float(np.polyfit(lk, lr, 1)[0]),
                records=recs)



# ==========================================================================
# E24: skew separation.  rho is a function of the share vector alone, so it is
# invariant to the data distribution; load is not.  And the textbook skew
# remedy -- pin the heavy hitter to a constant and solve the residual query --
# is exactly the ANCHORING operation, which deletes a packing constraint and
# therefore can only INCREASE rho_max.  Fixing load can cost security.
# ==========================================================================
def e24_skew_separation(K=64, N=6000, seed=SEED):
    rng = random.Random(seed)
    atoms = [("R", ["x", "y"]), ("S", ["y", "z"]), ("T", ["z", "x"])]
    _, shares = share_lp_cq(atoms, frozenset())
    p = {u: max(1, int(round(K ** shares[u]))) for u in shares}
    U = sorted(p)

    def rho_pred(vs):
        r = 1
        for u in U:
            if u not in vs:
                r *= p[u]
        return r

    rows = []
    for name, (fx, fy) in [("uniform", (0.0, 0.0)), ("skew-x", (0.5, 0.0)),
                           ("skew-x hard", (0.8, 0.0)), ("skew-xy", (0.3, 0.3)),
                           ("skew-xy hard", (0.7, 0.7))]:
        hsh = {u: {} for u in U}

        def h(u, x):
            if x not in hsh[u]:
                hsh[u][x] = rng.randrange(p[u])
            return hsh[u][x]

        tup = [{"x": 0 if rng.random() < fx else rng.randrange(1, 300),
                "y": 0 if rng.random() < fy else rng.randrange(1, 300)}
               for _ in range(N)]
        load = {}
        for c in itertools.product(*[range(p[u]) for u in U]):
            load[c] = sum(1 for t in tup
                          if all(h(u, t[u]) == c[i] for i, u in enumerate(U)
                                 if u in ("x", "y")))
        mx = max(load.values())
        mean = sum(load.values()) / len(load)
        # rho, measured by enumeration, for one R-tuple
        t0 = tup[0]
        rho_meas = sum(1 for c in itertools.product(*[range(p[u]) for u in U])
                       if all(h(u, t0[u]) == c[i] for i, u in enumerate(U)
                              if u in ("x", "y")))
        fX = max(sum(1 for t in tup if t["x"] == v) for v in {t["x"] for t in tup}) / N
        fY = max(sum(1 for t in tup if t["y"] == v) for v in {t["y"] for t in tup}) / N
        rows.append(dict(case=name, rho_meas=rho_meas, rho_pred=rho_pred({"x", "y"}),
                         inflation=mx / mean, max_load=mx, mean_load=mean,
                         lb_single=max(fX * p["x"], fY * p["y"]),
                         lb_product=fX * p["x"] * fY * p["y"], f_x=fX, f_y=fY))

    # anchoring monotonicity over random queries
    bad = strict = 0
    trials = 400
    for _ in range(trials):
        nv = rng.randrange(3, 7)
        V = [f"v{i}" for i in range(nv)]
        Q = [(f"R{j}", rng.sample(V, rng.randrange(2, min(4, nv) + 1)))
             for j in range(rng.randrange(2, 6))]
        VS = sorted({v for _, vs in Q for v in vs})
        A = frozenset(rng.sample(VS, rng.randrange(0, max(1, len(VS) - 1))))
        free = [v for v in VS if v not in A]
        if not free:
            continue
        hot = rng.choice(free)
        t0, t1 = tau_star_cq(Q, A), tau_star_cq(Q, A | {hot})
        if t1 < t0 - 1e-9:
            bad += 1
        if t1 > t0 + 1e-9:
            strict += 1

    Kbig = 1024
    tf, tp = tau_star_cq(atoms, frozenset()), tau_star_cq(atoms, frozenset({"x"}))
    return dict(K=K, rows=rows,
                rho_invariant=len({r["rho_meas"] for r in rows}) == 1
                and all(r["rho_meas"] == r["rho_pred"] for r in rows),
                lb_single_holds=all(r["inflation"] >= r["lb_single"] - 1e-9 for r in rows),
                lb_product_holds=all(r["inflation"] >= r["lb_product"] - 1e-9 for r in rows),
                max_inflation=max(r["inflation"] for r in rows),
                anchor_monotone_violations=bad, anchor_monotone_trials=trials,
                anchor_strict_increases=strict,
                triangle_tau_free=tf, triangle_tau_pinned=tp, K_big=Kbig,
                triangle_rho_free=Kbig ** (1 - 1 / tf),
                triangle_rho_pinned=Kbig ** (1 - 1 / tp),
                pinning_cost=(Kbig ** (1 - 1 / tp)) / (Kbig ** (1 - 1 / tf)))



# ==========================================================================
# E27: is the law cycle-specific?  Enumerate EVERY connected pattern up to 5
# edges -- all shapes, not a curated list -- anchor at every vertex orbit, and
# compute tau*_A.  Also: the two corners the formula must reproduce (a single
# atom = the disjoint-partition regime of passage-level defenses; a matching =
# the arity rate), tau* monotonicity under adding atoms (so a query CLASS is
# bounded by its largest pattern), and the cost of serving a class with one
# partition instead of re-drawing per query.
# ==========================================================================
def _pattern_class(G):
    n, m = G.number_of_nodes(), G.number_of_edges()
    degs = [d for _, d in G.degree()]
    if m == n - 1:
        if max(degs) == m:
            return "star"
        if max(degs) <= 2:
            return "path"
        return "tree"
    if m == n and max(degs) <= 2:
        return "cycle"
    return "cyclic-other"


def e27_all_patterns(max_edges=5, max_nodes=6, K=1024, seed=SEED):
    import networkx as nx
    rng = random.Random(seed)

    seen, graphs = set(), []
    for G in nx.graph_atlas_g():
        if not (2 <= G.number_of_nodes() <= max_nodes):
            continue
        if not (1 <= G.number_of_edges() <= max_edges):
            continue
        if not nx.is_connected(G):
            continue
        h = nx.weisfeiler_lehman_graph_hash(G)
        if h in seen:
            continue
        seen.add(h)
        graphs.append(G)

    rows = []
    for G in graphs:
        d = G.number_of_edges()
        atoms = [("E", [f"n{u}", f"n{v}"]) for u, v in G.edges()]
        h = nx.weisfeiler_lehman_graph_hash(G)
        orbits = set()
        for v in G.nodes():
            sig = (G.degree(v), tuple(sorted(G.degree(u) for u in G.neighbors(v))), h)
            if sig in orbits:
                continue
            orbits.add(sig)
            t = _tau_star(atoms, frozenset({f"n{v}"}))
            rows.append(dict(d=d, n=G.number_of_nodes(), tau_star=t,
                             family=_pattern_class(G),
                             gain=K ** (1 / t) / K ** (1 / d),
                             strict=bool(t < d - 1e-9)))

    by_family = {}
    for f in sorted({r["family"] for r in rows}):
        rs = [r for r in rows if r["family"] == f]
        g = sorted(r["gain"] for r in rs)
        by_family[f] = dict(n=len(rs), n_strict=sum(r["strict"] for r in rs),
                            median_gain=g[len(g) // 2], max_gain=max(g))

    # the two corners the formula must reproduce
    single = _tau_star([("R", ["v", "x"])], frozenset({"v"}))
    matching = _tau_star([("R", [f"a{i}", f"b{i}"]) for i in range(4)], frozenset())
    star_centre = _tau_star([("R", ["v", f"u{i}"]) for i in range(4)], frozenset({"v"}))
    star_leaf = _tau_star([("R", ["v", "c"])] + [("R", ["c", f"u{i}"]) for i in range(3)],
                          frozenset({"v"}))

    # monotonicity: adding an atom cannot lower tau*, so a class is bounded by
    # its largest member
    mono_bad = mono_strict = 0
    trials = 600
    for _ in range(trials):
        V = [f"v{i}" for i in range(rng.randrange(3, 7))]
        Q = [("R", rng.sample(V, rng.randrange(2, min(4, len(V)) + 1)))
             for _ in range(rng.randrange(1, 5))]
        A = frozenset(rng.sample(V, rng.randrange(0, len(V) - 1)))
        extra = ("R", rng.sample(V, rng.randrange(2, min(4, len(V)) + 1)))
        t0, t1 = _tau_star(Q, A), _tau_star(Q + [extra], A)
        mono_bad += t1 < t0 - 1e-9
        mono_strict += t1 > t0 + 1e-9

    # a realistic GraphRAG query class, per-query vs one shared partition
    QCLASS = {
        "2hop_entity_qa": [("Reg", ["v", "p"]), ("Tgt", ["ans", "p"])],
        "3hop_provenance": [("Cite", ["v", "b"]), ("Cite", ["b", "c"]),
                            ("Asserts", ["c", "ans"])],
        "co_mention": [("Ment", ["d", "v"]), ("Ment", ["d", "ans"])],
    }
    per_query = {}
    for name, atoms in QCLASS.items():
        t = _tau_star(atoms, frozenset({"v"}))
        per_query[name] = dict(d=len(atoms), tau_star=t, rho=K ** (1 - 1 / t),
                               gain=K ** (1 / t) / K ** (1 / len(atoms)))
    pooled = [a for atoms in QCLASS.values() for a in atoms]
    t_pool = _tau_star(pooled, frozenset({"v"}))

    gains = sorted(r["gain"] for r in rows)
    return dict(K=K, max_edges=max_edges, n_cases=len(rows), n_graphs=len(graphs),
                n_strict=sum(r["strict"] for r in rows),
                frac_strict=sum(r["strict"] for r in rows) / len(rows),
                median_gain=gains[len(gains) // 2], max_gain=max(gains),
                by_family=by_family, rows=rows[:80],
                corner_single_atom=single, corner_matching=matching,
                corner_star_centre=star_centre, corner_star_leaf=star_leaf,
                mono_trials=trials, mono_violations=mono_bad, mono_strict=mono_strict,
                qclass=per_query, qclass_pooled_tau=t_pool,
                qclass_pooled_rho=K ** (1 - 1 / t_pool),
                qclass_per_query_max_rho=max(v["rho"] for v in per_query.values()))



# ==========================================================================
# E28: the security-native experiments.
#   (a) multi-target amplification.  A shared-relation attack across m target
#       queries does NOT multiply the affected-cell count by m: the cell set of
#       a forged tuple depends on the variable ROLES it fills, not on how many
#       queries use it.  Gamma(e) = |U_j C(Q_j,e)| <= sigma(e) rho(e), with
#       sigma the number of distinct roles -- independent of m.
#   (b) observed vs bound: rho_obs(e)/K^{1-1/tau*} must never exceed 1.
#   (c) adaptive attacker with FULL knowledge of the partition, greedily
#       maximising |U_{e in P} C(e)|; must still obey |.| <= b rho_max.
#   (d) certificate tightness: b_cert = floor((t-1)/rho_max) against the exact
#       break point ceil(t/rho_max); they differ by at most one poisoned tuple.
# ==========================================================================
class _Cube:
    """Anchored HyperCube exposing the affected-cell set C(e) of a tuple."""

    def __init__(self, atoms, anchors, K, rng):
        self.atoms = atoms
        self.anchors = frozenset(anchors)
        self.tau = _tau_star(atoms, self.anchors)
        _, sh = share_lp_cq(atoms, self.anchors)
        self.free = sorted(sh)
        self.p = {u: max(1, int(round(K ** sh[u]))) for u in self.free}
        self.K = int(np.prod([self.p[u] for u in self.free]))
        self.rng = rng
        self._h = {u: {} for u in self.free}
        self._stride = {}
        s = 1
        for u in reversed(self.free):
            self._stride[u] = s
            s *= self.p[u]

    def h(self, u, x):
        t = self._h[u]
        if x not in t:
            t[x] = self.rng.randrange(self.p[u])
        return t[x]

    def cells_of(self, tup, slot):
        """C(e): the cells a tuple filling `slot` lands in.  Built directly from
        the fixed coordinates -- never scans all K cells."""
        fix = {u: self.h(u, tup[i]) for i, u in enumerate(slot) if u in self._h}
        ranges = [[fix[u]] if u in fix else range(self.p[u]) for u in self.free]
        return {sum(c[i] * self._stride[u] for i, u in enumerate(self.free))
                for c in itertools.product(*ranges)}

    def rho_pred(self, slot):
        r = 1
        for u in self.free:
            if u not in slot:
                r *= self.p[u]
        return r

    @property
    def rho_max(self):
        return max(self.rho_pred(tuple(a[1])) for a in self.atoms)


def e28_security(K=256, seed=SEED):
    rng = random.Random(seed)

    # ---- (a) multi-target ------------------------------------------------
    # Gamma(e) = |U_r C_r| over the roles r that e fills across all m targets.
    # Computed by inclusion-exclusion on subcubes: no cell enumeration, so m can
    # be large.  C_r fixes the coordinates of the free variables in role r; an
    # intersection is a subcube fixing the union of those coordinates, and is
    # empty if two roles disagree on a shared coordinate (they cannot here,
    # since one tuple gives one value per variable).
    multi = []
    for m in (1, 2, 4, 8, 16):
        atoms = ([("Rel", ["v", "p"])]
                 + [("Mid", ["p", f"b{j}"]) for j in range(m)]
                 + [("Tgt", [f"b{j}", f"a{j}"]) for j in range(m)])
        free = sorted({u for _, vs in atoms for u in vs} - {"v"})
        p = {u: 2 for u in free}                    # one bit per free variable
        K = 2 ** len(free)
        roles = [("v", "p")] + [("p", f"b{j}") for j in range(m)]

        def size(fixed):                            # |subcube fixing `fixed`|
            r = 1
            for u in free:
                if u not in fixed:
                    r *= p[u]
            return r

        gamma = 0
        for k in range(1, len(roles) + 1):
            for S in itertools.combinations(roles, k):
                fixed = {u for r in S for u in r if u in p}
                gamma += (-1) ** (k + 1) * size(fixed)
        single = size({u for u in roles[0] if u in p})
        multi.append(dict(m=m, K=K, tau_star=_tau_star(atoms, frozenset({"v"})),
                          n_roles=len(roles), rho_single=single, gamma=gamma,
                          gamma_over_rho=gamma / max(single, 1),
                          naive_m_rho=m * single,
                          sum_of_roles=sum(size({u for u in r if u in p})
                                           for r in roles)))

    QS = {"2hop": [("R", ["v", "p"]), ("S", ["ans", "p"])],
          "3chain": [("R", ["v", "b"]), ("R", ["b", "c"]), ("S", ["c", "ans"])],
          "triangle": [("R", ["v", "x"]), ("R", ["x", "y"]), ("R", ["y", "v"])],
          "C4": [("R", ["v", "x"]), ("R", ["x", "y"]), ("R", ["y", "z"]),
                 ("R", ["z", "v"])]}

    # ---- (b) observed vs bound ------------------------------------------
    obs = []
    for name, atoms in QS.items():
        for Kq in (64, 256, 1024):
            cube = _Cube(atoms, {"v"}, Kq, rng)
            bound = cube.K ** (1 - 1 / cube.tau)
            mx = 0
            for i in range(200):
                e = (f"X{rng.randrange(9999)}", f"Y{rng.randrange(9999)}")
                slot = tuple(rng.choice(atoms)[1])
                mx = max(mx, len(cube.cells_of(e, slot)))
            obs.append(dict(query=name, K=cube.K, tau_star=cube.tau,
                            rho_obs_max=mx, bound=bound, ratio=mx / bound))

    # ---- (c) adaptive attacker, full knowledge --------------------------
    adaptive = []
    for name in ("triangle", "C4"):
        atoms = QS[name]
        for Kq in (64, 256):
            cube = _Cube(atoms, {"v"}, Kq, rng)
            cand = {}
            for i in range(150):
                ce = ((f"X{i}", f"Y{i}"), tuple(rng.choice(atoms)[1]))
                cand[ce] = cube.cells_of(*ce)
            got = set()
            for b in range(1, 9):
                best = max(cand, key=lambda ce: len(cand[ce] - got))
                got |= cand[best]
                if b in (1, 2, 4, 8):
                    adaptive.append(dict(query=name, K=cube.K, b=b,
                                         affected=len(got),
                                         bound=b * cube.rho_max,
                                         holds=len(got) <= b * cube.rho_max))

    # ---- (d) certificate tightness --------------------------------------
    tight = []
    for rho in (1, 2, 4, 8, 16, 32, 64):
        for t in (16, 64, 128, 256, 512):
            b_cert = (t - 1) // rho              # floor((t-1)/rho): certified safe
            b_break = -(-t // rho)               # ceil(t/rho): exact break (Thm tight)
            tight.append(dict(rho=rho, t=t, b_cert=b_cert, b_break=b_break,
                              gap=b_break - b_cert))

    return dict(K=K, multi_target=multi, observed=obs, adaptive=adaptive,
                tightness=tight,
                gamma_never_scales_with_m=all(r["gamma"] <= r["naive_m_rho"] for r in multi),
                gamma_max_over_rho=max(r["gamma_over_rho"] for r in multi),
                obs_never_exceeds_bound=all(r["ratio"] <= 1 + 1e-9 for r in obs),
                obs_max_ratio=max(r["ratio"] for r in obs),
                adaptive_bound_holds=all(r["holds"] for r in adaptive),
                tightness_max_gap=max(r["gap"] for r in tight),
                tightness_all_within_one=all(r["gap"] <= 1 for r in tight))


if __name__ == "__main__":
    res = {"seed": SEED,
           "E12_invariants": E12(),
           "E13_hypercube": E13(),
           "E14_design_vs_random": E14(),
           "E15_certified_robustness": E15(),
           "E16_packing_counting_bound": E16(),
           "E17_skew": E17(),
           "E24_skew_separation": e24_skew_separation(),
           "E27_all_patterns": e27_all_patterns(),
           "E28_security": e28_security(),
           "E18_offgraph_multihop_qa": E18()}
    with open(os.path.join(HERE, "results_packing.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    print("E12  motif                d   tau*_v   r_v  r_mix   paper 1-1/d   true 1-1/tau*   dual gap")
    for r in res["E12_invariants"]:
        rv = "inf" if r["r_v"] is None else str(r["r_v"])
        print(f"     {r['motif']:<20}{r['d']:>3}  {r['tau_star_v']:>6.3f}  {rv:>4} {r['r_v_mixed']:>5}"
              f"   {r['paper_exponent']:>10.4f}   {r['true_exponent']:>12.4f}"
              f"   {r['lp_duality_gap']:>8.2e}")
    print("\nE13  anchored HyperCube (P_det must be 1.0)")
    for name, o in res["E13_hypercube"].items():
        print(f"     {name:<20} d={o['d']} tau*={o['tau_star_v']:.3f}  "
              f"pred={o['predicted_exponent']:.4f} meas={o['measured_exponent']:.4f} "
              f"paper={o['paper_exponent']:.4f}")
        for r in o["records"]:
            print(f"        K={r['K']:>5} N={r['N']:>5} P_det={r['P_det']:.3f} "
                  f"rho_max={r['rho_max']:>5} paper_floor={r['paper_floor']:>9.2f} "
                  f"new_floor={r['new_floor']:>9.2f} B={r['B']}")
    print("\nE14  design vs random replication")
    for name, o in res["E14_design_vs_random"].items():
        print(f"     {name:<16} pred gap exp={o['predicted_gap_exponent']:.4f} "
              f"meas={o['measured_gap_exponent']:.4f}")
    print("\nE15  certified robustness improvement")
    for name, o in res["E15_certified_robustness"].items():
        last = o["records"][-1]
        print(f"     {name:<16} K={last['K']}: m_s* {last['ms_random']:.2f} -> "
              f"{last['ms_design']:.2f}  ({last['improvement']:.1f}x)")
    print("\nE16  per-cell packing counting bound (must be <= 1.0)")
    for name, o in res["E16_packing_counting_bound"].items():
        print(f"     {name:<20} tau*={o['tau_star_v']:.2f} "
              f"max cnt/b^tau*={o['max_ratio_to_b_pow_tau']:.4f} "
              f"holds={o['packing_bound_holds']}  cnt/binom(b,d) max={o['max_ratio_to_binom_bd']:.2e}")
    print("\nE17  skew fee (anchored triangle, K=%d)" % res["E17_skew"]["K"])
    for r in res["E17_skew"]["records"]:
        print(f"     hub x{r['hub_factor']:<3} N={r['N']:>5} max_load={r['max_load']:>5} "
              f"avg={r['avg_load']:>8.1f} skew_ratio={r['skew_ratio']:.2f}")
    o = res["E27_all_patterns"]
    print(f"\nE27  every connected pattern up to {o['max_edges']} edges "
          f"({o['n_graphs']} shapes, {o['n_cases']} shape-anchor pairs)")
    print(f"     tau*_A < d in {o['n_strict']}/{o['n_cases']} = {100*o['frac_strict']:.0f}%   "
          f"median gain {o['median_gain']:.2f}x  max {o['max_gain']:.1f}x  (K={o['K']})")
    for f, v in o["by_family"].items():
        print(f"       {f:<14} {v['n_strict']:3d}/{v['n']:<3d} strict   "
              f"median {v['median_gain']:5.2f}x  max {v['max_gain']:6.1f}x")
    print(f"     corners: single atom tau*={o['corner_single_atom']:.0f} (rho=1, disjoint); "
          f"matching tau*={o['corner_matching']:.0f}=d (the arity rate);")
    print(f"              star at centre tau*={o['corner_star_centre']:.0f}=d, "
          f"same star at a leaf tau*={o['corner_star_leaf']:.0f}")
    print(f"     tau* monotone under adding an atom: "
          f"{o['mono_trials']-o['mono_violations']}/{o['mono_trials']} "
          f"(strict in {o['mono_strict']})")
    for n, v in o["qclass"].items():
        print(f"     class member {n:<16} d={v['d']} tau*={v['tau_star']:.2f} "
              f"rho={v['rho']:.0f} gain {v['gain']:.1f}x")
    print(f"     one shared partition for the class: rho={o['qclass_pooled_rho']:.0f} vs "
          f"per-query max {o['qclass_per_query_max_rho']:.0f} -> re-draw per query")

    o = res["E24_skew_separation"]
    print("\nE24  skew separation: rho is share-determined, load is not")
    for r in o["rows"]:
        print(f"     {r['case']:<14} rho={r['rho_meas']:>3} (pred {r['rho_pred']:>3})  "
              f"load max/mean={r['inflation']:6.2f}  lb_single={r['lb_single']:5.2f} "
              f"lb_prod={r['lb_product']:5.2f}")
    print(f"     rho invariant={o['rho_invariant']}  lb_single={o['lb_single_holds']} "
          f"lb_product={o['lb_product_holds']}")
    print(f"     anchoring monotone: {o['anchor_monotone_trials']-o['anchor_monotone_violations']}"
          f"/{o['anchor_monotone_trials']} (strict in {o['anchor_strict_increases']})")
    print(f"     pinning x in the triangle: tau* {o['triangle_tau_free']:.2f} -> "
          f"{o['triangle_tau_pinned']:.2f}, rho_max {o['triangle_rho_free']:.0f} -> "
          f"{o['triangle_rho_pinned']:.0f} at K={o['K_big']} ({o['pinning_cost']:.1f}x worse)")

    o = res["E18_offgraph_multihop_qa"]
    print(f"\nE18  OFF-GRAPH multi-hop QA: d={o['d']} tau*_A={o['tau_star_A']:g} "
          f"dualgap={o['lp_duality_gap']:.1e} paper={o['paper_exponent']:.4f} "
          f"pred={o['predicted_exponent']:.4f} meas={o['measured_exponent']:.4f}")
    for r in o["records"]:
        print(f"     K={r['K']:>5} tuples={r['n_tuples']:>4} P_det={r['P_det']:.3f} "
              f"rho_max={r['rho_max']:>4} paper_floor={r['paper_floor']:>8.2f} "
              f"mu={r['mu']:>5} cert_pred={r['cert_predicted_budget']} "
              f"flip@={r['flip_budget']}")
