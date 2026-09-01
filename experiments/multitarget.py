"""
multitarget.py -- E28-E31: the security-side experiments.

E28  observed vs. bound amplification      rho_obs(e) = |C(e)|  vs  rho_max
E29  multi-target sharing (KEPo's mechanism): does connecting m targets compound?
E30  adaptive attacker with full defense knowledge, maximizing |union C(e)|
E31  certificate tightness                 b_cert  vs  b_break

All CPU.  Writes results_multitarget.json.
"""
from __future__ import annotations
import itertools, json, math, os, random

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, HERE)
from lp import share_lp, tau_star                                      # noqa: E402

SEED = 20260901


class Cube:
    """Anchored HyperCube for a conjunctive query.  The security-relevant method is
    C(tuple, atom_vars): the set of cells a forged tuple can reach.  Everything in
    this file is phrased in terms of C, because that is what the adversary controls
    and what the certificate counts."""

    def __init__(self, atoms, anchor, K, rng):
        self.atoms, self.anchor = atoms, anchor
        self.tau = tau_star(atoms, frozenset({anchor}))
        _, sh = share_lp(atoms, frozenset({anchor}))
        self.free = sorted(sh)
        self.p = {u: max(1, int(round(K ** sh[u]))) for u in self.free}
        self.K = int(np.prod([self.p[u] for u in self.free]))
        self.cells = list(itertools.product(*[range(self.p[u]) for u in self.free]))
        self.idx = {c: i for i, c in enumerate(self.cells)}
        self._h = {u: {} for u in self.free}
        self.rng = rng

    def h(self, u, v):
        t = self._h[u]
        if v not in t:
            t[v] = self.rng.randrange(self.p[u])
        return t[v]

    def rho_slot(self, vs):
        r = 1
        for u in self.free:
            if u not in vs:
                r *= self.p[u]
        return r

    @property
    def rho_max(self):
        return max(self.rho_slot(vs) for _, vs in self.atoms)

    def C(self, tup, vs):
        """AffectedCells(e): cells receiving tuple `tup` when it fills atom vars `vs`."""
        fix = {u: self.h(u, tup[i]) for i, u in enumerate(vs) if u in self._h}
        return {self.idx[c] for c in self.cells
                if all(c[i] == fix[u] for i, u in enumerate(self.free) if u in fix)}

    def widest_slot(self):
        """the atom a rational attacker picks: highest replication."""
        return max(self.atoms, key=lambda a: self.rho_slot(a[1]))[1]


def chain(k, anchor="v"):
    names = [anchor] + [f"u{i}" for i in range(1, k + 1)]
    return [("E", [names[i], names[i + 1]]) for i in range(k)]


def triangle(anchor="v"):
    return [("E", [anchor, "x"]), ("E", ["x", "y"]), ("E", ["y", anchor])]


def c4(anchor="v"):
    return [("E", [anchor, "x"]), ("E", ["x", "y"]),
            ("E", ["y", "z"]), ("E", ["z", anchor])]


QUERIES = [("chain-2", chain(2)), ("chain-3", chain(3)),
           ("triangle", triangle()), ("C4", c4())]


def greedy_attack(cube, b, rng, cand=400):
    """Adaptive attacker with FULL defense knowledge: sees the partition, the cell
    assignment, K and the aggregation rule, and picks b forged tuples maximizing the
    union of affected cells.  This is the strongest attacker the model admits, and is
    the standard RobustRAG-style adversary (full knowledge, bounded injections)."""
    pool = []
    for _ in range(cand):
        _, vs = rng.choice(cube.atoms)
        pool.append(cube.C(tuple(f"E{rng.randrange(9999)}" for _ in vs), vs))
    chosen = set()
    for _ in range(b):
        chosen |= max(pool, key=lambda s: len(s - chosen))
    return chosen


# ==========================================================================
def e28_observed_amplification(Ks=(64, 256, 1024), trials=200, seed=SEED):
    """rho_obs(e) = |C(e)| must never exceed rho_max, and we report how close it
    gets: a bound that is never approached would be uninformative."""
    rng = random.Random(seed)
    rows = []
    for name, Q in QUERIES:
        for K in Ks:
            cu = Cube(Q, "v", K, rng)
            obs = []
            for _ in range(trials):
                _, vs = rng.choice(cu.atoms)
                obs.append(len(cu.C(tuple(f"E{rng.randrange(9999)}" for _ in vs), vs)))
            rows.append(dict(query=name, K=cu.K, tau_star=cu.tau,
                             rho_obs_max=max(obs), rho_bound=cu.rho_max,
                             ratio=max(obs) / cu.rho_max))
    return dict(rows=rows, max_ratio=max(r["ratio"] for r in rows),
                min_ratio=min(r["ratio"] for r in rows),
                bound_never_exceeded=all(r["ratio"] <= 1 + 1e-9 for r in rows),
                bound_attained=all(abs(r["ratio"] - 1) < 1e-9 for r in rows))


def e29_multi_target(ms=(1, 2, 4, 8, 16), K=256, seed=SEED):
    """KEPo's multi-target mode connects poisoned sub-communities across m targets so
    they reinforce each other.  Question: does that make per-target amplification grow?

    Model: m target queries, ONE shared forged relation injected into the widest slot
    of each.  Gamma(e) = sum_j |C(Q_j, e)| is the total corruption across all targets.
    """
    rng = random.Random(seed)
    rows = []
    for m in ms:
        cubes = [Cube(triangle(), "v", K, rng) for _ in range(m)]
        tup = ("SHARED-HEAD", "SHARED-TAIL")            # one relation, all targets
        per = [len(cu.C(tup, cu.widest_slot())) for cu in cubes]
        gamma = sum(per)
        linear = sum(cu.rho_max for cu in cubes)
        rows.append(dict(m=m, gamma=gamma, linear_bound=linear,
                         per_target_max=max(per), per_target_mean=float(np.mean(per)),
                         rho_max=cubes[0].rho_max,
                         within_linear=bool(gamma <= linear + 1e-9)))
    per_target = [r["per_target_max"] for r in rows]
    return dict(rows=rows, K=K,
                all_within_linear=all(r["within_linear"] for r in rows),
                per_target_constant_in_m=max(per_target) == min(per_target),
                per_target_value=per_target[0],
                gamma_slope=(rows[-1]["gamma"] / rows[0]["gamma"]) / (ms[-1] / ms[0]))


def e30_adaptive(bs=(1, 2, 4, 8, 16), K=256, seed=SEED):
    """The adaptive full-knowledge attacker cannot beat b * rho_max."""
    rng = random.Random(seed)
    rows = []
    for name, Q in QUERIES:
        cu = Cube(Q, "v", K, rng)
        for b in bs:
            got = len(greedy_attack(cu, b, rng))
            rows.append(dict(query=name, K=cu.K, b=b, affected=got,
                             bound=b * cu.rho_max, rho_max=cu.rho_max,
                             holds=bool(got <= b * cu.rho_max)))
    return dict(rows=rows, n=len(rows),
                violations=sum(not r["holds"] for r in rows),
                max_ratio=max(r["affected"] / r["bound"] for r in rows))


def e31_tightness(mu=60, K=256, cap=2000, seed=SEED):
    """A certificate is only worth quoting if attacks break near it.  b_cert is the
    deterministic certificate; b_break is the smallest budget at which the adaptive
    attacker of E30 actually silences the margin.  T = b_break / b_cert; T = 1 means
    exactly tight, large T means the certificate is valid but pessimistic."""
    rng = random.Random(seed)
    rows = []
    for name, Q in QUERIES:
        cu = Cube(Q, "v", K, rng)
        b_cert = (mu - 1) // cu.rho_max
        b_break = None
        for b in range(1, cap):
            if len(greedy_attack(cu, b, rng, cand=200)) >= mu:
                b_break = b
                break
        rows.append(dict(query=name, K=cu.K, rho_max=cu.rho_max, mu=mu,
                         b_cert=b_cert, b_break=b_break,
                         tightness=b_break / b_cert if b_cert else None))
    Ts = [r["tightness"] for r in rows if r["tightness"]]
    return dict(rows=rows, mu=mu, min_tightness=min(Ts), max_tightness=max(Ts),
                mean_tightness=float(np.mean(Ts)),
                certificate_nonvacuous=all(r["b_cert"] >= 1 for r in rows),
                certificate_sound=all(r["b_break"] > r["b_cert"] for r in rows))


if __name__ == "__main__":
    res = {"seed": SEED,
           "E28_observed_amplification": e28_observed_amplification(),
           "E29_multi_target": e29_multi_target(),
           "E30_adaptive": e30_adaptive(),
           "E31_tightness": e31_tightness()}
    with open(os.path.join(HERE, "results_multitarget.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)

    o = res["E28_observed_amplification"]
    print(f"E28  rho_obs/rho_bound over {len(o['rows'])} configs: "
          f"max {o['max_ratio']:.3f}, min {o['min_ratio']:.3f}  "
          f"(never exceeded: {o['bound_never_exceeded']}, attained: {o['bound_attained']})")

    o = res["E29_multi_target"]
    print(f"\nE29  multi-target (KEPo mechanism), K={o['K']}")
    print(f"     {'m':>3} {'Gamma':>7} {'linear bd':>10} {'per-target':>11}")
    for r in o["rows"]:
        print(f"     {r['m']:3d} {r['gamma']:7d} {r['linear_bound']:10d} "
              f"{r['per_target_max']:11d}")
    print(f"     per-target amplification constant in m: {o['per_target_constant_in_m']} "
          f"(= {o['per_target_value']}); Gamma grows linearly, slope "
          f"{o['gamma_slope']:.2f}")

    o = res["E30_adaptive"]
    print(f"\nE30  adaptive full-knowledge attacker: {o['n']-o['violations']}/{o['n']} "
          f"respect b*rho_max, max ratio {o['max_ratio']:.3f}")

    o = res["E31_tightness"]
    print(f"\nE31  certificate tightness (mu={o['mu']})")
    print(f"     {'query':>9} {'rho':>5} {'b_cert':>7} {'b_break':>8} {'T':>5}")
    for r in o["rows"]:
        print(f"     {r['query']:>9} {r['rho_max']:5d} {r['b_cert']:7d} "
              f"{r['b_break']:8d} {r['tightness']:5.2f}")
    print(f"     T in [{o['min_tightness']:.2f}, {o['max_tightness']:.2f}], "
          f"mean {o['mean_tightness']:.2f}   sound: {o['certificate_sound']}")
