"""kg_claims.py -- CLAIM-CHECK: how much corroboration a claim has under a declared support class.

kg_omega.py scores single-template questions, where the support class is one typed path.
Claim verification declares a richer class.  Here the claim is "compound c treats disease d"
on Hetionet v1.0, the direct CtD edge is removed, and the support class is the union of
seven metapaths (CbGaD, CuGuD, CdGdD, CrCtD, CtDrD, CpDrD, CiPCiCtD).  For every one of
the 755 CtD claims, and for 755 random compound--disease pairs with no CtD edge, we compute
over the witness hypergraph H (vertices = relations, hyperedges = witnesses):

  omega  maximum edge-disjoint witness family     (exact, set-packing ILP)
  nu*    fractional witness packing              (LP)
  tau    minimum transversal, the corroboration of Thm limit   (exact, hitting-set ILP)

and report the share of claims each rule certifies at t forged relations: the transversal
rule of Thm limit certifies t against an add-only adversary iff tau >= t+1, and against a
neutralising one iff tau >= 2t+1; witness packing certifies t iff omega >= 2t+1.

Writes results_kgclaims.json.  CPU only; reads experiments/data/ (fetch_data.sh).
"""
from __future__ import annotations

import gzip
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEED = 7
CAP = 4000                       # witnesses enumerated per claim (recorded if hit)
UND = {"CrC", "DrD", "GiG", "GcG"}
PATTERNS = {
    "CbGaD": [("CbG", 1), ("DaG", -1)],
    "CuGuD": [("CuG", 1), ("DuG", -1)],
    "CdGdD": [("CdG", 1), ("DdG", -1)],
    "CrCtD": [("CrC", 1), ("CtD", 1)],
    "CtDrD": [("CtD", 1), ("DrD", 1)],
    "CpDrD": [("CpD", 1), ("DrD", 1)],
    "CiPCiCtD": [("PCiC", -1), ("PCiC", 1), ("CtD", 1)],
}


def load():
    adj, kinds, ctd = defaultdict(set), {}, []
    with open(os.path.join(DATA, "hetionet-nodes.tsv")) as f:
        next(f)
        for line in f:
            nid, _, k = line.rstrip("\n").split("\t")
            kinds[nid] = k
    with gzip.open(os.path.join(DATA, "hetionet-edges.sif.gz"), "rt") as f:
        next(f)
        for line in f:
            s, r, o = line.rstrip("\n").split("\t")
            adj[(s, r, 1)].add(o)
            adj[(o, r, -1)].add(s)
            if r in UND:
                adj[(s, r, -1)].add(o)
                adj[(o, r, 1)].add(s)
            if r == "CtD":
                ctd.append((s, o))
    return adj, kinds, ctd


def _triple(x, r, d, y):
    if r in UND:
        return (min(x, y), r, max(x, y))
    return (x, r, y) if d > 0 else (y, r, x)


def witnesses(adj, c, d):
    """All witnesses of 'c treats d' in the declared class, the claim edge itself excluded."""
    out, capped = set(), False
    for pat in PATTERNS.values():
        def rec(x, i, used, nodes):
            nonlocal capped
            if len(out) >= CAP:
                capped = True
                return
            r, dr = pat[i]
            for y in adj.get((x, r, dr), ()):
                last = i == len(pat) - 1
                if (last and y != d) or (not last and (y in nodes or y == d)):
                    continue
                t = _triple(x, r, dr, y)
                if t == (c, "CtD", d):
                    continue
                if last:
                    out.add(frozenset(used + [t]))
                else:
                    rec(y, i + 1, used + [t], nodes | {y})
        rec(c, 0, [], {c})
    return list(out), capped


def invariants(W):
    """(omega, nu*, tau) of the witness hypergraph."""
    if not W:
        return 0, 0.0, 0
    elems = sorted(set().union(*W))
    idx = {e: i for i, e in enumerate(elems)}
    A = np.zeros((len(elems), len(W)))
    for j, w in enumerate(W):
        for e in w:
            A[idx[e], j] = 1.0
    nu = -linprog(-np.ones(len(W)), A_ub=A, b_ub=np.ones(len(elems)),
                  bounds=(0, None), method="highs").fun
    om = milp(-np.ones(len(W)), constraints=LinearConstraint(A, -np.inf, 1),
              integrality=np.ones(len(W)), bounds=Bounds(0, 1))
    tu = milp(np.ones(len(elems)), constraints=LinearConstraint(A.T, 1, np.inf),
              integrality=np.ones(len(elems)), bounds=Bounds(0, 1))
    assert om.success and tu.success
    return int(round(-om.fun)), float(nu), int(round(tu.fun))


def summarise(rows):
    om = np.array([r["omega"] for r in rows])
    nu = np.array([r["nu_star"] for r in rows])
    tu = np.array([r["tau"] for r in rows])
    share = lambda a: round(float(a.mean()) * 100, 1)
    return dict(
        n=len(rows), med_omega=float(np.median(om)), med_tau=float(np.median(tu)),
        max_tau=int(tu.max()),
        # witness packing certifies t=1 iff omega >= 3; the transversal rule iff tau >= 3
        # against a neutralising adversary and iff tau >= 2 against an add-only one
        pct_wpack_t1=share(om >= 3), pct_neut_t1=share(tu >= 3), pct_add_t1=share(tu >= 2),
        pct_neut_t3=share(tu >= 7), pct_tau0=share(tu == 0),
        n_tau_gt_nu=int(sum(t > n + 1e-9 for t, n in zip(tu, nu))),
        nu_le_tau=bool(all(n <= t + 1e-9 for t, n in zip(tu, nu))),
        omega_le_nu=bool(all(o <= n + 1e-9 for o, n in zip(om, nu))),
        n_capped=sum(r["capped"] for r in rows))


if __name__ == "__main__":
    adj, kinds, ctd = load()
    treats = set(ctd)
    comps = sorted({c for c, _ in ctd})
    dis = sorted({d for _, d in ctd})
    rng = random.Random(SEED)
    neg = set()
    while len(neg) < len(ctd):
        p = (rng.choice(comps), rng.choice(dis))
        if p not in treats:
            neg.add(p)
    out = {}
    for name, pairs in (("true_claims", ctd), ("non_claims", sorted(neg))):
        rows = []
        for c, d in pairs:
            W, capped = witnesses(adj, c, d)
            om, nu, tu = invariants(W)
            rows.append(dict(c=c, d=d, n_w=len(W), omega=om, nu_star=nu, tau=tu,
                             capped=capped))
        out[name] = dict(summary=summarise(rows), rows=rows)
        s = out[name]["summary"]
        print("%-12s n=%d  med omega %.0f  med tau %.0f  wpack t=1 %.1f%%  neut t=1 %.1f%%  "
              "add t=1 %.1f%%  neut t=3 %.1f%%  tau>nu* on %d"
              % (name, s["n"], s["med_omega"], s["med_tau"], s["pct_wpack_t1"],
                 s["pct_neut_t1"], s["pct_add_t1"], s["pct_neut_t3"], s["n_tau_gt_nu"]))
    out.update(seed=SEED, cap=CAP, support_class=list(PATTERNS))
    with open(os.path.join(HERE, "results_kgclaims.json"), "w") as f:
        json.dump(out, f, indent=1)
