"""nu_check.py -- the integral packing behind Theorem packing(ii).

Theorem packing(ii) is proved with the INTEGRAL packing

    nu(Q,A) = max { |P| : atoms in P have pairwise disjoint free-variable supports }

while part (i) achieves the exponent 1 - 1/tau*_A with the FRACTIONAL one.  The bound is
therefore tight exactly where nu == tau*_A.  This checks that on every query the paper
evaluates: the 15 motif families of E12 and the 20 multi-relation TPC-H queries of RD1,
anchored as in the paper's table (and unanchored, recorded alongside).

    python3 nu_check.py        exits non-zero if any family has nu < tau*_A
"""
from __future__ import annotations

import itertools
import json
import os
import sys

from lp import tau_star
import packing as P
import realdata as RD

HERE = os.path.dirname(os.path.abspath(__file__))


def nu_int(atoms, bound=frozenset()):
    """Largest set of atoms with pairwise disjoint free-variable supports."""
    sup = [frozenset(v for v in vs if v not in bound) for _, vs in atoms]
    fully_bound = sum(1 for s in sup if not s)
    sup = [s for s in sup if s]
    for k in range(len(sup), 0, -1):
        if any(all(sup[i].isdisjoint(sup[j]) for i, j in itertools.combinations(c, 2))
               for c in itertools.combinations(range(len(sup)), k)):
            return k + fully_bound
    return fully_bound


def _atoms(edges):
    return [("R", [a, b]) for a, b in edges]


FAMILIES = {
    "2-path": P.path(2), "3-path": P.path(3), "4-path": P.path(4),
    "C3": P.cycle(3), "C4": P.cycle(4), "C5": P.cycle(5), "C6": P.cycle(6),
    "K1,3": P.star(3), "K1,4": P.star(4), "K1,5": P.star(5),
    "scatter b=2": P.scatter_gather(2), "scatter b=3": P.scatter_gather(3),
    "bipartite 2,2": P.bipartite_cluster(2, 2), "bipartite 2,3": P.bipartite_cluster(2, 3),
    "theta k=5": P.theta(5),
}

rows, bad = [], 0
for name, edges in FAMILIES.items():
    atoms = _atoms(edges)
    tau, nu = tau_star(atoms, frozenset({"v"})), nu_int(atoms, frozenset({"v"}))
    bad += tau - nu > 1e-9
    rows.append(dict(query=name, kind="motif", d=len(atoms), tau_star=tau, nu=nu))

for q, tables in RD.TPCH_QUERIES.items():
    atoms = RD.query_atoms(tables)
    if len(atoms) < 2:
        continue
    # the paper's Table tpc-h quotes the ANCHORED value (constant predicates pinned), so
    # that is the one the theorem must be tight on; the unanchored one is kept beside it
    A = frozenset(RD.TPCH_ANCHORS.get(q, set()))
    tau, nu = tau_star(atoms, A), nu_int(atoms, A)
    tau0, nu0 = tau_star(atoms, frozenset()), nu_int(atoms, frozenset())
    bad += (tau - nu > 1e-9) + (tau0 - nu0 > 1e-9)
    rows.append(dict(query=q, kind="tpch", d=len(atoms), anchor=sorted(A), tau_star=tau, nu=nu,
                     tau_star_unanchored=tau0, nu_unanchored=nu0))

out = dict(n_cases=len(rows), n_tight=sum(1 for r in rows if abs(r["tau_star"] - r["nu"]) < 1e-9),
           n_slack=bad, rows=rows)
with open(os.path.join(HERE, "results_nu.json"), "w") as f:
    json.dump(out, f, indent=2)

for r in rows:
    print("  %-14s %-6s d=%d  tau*_A=%.3f  nu=%d%s"
          % (r["query"], r["kind"], r["d"], r["tau_star"], r["nu"],
             "" if abs(r["tau_star"] - r["nu"]) < 1e-9 else "   SLACK"))
print("\nnu = tau*_A on %d/%d cases" % (out["n_tight"], out["n_cases"]))
sys.exit(1 if bad else 0)
