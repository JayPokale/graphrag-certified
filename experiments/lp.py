"""
lp.py -- the two linear programs the whole paper rests on.

    tau_star(atoms, bound)  anchored fractional edge packing number  tau*_A(Q)
    share_lp(atoms, bound)  the anchored-HyperCube share allocation; value = 1/tau*_A

An `atom` is a pair (name, [variables]); `bound` is the set of anchored variables.
A graph motif is the special case where every atom has two variables.

    >>> triangle = [("R", ["v", "x"]), ("R", ["x", "y"]), ("R", ["y", "v"])]
    >>> round(tau_star(triangle, {"v"}), 6)
    2.0
    >>> round(share_lp(triangle, {"v"})[0], 6)
    0.5
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

__all__ = ["tau_star", "share_lp"]


def tau_star(atoms, bound=frozenset()):
    """max sum_R x_R  s.t.  sum_{R ni u} x_R <= 1 for every FREE variable u, 0 <= x <= 1.

    The x <= 1 bound is implied whenever every atom has a free variable; it is stated
    so that fully-bound atoms cannot make the LP unbounded.
    """
    U = sorted({v for _, vs in atoms for v in vs if v not in bound})
    A = np.zeros((len(U), len(atoms)))
    for j, (_, vs) in enumerate(atoms):
        for i, u in enumerate(U):
            if u in vs:
                A[i, j] = 1.0
    r = linprog(c=-np.ones(len(atoms)),
                A_ub=A if len(U) else None,
                b_ub=np.ones(len(U)) if len(U) else None,
                bounds=[(0, 1)] * len(atoms), method="highs")
    assert r.success, r.message
    return float(-r.fun)


def share_lp(atoms, bound=frozenset()):
    """max_a min_R sum_{u in R, u free} a_u  s.t. sum a = 1, a >= 0.

    Returns (value, shares).  By LP duality value == 1 / tau_star(atoms, bound), so the
    HyperCube's maximum replication exponent is 1 - 1/tau*_A.
    """
    U = sorted({v for _, vs in atoms for v in vs if v not in bound})
    n = len(U)
    idx = {u: i for i, u in enumerate(U)}
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A_ub, b_ub = [], []
    for _, vs in atoms:
        row = np.zeros(n + 1)
        for v in vs:
            if v in idx:
                row[idx[v]] = -1.0
        row[-1] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1.0
    r = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                A_eq=A_eq, b_eq=[1.0],
                bounds=[(0, None)] * n + [(None, None)], method="highs")
    assert r.success, r.message
    return float(r.x[-1]), {u: float(r.x[idx[u]]) for u in U}
