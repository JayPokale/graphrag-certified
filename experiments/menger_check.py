"""menger_check.py -- E34/E35: the fractional budget, and the flow that computes it.

E34  nu*(H) by LP against omega(H) by greedy: how much the integral design leaves on the
     table, i.e. how much the fractional witness partition of Thm nu-star recovers.
E35  Menger: for a 2-hop answer, the number of edge-disjoint witnesses must equal the number
     of common neighbours, so the certified budget is a max-flow rather than an enumeration.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.optimize import linprog

from realdata import MOTIFS_RD2, enumerate_embeddings
from witness_packing import greedy_packing, load_host

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7


def nu_star(emb):
    """max sum_W y_W  s.t.  sum_{W ni e} y_W <= 1 for every tuple e,  y >= 0."""
    edges = sorted({e for et in emb for e in et})
    idx = {e: i for i, e in enumerate(edges)}
    A = np.zeros((len(edges), len(emb)))
    for j, et in enumerate(emb):
        for e in set(et):
            A[idx[e], j] = 1.0
    r = linprog(c=-np.ones(len(emb)), A_ub=A, b_ub=np.ones(len(edges)),
                bounds=[(0, None)] * len(emb), method="highs")
    assert r.success, r.message
    return float(-r.fun)


def e34(adj, anchors):
    rows = []
    for motif in ("2-path (v-x-y)", "C4 (v-x-y-z)"):
        for v in anchors:
            emb = enumerate_embeddings(adj, v, motif)
            if len(emb) < 4:
                continue
            nu, om = nu_star(emb), len(greedy_packing(emb))
            rows.append(dict(motif=motif, anchor=v, n_emb=len(emb),
                             nu_star=nu, omega=om, gap=nu / om if om else None,
                             b_frac=int(nu // 2), b_int=max(0, -(-om // 2) - 1)))
    out = {}
    for motif in ("2-path (v-x-y)", "C4 (v-x-y-z)"):
        rs = [r for r in rows if r["motif"] == motif]
        out[motif] = dict(
            n_anchors=len(rs),
            med_nu=float(np.median([r["nu_star"] for r in rs])),
            med_omega=float(np.median([r["omega"] for r in rs])),
            max_gap=max(r["gap"] for r in rs),
            med_gap=float(np.median([r["gap"] for r in rs])),
            n_strict=sum(1 for r in rs if r["gap"] > 1 + 1e-9),
            med_b_frac=float(np.median([r["b_frac"] for r in rs])),
            med_b_int=float(np.median([r["b_int"] for r in rs])),
            nu_never_below_omega=all(r["nu_star"] >= r["omega"] - 1e-9 for r in rs))
    return dict(rows=rows[:10], summary=out)


def e35(adj, anchors):
    """For each 2-hop answer, edge-disjoint witnesses must equal the common-neighbour count."""
    checked = mism = 0
    per = []
    for v in anchors:
        emb = enumerate_embeddings(adj, v, "2-path (v-x-y)")
        by_ans = defaultdict(list)
        for et in emb:
            (a1, b1), (a2, b2) = et
            y = b2 if a2 in (a1, b1) else a2
            by_ans[y].append(et)
        for y, ets in by_ans.items():
            omega_a = len(greedy_packing(ets))
            menger = len(adj[v] & adj[y])
            checked += 1
            mism += (omega_a != menger)
            per.append((len(ets), omega_a, menger))
    return dict(n_answers=checked, n_mismatch=mism,
                max_omega=max(p[1] for p in per),
                med_omega=float(np.median([p[1] for p in per])),
                menger_exact=(mism == 0))


if __name__ == "__main__":
    adj, host = load_host()
    rng = random.Random(SEED)
    cand = sorted(adj, key=lambda u: -len(adj[u]))
    anchors = cand[:12] + rng.sample(cand[12:], 8)
    out = dict(seed=SEED, host=host, E34_fractional=e34(adj, anchors),
               E35_menger=e35(adj, anchors))
    with open(os.path.join(HERE, "results_menger.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    for motif, o in out["E34_fractional"]["summary"].items():
        print("E34 %-16s nu* med %.2f  omega med %.0f  gap med %.3f max %.3f  strict %d/%d"
              % (motif, o["med_nu"], o["med_omega"], o["med_gap"], o["max_gap"],
                 o["n_strict"], o["n_anchors"]))
    m = out["E35_menger"]
    print("E35 Menger: %d answers, %d mismatches, omega med %.0f max %d"
          % (m["n_answers"], m["n_mismatch"], m["med_omega"], m["max_omega"]))
