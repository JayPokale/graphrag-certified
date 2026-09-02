"""witness_packing.py -- E32/E33: what the adversary can reach, and isolation at
witness granularity.

E32  rho_F   the largest number of FIRING cells one forged content can reach, against
             rho_max, the number of cells it occupies.  The certificate needs the first;
             the paper bounds the second.
E33  three certificates on the same instance:
       A  HyperCube + majority          floor((c1 - K/2 - 1)/rho_max)
       B  HyperCube + recentred + rho_F floor((c1 - ceil(c1/2))/rho_F)
       C  witness packing               ceil(omega/2) - 1   with rho = 1, K = omega

Runs on PEGASE when pandapower is installed and on a degree-matched mesh otherwise;
the record says which.  No GPU.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
from collections import defaultdict

import networkx as nx
import numpy as np

from lp import share_lp, tau_star
from realdata import MOTIFS_RD2, enumerate_embeddings, ekey

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 7


def load_host():
    """PEGASE if available, else a mesh with the same n, m, mean and max degree."""
    try:
        from realdata import load_pegase
        _, adj = load_pegase("case2869pegase")
        return adj, "case2869pegase"
    except Exception:
        n, m, dmax, rng = 2869, 3968, 15, random.Random(SEED)
        side = int(n ** 0.5) + 1
        G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
        G.remove_nodes_from(list(G)[n:])
        for h in rng.sample(list(G), 45):
            near = [w for w in nx.single_source_shortest_path_length(G, h, cutoff=3) if w != h]
            rng.shuffle(near)
            for w in near:
                if G.degree(h) >= dmax:
                    break
                if not G.has_edge(h, w) and G.degree(w) < dmax:
                    G.add_edge(h, w)
        edges = list(G.edges())
        rng.shuffle(edges)
        for a, b in edges:
            if G.number_of_edges() <= m:
                break
            if min(G.degree(a), G.degree(b)) > 1 and min(G.degree(a), G.degree(b)) < dmax - 3:
                G.remove_edge(a, b)
        return {u: set(G[u]) for u in G}, "mesh-surrogate"


def greedy_packing(emb):
    """Maximal edge-disjoint set of witnesses.  Any edge-disjoint selection is sound;
    a larger one certifies more, and greedy is within a factor d of the maximum."""
    used, chosen = set(), []
    for et in sorted(emb, key=lambda t: len(set(t))):
        if not used & set(et):
            chosen.append(et)
            used |= set(et)
    return chosen


def hypercube(emb, atoms, U, p, ui, v, rng):
    """Cell-sets of every retrieved edge, exactly as realdata.RD2 places them."""
    h = {u: {} for u in U}

    def hv(u, x):
        if x not in h[u]:
            h[u][x] = rng.randrange(p[u])
        return h[u][x]

    place = defaultdict(set)
    for et in emb:
        for slot, e in zip(atoms, et):
            _, vs = slot
            a, b = e
            for (xa, xb) in ((a, b), (b, a)):
                fixed, ok = {}, True
                for var, val in zip(vs, (xa, xb)):
                    if var == "v":
                        ok &= (val == v)
                    else:
                        fixed[ui[var]] = hv(var, val)
                if not ok:
                    continue
                free = [i for i in range(len(U)) if i not in fixed]
                for combo in itertools.product(*[range(p[U[i]]) for i in free]):
                    c = [0] * len(U)
                    for i, q in fixed.items():
                        c[i] = q
                    for i, q in zip(free, combo):
                        c[i] = q
                    place[e].add(tuple(c))
    return place


def run(motif, Ks, adj, anchors):
    atoms = MOTIFS_RD2[motif]
    _, shares = share_lp(atoms, frozenset({"v"}))
    U = sorted(shares)
    ui = {u: i for i, u in enumerate(U)}
    supports = [[ui[x] for x in vs if x in ui] for _, vs in atoms]
    recs = []
    for K in Ks:
        p = {u: max(1, int(round(K ** shares[u]))) for u in U}
        Kreal = int(np.prod([p[u] for u in U]))
        rows = []
        for v in anchors:
            emb = enumerate_embeddings(adj, v, motif)
            if len(emb) < len(atoms):
                continue
            place = hypercube(emb, atoms, U, p, ui, v, random.Random(v * 7919 + K))
            F = set()
            for et in emb:
                inter = None
                for e in et:
                    inter = place[e] if inter is None else inter & place[e]
                    if not inter:
                        break
                if inter:
                    F |= inter
            c1 = len(F)
            rho_max = max(len(place[e]) for e in place)
            # strongest reading: the adversary picks its content's hash vector g freely,
            # and its cell-set is the union of the slices of every slot it can fill
            rho_F = 0
            for g in itertools.product(*[range(p[u]) for u in U]):
                hit = set()
                for S in supports:
                    if S:
                        hit |= {c for c in F if all(c[i] == g[i] for i in S)}
                rho_F = max(rho_F, len(hit))
            omega = len(greedy_packing(emb))
            rows.append(dict(
                anchor=v, deg=len(adj[v]), n_emb=len(emb), c1=c1,
                rho_max=rho_max, rho_F=rho_F, omega=omega,
                b_paper=max(0, (c1 - Kreal // 2 - 1) // rho_max) if c1 > Kreal // 2 else 0,
                b_tight=max(0, (c1 - math.ceil(c1 / 2)) // max(rho_F, 1)),
                b_witness=max(0, math.ceil(omega / 2) - 1)))
        if not rows:
            continue
        med = lambda k: float(np.median([r[k] for r in rows]))
        recs.append(dict(
            K=Kreal, n_anchors=len(rows),
            med_n_emb=med("n_emb"), med_c1=med("c1"),
            max_rho_max=max(r["rho_max"] for r in rows), med_rho_F=med("rho_F"),
            max_ratio_rho=max(r["rho_max"] / max(r["rho_F"], 1) for r in rows),
            min_ratio_rho=min(r["rho_max"] / max(r["rho_F"], 1) for r in rows),
            rho_F_never_exceeds_rho_max=all(r["rho_F"] <= r["rho_max"] for r in rows),
            med_omega=med("omega"),
            med_b_paper=med("b_paper"), med_b_tight=med("b_tight"),
            med_b_witness=med("b_witness"),
            paper_vacuous_anchors=sum(1 for r in rows if r["b_paper"] == 0),
            witness_beats_paper=sum(1 for r in rows if r["b_witness"] > r["b_paper"]),
            rows=rows[:8]))
    return dict(motif=motif, d=len(atoms), tau_star_A=tau_star(atoms, frozenset({"v"})),
                shares=shares, records=recs)


if __name__ == "__main__":
    adj, host = load_host()
    degs = [len(adj[u]) for u in adj]
    rng = random.Random(SEED)
    cand = sorted(adj, key=lambda u: -len(adj[u]))
    anchors = cand[:12] + rng.sample(cand[12:], 8)
    out = dict(seed=SEED, host=host, n_nodes=len(adj),
               deg_mean=float(np.mean(degs)), deg_max=int(max(degs)),
               E32_E33_2path=run("2-path (v-x-y)", (16, 64, 256), adj, anchors),
               E32_E33_C4=run("C4 (v-x-y-z)", (16, 64), adj, anchors))
    with open(os.path.join(HERE, "results_witness.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)

    print("host %s  n=%d  deg mean %.2f max %d" % (host, out["n_nodes"],
                                                   out["deg_mean"], out["deg_max"]))
    for key in ("E32_E33_2path", "E32_E33_C4"):
        o = out[key]
        print("\n%s  d=%d tau*_A=%g" % (o["motif"], o["d"], o["tau_star_A"]))
        print("    K   emb   c1  rho_max  rho_F   A:paper  B:tightened  C:witness (omega)")
        for r in o["records"]:
            print("  %4d %5.0f %4.0f    %5.0f  %5.0f    %6.0f     %6.0f      %6.0f (%.0f)"
                  % (r["K"], r["med_n_emb"], r["med_c1"], r["max_rho_max"], r["med_rho_F"],
                     r["med_b_paper"], r["med_b_tight"], r["med_b_witness"], r["med_omega"]))
