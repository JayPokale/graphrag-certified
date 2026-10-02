"""witness_packing.py -- E32/E33: what the adversary can reach, and isolation at
witness granularity.

E32  rho_F   the largest number of FIRING cells one forged content can reach, against
             rho_max, the number of cells it occupies.  The certificate needs the first;
             the paper bounds the second.
E33  three certificates on the same instance:
       A  HyperCube + majority          floor((c1 - K/2 - 1)/rho_max)
       B  HyperCube + recentred + rho_F floor((c1 - ceil(c1/2))/rho_F)
       C  witness packing               ceil(omega/2) - 1   with rho = 1, K = omega
     omega is the MAXIMUM edge-disjoint family (an exact set-packing ILP; Remark
     maximum says why the selection rule must be maximum); the greedy family is kept
     for comparison.
E36  the duplicate attack.  A forged copy of an honest relation e -- the same two
     endpoints, a poisoned provenance document -- is routed exactly as e under A1,
     whether or not the hash is private, so its cell-set is place[e] and it reaches
     every firing cell of every witness through e.  Recorded per anchor:
       dup_blind   |place[e*] & F| for e* the honest edge in the most witnesses,
                   a choice that needs the graph and not the hash
       dup_wb      max_e |place[e] & F| over honest witness edges (sees the hash)
       dup_min     min_e |place[e] & F| over honest witness edges (>= 1 always)
       fresh_exp   rho_max * c1 / K, what a FRESH content reaches in expectation
                   under a private hash (the model of adv-blind and App. escapes)

Runs on PEGASE when pandapower is installed and on a degree-matched mesh otherwise;
the record says which.  No GPU.
"""
from __future__ import annotations

import itertools
import json
import math
import os
import random
from collections import Counter, defaultdict

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


def max_packing(emb):
    """Exact maximum edge-disjoint family: a set-packing ILP over the witnesses.
    Instances are small (tens to hundreds of witnesses per anchor)."""
    if not emb:
        return 0
    from scipy.optimize import Bounds, LinearConstraint, milp
    edges = sorted({e for et in emb for e in et})
    idx = {e: i for i, e in enumerate(edges)}
    A = np.zeros((len(edges), len(emb)))
    for j, et in enumerate(emb):
        for e in set(et):
            A[idx[e], j] = 1.0
    res = milp(c=-np.ones(len(emb)), constraints=LinearConstraint(A, -np.inf, 1.0),
               integrality=np.ones(len(emb)), bounds=Bounds(0.0, 1.0))
    if res.status != 0:
        raise RuntimeError("set-packing ILP did not solve: %s" % res.message)
    return int(round(-res.fun))


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
    return place, h                              # h: the draw, so the anchor's own vector is known


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
            place, place_h = hypercube(emb, atoms, U, p, ui, v, random.Random(v * 7919 + K))
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
            # strongest reading: the adversary picks its content freely.  A content is an
            # edge (a, b); a fills a slot position with ITS OWN hash vector and b with
            # its own, and on an undirected host both orientations fill every slot the
            # slot's variables allow, so the cell-set is a union over slots and
            # orientations of slices pinned by two INDEPENDENT hash vectors.  Either
            # endpoint may also be the anchor itself, whose hash vector is fixed by the
            # draw.  An earlier version enumerated a single vector g, which under-counts
            # whenever one content fills two slots in different positions (Prop. rhoF).
            rho_F = 0
            gv = tuple(place_h[u].get(v, -1) for u in U)          # the anchor's own vector
            vecs = [None] + list(itertools.product(*[range(p[u]) for u in U]))
            for ga in vecs:
                for gb in vecs:
                    hit = set()
                    for _, vs in atoms:
                        for (xa, xb) in ((ga, gb), (gb, ga)):
                            fixed, ok = {}, True
                            for var, val in zip(vs, (xa, xb)):
                                if var == "v":
                                    ok &= (val is None)          # only the anchor fills v
                                else:
                                    vec = gv if val is None else val
                                    fixed[ui[var]] = vec[ui[var]]
                            if ok:
                                hit |= {c for c in F if all(c[i] == q for i, q in fixed.items())}
                    rho_F = max(rho_F, len(hit))
            omega_greedy = len(greedy_packing(emb))
            omega = max_packing(emb)
            # E36: duplicates of honest witness edges, routed as those edges are
            wit_edges = Counter(e for et in emb for e in set(et))
            hits = {e: len(place[e] & F) for e in wit_edges}
            e_star = max(wit_edges, key=lambda e: (wit_edges[e], e))
            rows.append(dict(
                anchor=v, deg=len(adj[v]), n_emb=len(emb), c1=c1,
                rho_max=rho_max, rho_F=rho_F, omega=omega, omega_greedy=omega_greedy,
                b_paper=max(0, (c1 - Kreal // 2 - 1) // rho_max) if c1 > Kreal // 2 else 0,
                b_tight=max(0, (c1 - math.ceil(c1 / 2)) // max(rho_F, 1)),
                b_witness=max(0, math.ceil(omega / 2) - 1),
                b_witness_greedy=max(0, math.ceil(omega_greedy / 2) - 1),
                dup_blind=hits[e_star], dup_wb=max(hits.values()), dup_min=min(hits.values()),
                dup_star_witnesses=wit_edges[e_star],
                fresh_exp=rho_max * c1 / Kreal))
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
            med_omega=med("omega"), med_omega_greedy=med("omega_greedy"),
            greedy_is_maximum=sum(1 for r in rows if r["omega_greedy"] == r["omega"]),
            med_b_paper=med("b_paper"), med_b_tight=med("b_tight"),
            med_b_witness=med("b_witness"), med_b_witness_greedy=med("b_witness_greedy"),
            paper_vacuous_anchors=sum(1 for r in rows if r["b_paper"] == 0),
            witness_beats_paper=sum(1 for r in rows if r["b_witness"] > r["b_paper"]),
            rho_F_eq_rho_max=sum(1 for r in rows if r["rho_F"] == r["rho_max"]),
            # E36
            med_dup_blind=med("dup_blind"), med_dup_wb=med("dup_wb"),
            med_fresh_exp=med("fresh_exp"),
            dup_blind_over_rho_F=float(np.median([r["dup_blind"] / r["rho_F"] for r in rows])),
            dup_wb_eq_rho_F=sum(1 for r in rows if r["dup_wb"] == r["rho_F"]),
            dup_min_at_least_1=all(r["dup_min"] >= 1 for r in rows),
            dup_blind_over_fresh=float(np.median([r["dup_blind"] / max(r["fresh_exp"], 1e-9)
                                                  for r in rows])),
            dup_blind_frac_of_c1=float(np.median([r["dup_blind"] / r["c1"] for r in rows])),
            dup_wb_le_rho_max=all(r["dup_wb"] <= r["rho_max"] for r in rows),
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
        print("    K   emb   c1  rho_max  rho_F   A:paper  B:tightened  C:witness (omega, greedy)"
              "  | dup: blind   wb  fresh_exp")
        for r in o["records"]:
            print("  %4d %5.0f %4.0f    %5.0f  %5.0f    %6.0f     %6.0f      %6.0f (%.0f, %.0f)"
                  "  |     %5.1f %5.1f  %7.2f"
                  % (r["K"], r["med_n_emb"], r["med_c1"], r["max_rho_max"], r["med_rho_F"],
                     r["med_b_paper"], r["med_b_tight"], r["med_b_witness"], r["med_omega"],
                     r["med_omega_greedy"], r["med_dup_blind"], r["med_dup_wb"],
                     r["med_fresh_exp"]))
